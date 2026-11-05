"""Geometry, legacy compatibility and real train/deploy pixel parity (CPU)."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Dict
import unittest

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch
from torchvision.transforms import InterpolationMode, Resize
from torchvision.transforms import functional as F

from agenticwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from agenticwam.datasets.lerobot.transforms.image import (
    LetterboxResize, ToTensor, configure_image_resize, resolve_eval_image_preproc,
)
from agenticwam.utils.config_resolvers import register_default_resolvers


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deployment_method():
    # Execute the actual method without loading model weights, text encoders or CUDA.
    tree = ast.parse((ROOT / "robotwin/deploy/deploy_policy.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "AgenticWAMCosmosRobotWinPolicy")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_build_obs_video")
    namespace = dict(torch=torch, np=np, Dict=Dict, Any=Any,
                     _SIM_CAMERA_KEYS=["head_camera", "left_camera", "right_camera"],
                     _tv_resize=F.resize, _tv_center_crop=F.center_crop)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "deploy_policy.py", "exec"), namespace)
    return namespace[method.name]


def make_processor(mode, per_camera=False):
    register_default_resolvers()
    cfg = OmegaConf.load(ROOT / "configs/real_post_train/data/custom.yaml")
    cfg.data.train.processor.image_resize_mode = mode
    if per_camera:
        for split in ("train_transforms", "val_transforms"):
            transforms = OmegaConf.to_container(cfg.data.train.processor[split], resolve=False)
            cfg.data.train.processor[split] = {key: deepcopy(transforms) for key in ("head", "left", "right")}
    return instantiate(cfg.data.train.processor)


class LetterboxGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_landscape_full_field_and_exact_padding(self):
        clip = torch.ones(9, 3, 240, 424)
        result = LetterboxResize((224, 320))(clip)
        self.assertEqual(tuple(result.shape), (9, 3, 224, 320))
        self.assertTrue(torch.allclose(result[..., 21:202, :], torch.ones(9, 3, 181, 320), atol=1e-6, rtol=0))
        self.assertEqual(torch.count_nonzero(result[..., :21, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(result[..., 202:, :]).item(), 0)

    def test_portrait_and_rounding(self):
        result = LetterboxResize((224, 320))(torch.ones(3, 424, 240))
        self.assertEqual(tuple(result.shape), (3, 224, 320))
        self.assertTrue(torch.allclose(result[..., 96:223], torch.ones(3, 224, 127), atol=1e-6, rtol=0))
        self.assertEqual(torch.count_nonzero(result[..., :96]).item(), 0)
        self.assertEqual(torch.count_nonzero(result[..., 223:]).item(), 0)

    def test_circle_preserves_geometry_and_corners_are_not_cropped(self):
        yy, xx = torch.meshgrid(torch.arange(240), torch.arange(424), indexing="ij")
        circle = (((yy - 120)**2 + (xx - 212)**2) <= 50**2).float().unsqueeze(0)
        result = LetterboxResize((224, 320), interpolation=InterpolationMode.NEAREST)(circle)
        points = torch.nonzero(result[0])
        height, width = (points.max(0).values - points.min(0).values + 1).tolist()
        self.assertLessEqual(abs(height - width), 1)
        raw = torch.rand(2, 3, 240, 424, generator=torch.Generator().manual_seed(9))
        expected = F.pad(F.resize(raw, [181, 320], antialias=True), [0, 21, 0, 22])
        self.assertTrue(torch.equal(LetterboxResize((224, 320))(raw), expected))

    def test_same_size_and_single_clip_consistency(self):
        raw = torch.rand(3, 224, 320)
        operation = LetterboxResize((224, 320))
        self.assertTrue(torch.equal(operation(raw), raw))
        self.assertTrue(torch.equal(operation(raw), operation(raw.unsqueeze(0))[0]))

    def test_invalid_resize_recipes_fail_before_training(self):
        for size in (224, (0, 320), (224,), (224.5, 320)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                LetterboxResize(size)
        for transforms in ([ToTensor()], [ToTensor(), Resize(224)], None):
            with self.assertRaises(ValueError):
                configure_image_resize(transforms, "letterbox")
        with self.assertRaises(ValueError):
            configure_image_resize([], "unknown")

    def test_old_resize_operations_are_unchanged(self):
        transforms = [ToTensor(), Resize((224, 320))]
        self.assertIs(configure_image_resize(transforms, "stretch"), transforms)
        raw = torch.randint(0, 256, (2, 3, 240, 424), dtype=torch.uint8)
        expected = F.resize(raw.float() / 255.0, [224, 320], antialias=True)
        processor = make_processor("stretch").train()
        self.assertTrue(torch.equal(processor.transform_image(raw, "head"), expected))


class TrainDeployParityTests(unittest.TestCase):
    def test_actual_deployment_method_matches_training_first_frame_in_all_views(self):
        method = deployment_method()
        raw = torch.randint(0, 256, (3, 9, 3, 240, 424), dtype=torch.uint8,
                            generator=torch.Generator().manual_seed(42))
        for mode, per_camera in (("stretch", False), ("letterbox", False), ("letterbox", True)):
            with self.subTest(mode=mode, per_camera=per_camera):
                processor = make_processor(mode, per_camera).train()
                resize = ResizeSmallestSideAspectPreserving(args={"img_h": 224, "img_w": 320})
                crop = CenterCrop(args={"img_h": 224, "img_w": 320})
                norm = Normalize(args={"mean": .5, "std": .5})
                expected = []
                for camera, clip in zip(("head", "left", "right"), raw):
                    transformed = processor.transform_image(clip, camera)
                    expected.append(norm(crop(resize(transformed)))[0])
                expected = torch.stack(expected).permute(1, 0, 2, 3).unsqueeze(0)
                policy = SimpleNamespace(processor=processor.eval(), n_cameras=3, device="cpu",
                                         _preproc_mode=resolve_eval_image_preproc(mode), _preproc_crop=.95,
                                         _resize=resize, _crop=crop, _norm=norm)
                observation = {"observation": {
                    camera: {"rgb": clip[0].permute(1, 2, 0).numpy()}
                    for camera, clip in zip(("head_camera", "left_camera", "right_camera"), raw)}}
                observed = method(policy, observation)
                self.assertTrue(torch.equal(expected, observed))
                self.assertEqual(tuple(observed.shape), (1, 3, 3, 224, 320))
                if mode == "letterbox":
                    self.assertTrue(torch.all(observed[..., :21, :] == -1))
                    self.assertTrue(torch.all(observed[..., 202:, :] == -1))

    def test_validation_geometry_matches_training(self):
        processor = make_processor("letterbox")
        raw = torch.randint(0, 256, (2, 3, 240, 424), dtype=torch.uint8)
        train = processor.train().transform_image(raw, "left")
        val = processor.eval().transform_image(raw, "left")
        self.assertTrue(torch.equal(train, val))

    def test_conflicting_server_override_is_rejected(self):
        self.assertEqual(resolve_eval_image_preproc("letterbox"), "letterbox")
        self.assertEqual(resolve_eval_image_preproc("letterbox", "auto"), "letterbox")
        self.assertEqual(resolve_eval_image_preproc("stretch"), "parity")
        self.assertEqual(resolve_eval_image_preproc("stretch", "legacy"), "legacy")
        for mode, requested in (("letterbox", "parity"), ("letterbox", "legacy"),
                                ("letterbox", "parity_crop"), ("stretch", "letterbox")):
            with self.subTest(mode=mode, requested=requested), self.assertRaises(ValueError):
                resolve_eval_image_preproc(mode, requested)


class FullDatasetLetterboxTests(unittest.TestCase):
    def test_video_changes_without_changing_actions_codes_or_sample_alignment(self):
        from test_sparse_video_decode import build_dataset, make_fixture
        with TemporaryDirectory() as tmp:
            root = make_fixture(Path(tmp))
            stretch = build_dataset(root, sparse=True, image_size=(32, 32))
            letterbox = build_dataset(root, sparse=True, image_size=(32, 32), image_resize_mode="letterbox")
            dense = build_dataset(root, sparse=False, image_size=(32, 32), image_resize_mode="letterbox")
            for index in (0, 39, 40, 51, 52, 88):
                old, new, full = stretch[index], letterbox[index], dense[index]
                self.assertFalse(torch.equal(old["video"], new["video"]))
                self.assertTrue(torch.equal(new["video"], full["video"]))
                self.assertEqual(tuple(new["video"].shape), (3, 27, 32, 32))
                for key in old:
                    if key == "video":
                        continue
                    if torch.is_tensor(old[key]):
                        self.assertTrue(torch.equal(old[key], new[key]), key)
                    else:
                        self.assertEqual(old[key], new[key], key)


class ConfigureAndInstallTests(unittest.TestCase):
    def test_configuration_preserves_comments_paths_and_line_endings(self):
        configure = load_script("configure_custom_image_preproc").configure
        original = (ROOT / "configs/real_post_train/data/custom.yaml").read_text()
        original = original.replace("coding_v3/", "coding_v4/")
        for newlines in ("\n", "\r\n"):
            with self.subTest(newlines=repr(newlines)), TemporaryDirectory() as tmp:
                config = Path(tmp) / "custom.yaml"
                before = original.replace("\n", newlines).encode()
                config.write_bytes(before)
                backup = configure(config, "letterbox")
                self.assertEqual(backup.read_bytes(), before)
                self.assertEqual(config.read_bytes(), before.replace(b"image_resize_mode: stretch", b"image_resize_mode: letterbox"))
                self.assertIsNone(configure(config, "letterbox"))
                configure(config, "stretch")
                self.assertEqual(config.read_bytes(), before)

    def test_configuration_also_supports_existing_yaml_without_new_fields(self):
        configure = load_script("configure_custom_image_preproc").configure
        raw = (ROOT / "configs/real_post_train/data/custom.yaml").read_text()
        raw = "\n".join(line for line in raw.split("\n") if "image_resize_mode:" not in line)
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "custom.yaml"
            config.write_text(raw)
            configure(config, "letterbox")
            result = OmegaConf.load(config)
            self.assertEqual(result.data.train.processor.image_resize_mode, "letterbox")
            self.assertEqual(result.data.val.processor.image_resize_mode, "letterbox")
            self.assertEqual(config.read_text().replace("      image_resize_mode: letterbox\n", ""), raw)

    def test_verified_installer_preserves_local_yaml_and_launch(self):
        module = load_script("install_image_preproc_patch")
        with TemporaryDirectory() as tmp:
            bundle, project = Path(tmp) / "bundle", Path(tmp) / "project"
            (project / "src/agenticwam").mkdir(parents=True)
            launch = project / "train.sh"
            launch.write_text("local launch")
            config = project / "configs/real_post_train/data/custom.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("local coding_v4 and custom prompt paths")
            manifest = {}
            for name in module.SOURCE_FILES:
                source = bundle / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("new " + name)
                manifest[name] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            (bundle / "MANIFEST.json").write_text(json.dumps(manifest))
            existing = project / module.SOURCE_FILES[0]
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_text("old image transforms")
            (bundle / module.SOURCE_FILES[-1]).write_text("bad checksum")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                module.install(bundle, project)
            self.assertEqual(existing.read_text(), "old image transforms")
            self.assertFalse((project / ".image-preproc-backups").exists())
            (bundle / module.SOURCE_FILES[-1]).write_text("new " + module.SOURCE_FILES[-1])
            backup = module.install(bundle, project)
            self.assertEqual((backup / module.SOURCE_FILES[0]).read_text(), "old image transforms")
            self.assertEqual(launch.read_text(), "local launch")
            self.assertEqual(config.read_text(), "local coding_v4 and custom prompt paths")


if __name__ == "__main__":
    unittest.main()
