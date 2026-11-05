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
    LetterboxResize, ToTensor, configure_image_resize, resolve_eval_image_preproc, validate_video_image_size,
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


def make_processor(mode, per_camera=False, size=(224, 320), scale_up=True):
    register_default_resolvers()
    cfg = OmegaConf.load(ROOT / "configs/real_post_train/data/custom.yaml")
    cfg.data.train.processor.image_resize_mode = mode
    cfg.data.train.processor.image_letterbox_scale_up = scale_up
    cfg.data.train.video_size = list(size)
    for image in cfg.data.train.shape_meta.images:
        image.shape = [3, *size]
    for phase in ("train_transforms", "val_transforms"):
        for transform in cfg.data.train.processor[phase]:
            if str(transform._target_) == "torchvision.transforms.Resize":
                transform.size = list(size)
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

    def test_256x448_no_upscale_preserves_every_original_pixel(self):
        raw = torch.rand(9, 3, 240, 424, generator=torch.Generator().manual_seed(34))
        result = LetterboxResize((256, 448), scale_up=False)(raw)
        self.assertEqual(tuple(result.shape), (9, 3, 256, 448))
        self.assertTrue(torch.equal(result[..., 8:248, 12:436], raw))
        self.assertEqual(torch.count_nonzero(result[..., :8, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(result[..., 248:, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(result[..., :, :12]).item(), 0)
        self.assertEqual(torch.count_nonzero(result[..., :, 436:]).item(), 0)

    def test_no_upscale_still_downscales_larger_images(self):
        raw = torch.rand(2, 3, 480, 848)
        result = LetterboxResize((256, 448), scale_up=False)(raw)
        expected = F.pad(F.resize(raw, [254, 448], antialias=True), [0, 1, 0, 1])
        self.assertTrue(torch.equal(result, expected))

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
        cases = [("stretch", False, (224, 320), True), ("letterbox", False, (224, 320), True),
                 ("letterbox", True, (224, 320), True), ("letterbox", False, (256, 448), False),
                 ("letterbox", True, (256, 448), False), ("stretch", False, (256, 448), True)]
        for mode, per_camera, size, scale_up in cases:
            with self.subTest(mode=mode, per_camera=per_camera, size=size, scale_up=scale_up):
                processor = make_processor(mode, per_camera, size, scale_up).train()
                resize = ResizeSmallestSideAspectPreserving(args={"img_h": size[0], "img_w": size[1]})
                crop = CenterCrop(args={"img_h": size[0], "img_w": size[1]})
                norm = Normalize(args={"mean": .5, "std": .5})
                expected = []
                for camera, clip in zip(("head", "left", "right"), raw):
                    transformed = processor.transform_image(clip, camera)
                    expected.append(norm(crop(resize(transformed)))[0])
                expected = torch.stack(expected).permute(1, 0, 2, 3).unsqueeze(0)
                policy = SimpleNamespace(processor=processor.eval(), n_cameras=3, device="cpu", video_size=size,
                                         _preproc_mode=resolve_eval_image_preproc(mode), _preproc_crop=.95,
                                         _resize=resize, _crop=crop, _norm=norm)
                observation = {"observation": {
                    camera: {"rgb": clip[0].permute(1, 2, 0).numpy()}
                    for camera, clip in zip(("head_camera", "left_camera", "right_camera"), raw)}}
                observed = method(policy, observation)
                self.assertTrue(torch.equal(expected, observed))
                self.assertEqual(tuple(observed.shape), (1, 3, 3, *size))
                if mode == "letterbox" and size == (224, 320):
                    self.assertTrue(torch.all(observed[..., :21, :] == -1))
                    self.assertTrue(torch.all(observed[..., 202:, :] == -1))
                if mode == "letterbox" and size == (256, 448):
                    self.assertTrue(torch.all(observed[..., :8, :] == -1))
                    self.assertTrue(torch.all(observed[..., 248:, :] == -1))
                    for camera in range(3):
                        original = norm(raw[camera, 0].float() / 255.0)
                        self.assertTrue(torch.equal(observed[0, :, camera, 8:248, 12:436], original))

    def test_validation_geometry_matches_training(self):
        raw = torch.randint(0, 256, (2, 3, 240, 424), dtype=torch.uint8)
        for size, scale_up in (((224, 320), True), ((256, 448), False)):
            processor = make_processor("letterbox", size=size, scale_up=scale_up)
            train = processor.train().transform_image(raw, "left")
            val = processor.eval().transform_image(raw, "left")
            self.assertTrue(torch.equal(train, val))

    def test_video_size_validation_rejects_partial_config_updates(self):
        meta = {"images": [{"key": key, "shape": [3, 256, 448]} for key in ("head", "left", "right")]}
        self.assertEqual(validate_video_image_size([256, 448], meta), (256, 448))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_video_image_size([224, 320], meta)
        for size in ((240, 424), (256, -448), (256,), (256.5, 448)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                validate_video_image_size(size)

    def test_partial_resize_update_is_rejected_by_processor(self):
        register_default_resolvers()
        cfg = OmegaConf.load(ROOT / "configs/real_post_train/data/custom.yaml")
        cfg.data.train.processor.image_resize_mode = "letterbox"
        for image in cfg.data.train.shape_meta.images:
            image.shape = [3, 256, 448]
        with self.assertRaisesRegex(Exception, "Resize size disagrees"):
            instantiate(cfg.data.train.processor)

    def test_256x448_reference_recipe_instantiates_processor(self):
        register_default_resolvers()
        cfg = OmegaConf.load(ROOT / "configs/real_post_train/data/custom_256x448.yaml")
        for split in (cfg.data.train, cfg.data.val):
            self.assertEqual(list(split.video_size), [256, 448])
            self.assertEqual(split.processor.image_resize_mode, "letterbox")
            self.assertFalse(split.processor.image_letterbox_scale_up)
        self.assertEqual(instantiate(cfg.data.train.processor).image_resize_mode, "letterbox")

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
    def test_256x448_full_loader_keeps_non_image_fields_and_sparse_alignment(self):
        from test_sparse_video_decode import build_dataset, make_fixture
        with TemporaryDirectory() as tmp:
            root = make_fixture(Path(tmp))
            baseline = build_dataset(root, sparse=True)
            kwargs = dict(image_size=(256, 448), image_resize_mode="letterbox", image_letterbox_scale_up=False)
            sparse = build_dataset(root, sparse=True, **kwargs)
            dense = build_dataset(root, sparse=False, **kwargs)
            for index in (0, 40, 88):
                old, new, full = baseline[index], sparse[index], dense[index]
                self.assertEqual(tuple(new["video"].shape), (3, 27, 256, 448))
                self.assertTrue(torch.equal(new["video"], full["video"]))
                self.assertEqual(set(old), set(new))
                for key in old:
                    if key == "video":
                        continue
                    if torch.is_tensor(old[key]):
                        self.assertTrue(torch.equal(old[key], new[key]), key)
                    else:
                        self.assertEqual(old[key], new[key], key)

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


class ModelSpatialShapeTests(unittest.TestCase):
    def test_real_qwen_vision_and_cosmos_tokenizer_support_256x448(self):
        from dataclasses import replace
        from torch import nn
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        from agenticwam.models.cosmos_mot.code_vision import CodeVisionPrefix
        from agenticwam.models.cosmos import COSMOS_2B_CONFIG, VideoTokenizer, MultiViewRoPE

        torch.set_num_threads(1)
        # Exercise the real vision implementation with a small random model, no download.
        vision = CodeVisionPrefix.__new__(CodeVisionPrefix)
        nn.Module.__init__(vision)
        vision.vcfg = Qwen3VLVisionConfig(depth=1, hidden_size=32, intermediate_size=64,
            num_heads=4, out_hidden_size=64, patch_size=16, spatial_merge_size=2,
            temporal_patch_size=2, deepstack_visual_indexes=[])
        vision.vit = Qwen3VLVisionModel(vision.vcfg).eval().requires_grad_(False)
        vision.vit.merger.requires_grad_(True)
        vision.register_buffer("img_mean", torch.zeros(1, 3, 1, 1))
        vision.register_buffer("img_std", torch.ones(1, 3, 1, 1))
        vision.patch, vision.merge, vision.tpatch = 16, 2, 2
        prefix = vision(torch.zeros(1, 3, 3, 256, 448))
        self.assertEqual(tuple(prefix.shape), (1, 336, 64))
        prefix.square().mean().backward()
        gradients = [p.grad for p in vision.vit.merger.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))

        # Use the actual video patchifier and multiview positional encoding at /16 VAE size.
        cfg = replace(COSMOS_2B_CONFIG, in_channels=49, out_channels=48, view_embed_dim=7, max_n_cameras=4)
        tokenizer = VideoTokenizer(cfg)
        latents = torch.zeros(1, 48, 9, 16, 28)
        with torch.no_grad():
            tokens, _, shape = tokenizer.patchify(latents, torch.zeros(1, 1, 9, 16, 28), n_cameras=3)
            reconstructed = tokenizer.unpatchify(tokens, shape)
            rope = MultiViewRoPE(cfg.attention_head_dim, cfg.max_size, cfg.patch_size, rope_scale=cfg.rope_scale)
            frequencies = rope(latents, n_cameras=3)
        self.assertEqual(tuple(tokens.shape), (1, 1008, 2048))
        self.assertEqual(tuple(reconstructed.shape), tuple(latents.shape))
        for frequency in frequencies:
            self.assertEqual(frequency.shape[0], tokens.shape[1])
            self.assertTrue(torch.isfinite(frequency).all())


class ConfigureAndInstallTests(unittest.TestCase):
    def test_size_update_supports_camera_specific_transforms_and_inline_metadata(self):
        configure = load_script("configure_custom_image_preproc").configure
        cfg = OmegaConf.load(ROOT / "configs/real_post_train/data/custom.yaml")
        for split in (cfg.data.train, cfg.data.val):
            split.processor.shape_meta = OmegaConf.to_container(split.shape_meta, resolve=False)
            for phase in ("train_transforms", "val_transforms"):
                transforms = OmegaConf.to_container(split.processor[phase], resolve=False)
                split.processor[phase] = {key: deepcopy(transforms) for key in ("head", "left", "right")}
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "custom.yaml"
            OmegaConf.save(cfg, config)
            configure(config, "letterbox", (256, 448), False)
            updated = OmegaConf.load(config)
            processor = instantiate(updated.data.train.processor).train()
            raw = torch.rand(1, 3, 240, 424).mul(255).to(torch.uint8)
            for camera in ("head", "left", "right"):
                image = processor.transform_image(raw, camera)
                self.assertEqual(tuple(image.shape), (1, 3, 256, 448))
                self.assertTrue(torch.equal(image[..., 8:248, 12:436], raw.float() / 255))

    def test_size_update_adds_both_flags_to_older_yaml_without_losing_fields(self):
        configure = load_script("configure_custom_image_preproc").configure
        raw = (ROOT / "configs/real_post_train/data/custom.yaml").read_text()
        raw = "\n".join(line for line in raw.split("\n") if not any(
            field in line for field in ("image_resize_mode:", "image_letterbox_scale_up:")))
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "custom.yaml"
            config.write_text(raw)
            configure(config, "letterbox", (256, 448), False)
            updated = OmegaConf.load(config)
            for split in (updated.data.train, updated.data.val):
                self.assertEqual(split.processor.image_resize_mode, "letterbox")
                self.assertFalse(split.processor.image_letterbox_scale_up)
            self.assertIsNone(configure(config, "letterbox", (256, 448), False))

    def test_256x448_configuration_updates_all_geometry_and_preserves_local_values(self):
        configure = load_script("configure_custom_image_preproc").configure
        raw = (ROOT / "configs/real_post_train/data/custom.yaml").read_text()
        raw = raw.replace("coding_v3/", "coding_v4/").replace("  skill_prompt_map: null", "  skill_prompt_map: /local/skill_prompt_by_instruction.json")
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)), TemporaryDirectory() as tmp:
                config = Path(tmp) / "custom.yaml"
                original = raw.replace("\n", newline).encode()
                config.write_bytes(original)
                backup = configure(config, "letterbox", (256, 448), False)
                self.assertEqual(backup.read_bytes(), original)
                cfg = OmegaConf.load(config)
                for split in (cfg.data.train, cfg.data.val):
                    self.assertEqual(list(split.video_size), [256, 448])
                    for image in split.shape_meta.images:
                        self.assertEqual(list(image.raw_shape), [3, 240, 424])
                        self.assertEqual(list(image.shape), [3, 256, 448])
                    self.assertEqual(split.processor.image_resize_mode, "letterbox")
                    self.assertFalse(split.processor.image_letterbox_scale_up)
                    for phase in ("train_transforms", "val_transforms"):
                        self.assertEqual(list(split.processor[phase][1].size), [256, 448])
                self.assertIn(b"coding_v4/", config.read_bytes())
                self.assertIn(b"skill_prompt_by_instruction.json", config.read_bytes())
                self.assertIsNone(configure(config, "letterbox", (256, 448), False))
                configure(config, "stretch", (224, 320), True)
                self.assertEqual(config.read_bytes(), original)

    def test_invalid_size_does_not_write_or_create_backups(self):
        configure = load_script("configure_custom_image_preproc").configure
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "custom.yaml"
            original = (ROOT / "configs/real_post_train/data/custom.yaml").read_bytes()
            config.write_bytes(original)
            with self.assertRaises(ValueError):
                configure(config, "letterbox", (240, 424), False)
            self.assertEqual(config.read_bytes(), original)
            self.assertEqual(list(Path(tmp).iterdir()), [config])

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
