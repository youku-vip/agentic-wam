from copy import deepcopy
import hashlib
import json
import os
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torchvision.transforms import Resize

from agenticwam.datasets.lerobot import decoder_cache
from agenticwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset
from agenticwam.datasets.lerobot.processors.agenticwam_processor import AgenticWAMProcessor
from agenticwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from agenticwam.datasets.lerobot.transforms.custom_unified_eef import CustomUnifiedEEFTransform
from agenticwam.datasets.lerobot.transforms.image import ToTensor
from agenticwam.datasets.lerobot.v3 import dataset as v3_dataset


class CacheProbe(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 6

    def __getitem__(self, index):
        from agenticwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
        pixels = decode_video_frames(self.path, [0, .1, .2], .0999, "torchcodec", decoder_cache_size=2,
                                     decoder_num_threads=1)
        return {"index": index, "hash": hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
                "cache": decoder_cache.decoder_cache_info()}


def make_fixture(root):
    cameras = ("head", "left", "right")
    lengths = (40, 12, 37)
    starts = (0, 40, 52)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "cosmos").mkdir()
    features = {f"observation.images.{key}": {"dtype": "video", "shape": [3, 48, 64]} for key in cameras}
    features.update({"observation.state": {"dtype": "float32", "shape": [34]},
                     "action": {"dtype": "float32", "shape": [20]}})
    info = dict(codebase_version="v3.0", fps=10, total_episodes=3, total_frames=89, features=features,
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes, rows = [], []
    for ep, (start, length) in enumerate(zip(starts, lengths)):
        record = {"episode_index": ep, "length": length, "data/chunk_index": 0, "data/file_index": 0,
                  "dataset_from_index": start, "dataset_to_index": start + length}
        for camera, lead in zip(cameras, (0, 3, 7)):
            key = f"videos/observation.images.{camera}"
            record.update({key + "/chunk_index": 0, key + "/file_index": 0,
                           key + "/from_timestamp": (start + lead)/10,
                           key + "/to_timestamp": (start + length + lead)/10})
        episodes.append(record)
        for local in range(length):
            index = start + local
            left = [.1 + index/1000, .2, -.3, 1, 0, 0, 0, 1, 0]
            right = [-.1, .15 + index/2000, -.25, 1, 0, 0, 0, 1, 0]
            grips = [float(local % 8 < 4), float(local % 10 < 5)]
            state = np.array([0.0]*14 + left + right + grips, dtype=np.float32)
            action = np.array(left + right + grips, dtype=np.float32)
            action[0] += .01
            action[10] += .015
            rows.append({"episode_index": ep, "frame_index": local, "index": index, "task_index": ep,
                         "timestamp": index/15 + .002*(index % 3),  # deliberately NOT encoded 10fps
                         "observation.state": state, "action": action})
        text = DEFAULT_PROMPT.format(task=f"fixture task {ep}")
        digest = hashlib.sha256(text.encode()).hexdigest()
        torch.save({"prompt_embs": torch.full((2, 4), float(ep))}, root / "cosmos" / (digest + ".cosmos_reason1.pt"))
    pd.DataFrame(episodes).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [0, 1, 2], "task": [f"fixture task {i}" for i in range(3)]}).to_parquet(root / "meta/tasks.parquet")
    for number, (camera, lead) in enumerate(zip(cameras, (0, 3, 7))):
        video = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            stream.codec_context.gop_size = 12
            for index in range(89 + lead):
                rgb = np.zeros((48, 64, 3), dtype=np.uint8)
                rgb[..., 0] = (index*2 + number*50) % 256
                rgb[..., 1] = np.arange(64, dtype=np.uint8)[None, :] * 3
                rgb[..., 2] = np.arange(48, dtype=np.uint8)[:, None] * 4
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    stats = {}
    for field in ("action", "state"):
        stats[field] = {"default": {}}
        for prefix, shape in (("global", (64,)), ("stepwise", (32, 64) if field == "action" else (1, 64))):
            for key, value in (("min", -2), ("max", 2), ("q01", -2), ("q99", 2), ("mean", 0), ("std", 1)):
                stats[field]["default"][f"{prefix}_{key}"] = np.full(shape, value, dtype=float).tolist()
    (root / "dataset_stats_eef64.json").write_text(json.dumps(stats))
    codes = [f"def policy_chunk(obs, state, robot, skills):\n    return robot.move_arm(side='left', target_pos=[{i}, 0, 0])\n" for i in range(3)]
    torch.save({"codes": codes, "code_id_by_idx": torch.arange(89) % 3}, root / "code.pt")
    return root


def build_dataset(root, *, sparse=False, cache_size=0, stride=1, use_environment=False):
    shape = {"images": [{"key": key, "raw_shape": [3, 48, 64], "shape": [3, 24, 32]} for key in ("head", "left", "right")],
             "action": [{"key": "default", "raw_shape": 20, "shape": 64}],
             "state": [{"key": "default", "raw_shape": 34, "shape": 64}]}
    processor = AgenticWAMProcessor(
        shape_meta=deepcopy(shape), num_obs_steps=33, num_output_cameras=3,
        action_output_dim=64, proprio_output_dim=64,
        action_state_transforms=[CustomUnifiedEEFTransform(keys=["default"])],
        use_stepwise_action_norm=True, norm_default_mode="q01/q99", norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(), train_transforms=[ToTensor(), Resize((24, 32))],
        val_transforms=[ToTensor(), Resize((24, 32))],
    )
    return RobotVideoDataset(
        dataset_dirs=[str(root)], shape_meta=OmegaConf.create(shape), num_frames=33,
        action_video_freq_ratio=4, global_sample_stride=stride, processor=processor,
        video_size=[24, 32], concat_multi_camera="multiview", pretrained_norm_stats=str(root / "dataset_stats_eef64.json"),
        is_training_set=True, val_set_proportion=0, cosmos_text_cache_dir=str(root / "cosmos"), use_policy_code=True,
        policy_code_idx_path=str(root / "code.pt"), save_norm_stats=False,
        sparse_video_decode=None if use_environment else sparse, video_backend="torchcodec",
        video_decoder_cache_size=None if use_environment else cache_size, video_decoder_num_threads=1,
    )


class SparsePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from torchcodec.decoders import VideoDecoder
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"Working TorchCodec unavailable: {exc}")
        cls.tmp = TemporaryDirectory()
        cls.root = make_fixture(Path(cls.tmp.name))
        cls.dense = build_dataset(cls.root)
        cls.sparse = build_dataset(cls.root, sparse=True)

    @classmethod
    def tearDownClass(cls):
        decoder_cache.clear_decoder_cache()
        cls.tmp.cleanup()

    def test_complete_samples_match_at_episode_edges_and_short_episode(self):
        stats_before = (self.root / "dataset_stats_eef64.json").read_bytes()
        for index in (0, 5, 20, 38, 39, 40, 46, 51, 52, 70, 88):
            np.random.seed(100 + index)
            left = self.dense[index]
            np.random.seed(100 + index)
            right = self.sparse[index]
            self.assertEqual(set(left), set(right))
            for key in left:
                if torch.is_tensor(left[key]):
                    self.assertTrue(torch.equal(left[key], right[key]), (index, key))
                else:
                    self.assertEqual(left[key], right[key], (index, key))
            self.assertEqual(right["video"].shape, (3, 27, 24, 32))
            self.assertEqual(right["action"].shape, (32, 64))
            self.assertEqual(right["proprio"].shape, (32, 64))
            self.assertEqual(right["image_is_pad"].shape, (9,))
            self.assertIn(f"target_pos=[{index % 3},", right["policy_code"])
        self.assertEqual(stats_before, (self.root / "dataset_stats_eef64.json").read_bytes())

    def test_raw_windows_and_shared_mp4_offsets(self):
        original = v3_dataset.decode_video_frames
        for index in (0, 39, 40, 46, 51, 52, 88):
            with patch.object(v3_dataset, "decode_video_frames", wraps=original) as calls:
                left = self.dense.lerobot_dataset[index]
                dense_times = [call.args[1] for call in calls.call_args_list]
            with patch.object(v3_dataset, "decode_video_frames", wraps=original) as calls:
                right = self.sparse.lerobot_dataset[index]
                sparse_times = [call.args[1] for call in calls.call_args_list]
            self.assertEqual(len(dense_times), 3)
            self.assertEqual(sparse_times, [timestamps[::4] for timestamps in dense_times])
            self.assertTrue(torch.equal(left["pixel_values"][:, ::4], right["pixel_values"]))
            self.assertTrue(torch.equal(left["image_is_pad"][::4], right["image_is_pad"]))
            self.assertEqual(right["proprio"].shape, (33, 64))
            self.assertEqual(right["action"].shape, (32, 64))
            self.assertEqual(left["idx"], right["idx"])
        self.assertFalse(torch.equal(right["pixel_values"][0], right["pixel_values"][1]))

    def test_nonunit_stride_and_environment_override(self):
        dense = build_dataset(self.root, stride=2)
        sparse = build_dataset(self.root, sparse=True, stride=2)
        for index in (0, 40, 88):
            self.assertTrue(torch.equal(dense[index]["video"], sparse[index]["video"]))
            self.assertTrue(torch.equal(dense[index]["action"], sparse[index]["action"]))
        with patch.dict(os.environ, {"AGENTICWAM_SPARSE_VIDEO_DECODE": "1", "AGENTICWAM_VIDEO_DECODER_CACHE_SIZE": "4"}):
            automatic = build_dataset(self.root, use_environment=True)
            explicit = build_dataset(self.root, sparse=False)
        self.assertTrue(automatic.sparse_video_decode)
        self.assertEqual(automatic.lerobot_dataset.multi_dataset._datasets[0].decoder_cache_size, 4)
        self.assertFalse(explicit.sparse_video_decode)

    def test_native_cache_reuse_is_isolated_in_two_real_workers(self):
        path = str(self.root / "videos/observation.images.head/chunk-000/file-000.mp4")
        decoder_cache.clear_decoder_cache()
        rows = list(torch.utils.data.DataLoader(CacheProbe(path), batch_size=None, num_workers=2))
        by_pid = {row["cache"]["pid"]: row["cache"] for row in rows}
        self.assertEqual(len(by_pid), 2)
        self.assertNotIn(os.getpid(), by_pid)
        self.assertEqual(len({row["hash"] for row in rows}), 1)
        for info in by_pid.values():
            self.assertEqual(info["size"], 1)
            self.assertEqual(info["misses"], 1)
            self.assertEqual(info["hits"], 2)
        self.assertEqual(decoder_cache.decoder_cache_info()["size"], 0)

    def test_real_verifier_compares_final_fields_indices_and_rng(self):
        spec = importlib.util.spec_from_file_location("verify_decode", Path(__file__).resolve().parents[1] / "scripts/verify_custom_video_decode.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        indices = [0, 39, 40, 51, 88]
        baseline = module.collect(build_dataset(self.root), indices, workers=0)
        optimized = module.collect(build_dataset(self.root, sparse=True, cache_size=4), indices, workers=0)
        report = module.compare_records(baseline, optimized)
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["checked_samples"], 5)


class DecoderCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.video = Path(self.tmp.name) / "video.mp4"
        self.video.write_bytes(b"fixture")
        decoder_cache.clear_decoder_cache()

    def tearDown(self):
        decoder_cache.clear_decoder_cache()
        self.tmp.cleanup()

    def test_parent_never_caches_and_worker_reuses(self):
        with patch("torchcodec.decoders.VideoDecoder", side_effect=lambda *a, **k: object()) as factory:
            first = decoder_cache.get_video_decoder(self.video, cache_size=4)
            second = decoder_cache.get_video_decoder(self.video, cache_size=4)
            self.assertIsNot(first, second)
            self.assertEqual(decoder_cache.decoder_cache_info()["size"], 0)
            with patch("torch.utils.data.get_worker_info", return_value=SimpleNamespace(id=0)):
                first = decoder_cache.get_video_decoder(self.video, cache_size=4)
                second = decoder_cache.get_video_decoder(self.video, cache_size=4)
            self.assertIs(first, second)
            self.assertEqual(factory.call_count, 3)
            self.assertEqual(decoder_cache.decoder_cache_info()["hits"], 1)

    def test_lru_bound_file_change_and_thread_settings(self):
        other = self.video.with_name("other.mp4")
        other.write_bytes(b"other")
        with patch("torchcodec.decoders.VideoDecoder", side_effect=lambda *a, **k: object()) as factory, \
                patch("torch.utils.data.get_worker_info", return_value=SimpleNamespace(id=0)):
            first = decoder_cache.get_video_decoder(self.video, cache_size=1, num_threads=1)
            decoder_cache.get_video_decoder(other, cache_size=1, num_threads=1)
            second = decoder_cache.get_video_decoder(self.video, cache_size=1, num_threads=1)
            self.assertIsNot(first, second)
            self.video.write_bytes(b"changed video generation")
            third = decoder_cache.get_video_decoder(self.video, cache_size=1, num_threads=1)
            fourth = decoder_cache.get_video_decoder(self.video, cache_size=1, num_threads=2)
            self.assertIsNot(second, third)
            self.assertIsNot(third, fourth)
            self.assertEqual(decoder_cache.decoder_cache_info()["size"], 1)
            self.assertGreaterEqual(decoder_cache.decoder_cache_info()["evictions"], 2)
            self.assertEqual(factory.call_args.kwargs["num_ffmpeg_threads"], 2)

    def test_pid_change_and_failed_decoder_eviction(self):
        with patch("torchcodec.decoders.VideoDecoder", side_effect=lambda *a, **k: object()), \
                patch("torch.utils.data.get_worker_info", return_value=SimpleNamespace(id=0)):
            decoder_cache.get_video_decoder(self.video, cache_size=4)
            with patch("os.getpid", return_value=os.getpid()+10000):
                self.assertEqual(decoder_cache.decoder_cache_info()["size"], 0)
            decoder_cache.get_video_decoder(self.video, cache_size=4)
            decoder_cache.discard_video_decoder(self.video)
            self.assertEqual(decoder_cache.decoder_cache_info()["size"], 0)


if __name__ == "__main__":
    unittest.main()
