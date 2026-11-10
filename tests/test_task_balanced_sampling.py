from collections import Counter
import hashlib
import importlib.util
from itertools import islice
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from accelerate.data_loader import BatchSamplerShard
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch.utils.data import BatchSampler

from agenticwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from agenticwam.datasets.lerobot.task_sampling import TaskSamplingIndex, build_task_sampling_index
from agenticwam.datasets.lerobot.v3.dataset import V3MultiLeRobotDataset
from agenticwam.trainer import AgenticWAMTrainer
from agenticwam.utils.samplers import ResumableEpochSampler, TaskBalancedEpochSampler
from test_sparse_video_decode import make_fixture, build_dataset


def toy_index(counts=(10, 17, 26, 31)):
    labels = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return TaskSamplingIndex(labels, tuple(f"task-{i}" for i in range(len(counts))),
                             hashlib.sha256(labels.tobytes()).hexdigest())


class BalancedToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.index = toy_index((20, 24, 28, 32))
        self.enabled = False

    def __len__(self):
        return len(self.index.task_ids)

    def get_task_sampling_index(self):
        return self.index

    def enable_task_balanced_sampling(self, index):
        self.enabled = True

    def __getitem__(self, index):
        return {"x": torch.tensor([index / 100, .5, 1.0]),
                "sampling_task_id": int(self.index.task_ids[index])}


def make_metadata_fixture(root, mixed=False):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    ids, lengths = (2, 7, 11, 20), (12, 17, 22, 27)
    records, files, start = [], {0: [], 1: []}, 0
    for task, (ep, length) in enumerate(zip(ids, lengths)):
        file = task % 2
        records.append(dict(episode_index=ep, length=length, **{"data/chunk_index": 0, "data/file_index": file,
                          "dataset_from_index": start, "dataset_to_index": start + length}))
        for local in range(length):
            files[file].append(dict(episode_index=ep, frame_index=local,
                                    task_index=(task + local) % 4 if mixed else task))
        start += length
    info = dict(codebase_version="v3.0", fps=10, total_episodes=4, total_frames=start, features={},
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    (root / "meta/info.json").write_text(json.dumps(info))
    pd.DataFrame(records).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": list(range(4))}, index=[f"task-{i}" for i in range(4)]).to_parquet(root / "meta/tasks.parquet")
    for file, rows in files.items():
        pd.DataFrame(rows).sample(frac=1, random_state=17).to_parquet(root / f"data/chunk-000/file-{file:03d}.parquet")
    return str(root)


class BalancedSamplerTests(unittest.TestCase):
    def sampler(self, index=None, batch=4, world=2, accum=8):
        index = index or toy_index()
        return TaskBalancedEpochSampler(range(len(index.task_ids)), index, 42, batch, world, accum)

    def test_exact_local_global_and_accumulated_counts_no_cross_rank_duplicates(self):
        sampler = self.sampler()
        state = torch.get_rng_state().clone()
        shards = [list(BatchSamplerShard(BatchSampler(sampler, 4, False), 2, rank, False, True))
                  for rank in range(2)]
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(len(shards[0]), len(shards[1]))
        self.assertEqual(len(shards[0]) % 8, 0)
        for left, right in zip(*shards):
            self.assertEqual(len(set(left + right)), 8)
            for batch in (left, right):
                self.assertEqual(Counter(sampler.task_index.task_ids[batch]), dict.fromkeys(range(4), 1))
        for begin in range(0, len(shards[0]), 8):
            samples = [item for rank in shards for batch in rank[begin:begin+8] for item in batch]
            self.assertEqual(Counter(sampler.task_index.task_ids[samples]), dict.fromkeys(range(4), 16))

    def test_resumption_and_epoch_sequence_are_exact(self):
        sampler = self.sampler()
        sampler.set_epoch(3)
        complete = list(sampler)
        resumed = self.sampler()
        resumed.set_epoch_offset(3)
        resumed.set_resume_batch_offset(3)
        self.assertEqual(list(resumed), complete[3*4*2:])
        self.assertEqual(len(list(resumed)), len(resumed))
        resumed.clear_resume_batch_offset()
        resumed.set_epoch(1)
        sampler.set_epoch(4)
        self.assertEqual(list(resumed), list(sampler))
        self.assertNotEqual(complete, list(sampler))

    def test_incompatible_batch_and_insufficient_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.sampler(batch=1, world=2)
        with self.assertRaisesRegex(ValueError, "distinct samples"):
            self.sampler(index=toy_index((1, 5, 6, 7)))
        self.assertEqual(len(list(self.sampler(batch=8, world=3, accum=2))) % 48, 0)

    def test_trainer_opt_in_and_configuration_override(self):
        trainer = AgenticWAMTrainer.__new__(AgenticWAMTrainer)
        trainer.cfg = OmegaConf.create({"task_balanced_batch": None})
        trainer.seed, trainer.batch_size, trainer.num_workers, trainer.gradient_accumulation_steps = 42, 4, 0, 8
        trainer.accelerator = SimpleNamespace(num_processes=2, device=torch.device("cpu"), gather=lambda x: torch.cat([x, x]))
        dataset = BalancedToyDataset()
        with patch.dict(os.environ, {"AGENTICWAM_TASK_BALANCED_BATCH": "1"}):
            loader = trainer._build_loader(dataset)
            self.assertIsInstance(loader.sampler, TaskBalancedEpochSampler)
            self.assertTrue(dataset.enabled)
            descriptor = trainer._sampler_descriptor()
            trainer._check_sampler_resume({"sampler": descriptor})
            trainer.global_step, trainer.epoch, trainer.batch_in_epoch = 7, 0, 56
            with TemporaryDirectory() as tmp:
                trainer._save_trainer_state(tmp)
                payload = json.loads((Path(tmp) / "trainer_state.json").read_text())
                self.assertEqual(payload["sampler"], descriptor)
                self.assertEqual(payload["batch_in_epoch"], 56)
            with self.assertRaisesRegex(ValueError, "same task mapping"):
                trainer._check_sampler_resume({"sampler": {**descriptor, "seed": 1}})
            trainer.cfg.task_balanced_batch = False
            loader = trainer._build_loader(BalancedToyDataset())
            self.assertIsInstance(loader.sampler, ResumableEpochSampler)
            self.assertNotIsInstance(loader.sampler, TaskBalancedEpochSampler)
            trainer.cfg.task_balanced_batch = True
            trainer.accelerator.split_batches = True
            with self.assertRaisesRegex(ValueError, "split_batches=False"):
                trainer._build_loader(BalancedToyDataset())


class TaskIndexTests(unittest.TestCase):
    def test_packed_files_selected_episode_order_mixed_frame_tasks_and_multiple_roots(self):
        with TemporaryDirectory() as tmp:
            root = make_metadata_fixture(Path(tmp) / "first", mixed=True)
            multi = V3MultiLeRobotDataset([root], episodes={root: [20, 2, 11]})
            with patch("pandas.read_parquet", wraps=pd.read_parquet) as reads:
                index = build_task_sampling_index(multi)
            self.assertEqual(reads.call_count, 2)
            actual = [index.task_names[int(task)] for task in index.task_ids]
            expected = [multi[i]["task"] for i in range(len(multi))]
            self.assertEqual(actual, expected)
            self.assertEqual(sum(index.counts()), len(multi))
            other = make_metadata_fixture(Path(tmp) / "second")
            combined = V3MultiLeRobotDataset([other, root], episodes={other: [7], root: [20, 2]})
            both = build_task_sampling_index(combined)
            self.assertEqual([both.task_names[int(task)] for task in both.task_ids],
                             [combined[i]["task"] for i in range(len(combined))])

    def test_missing_metadata_task_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(make_metadata_fixture(Path(tmp)))
            path = root / "data/chunk-000/file-000.parquet"
            frame = pd.read_parquet(path)
            frame.loc[0, "task_index"] = 99
            frame.to_parquet(path)
            with self.assertRaisesRegex(ValueError, "missing from metadata"):
                build_task_sampling_index(V3MultiLeRobotDataset([str(root)]))

    def test_read_only_verifier_uses_real_accelerate_batch_shards(self):
        spec = importlib.util.spec_from_file_location("verify_tasks", Path(__file__).resolve().parents[1] / "scripts/verify_task_balanced_batches.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            root = make_metadata_fixture(Path(tmp))
            report = module.verify(V3MultiLeRobotDataset([root]), grad_accum=2, steps=3)
            self.assertTrue(report["passed"])
            for counts in report["observed_optimizer_batch_counts"]:
                self.assertEqual(counts, dict.fromkeys(range(4), 4))


class BalancedRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.root = make_fixture(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_final_images_actions_code_and_padding_are_unchanged(self):
        dataset = build_dataset(self.root, sparse=True)
        before = dataset[39]
        dataset.enable_task_balanced_sampling()
        after = dataset[39]
        for key in before:
            if torch.is_tensor(before[key]):
                self.assertTrue(torch.equal(before[key], after[key]), key)
            else:
                self.assertEqual(before[key], after[key], key)
        self.assertEqual(after["sampling_task_id"], 0)

    def test_processing_retry_and_padding_retry_stay_in_the_same_task(self):
        dataset = build_dataset(self.root, sparse=True)
        dataset.enable_task_balanced_sampling()
        seen = []
        def attempt(index):
            seen.append(index)
            if len(seen) == 1:
                raise RuntimeError("synthetic processing failure")
            return {"task": int(dataset._balanced_task_index.task_ids[index])}
        with patch.object(dataset, "_get", side_effect=attempt):
            self.assertEqual(dataset[45]["task"], 1)
        self.assertTrue(all(dataset._balanced_task_index.task_ids[index] == 1 for index in seen))
        dataset.skip_padding_as_possible, dataset.max_padding_retry = True, 2
        with patch.object(dataset, "_retry_sample_index", wraps=dataset._retry_sample_index) as retries:
            self.assertEqual(dataset[51]["sampling_task_id"], 1)
            self.assertGreater(retries.call_count, 0)

    def test_base_read_failure_cannot_silently_fall_back_to_another_task(self):
        dataset = build_dataset(self.root)
        dataset.enable_task_balanced_sampling()
        base = dataset.lerobot_dataset
        with patch.object(type(base.multi_dataset), "__getitem__", side_effect=RuntimeError("synthetic decode error")) as reads:
            with self.assertRaisesRegex(RuntimeError, "synthetic decode error"):
                base[0]
        self.assertEqual(reads.call_count, 1)


if __name__ == "__main__":
    unittest.main()
