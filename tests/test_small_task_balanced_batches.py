from collections import Counter
import ast
import hashlib
import importlib.util
import json
from itertools import islice
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from accelerate.data_loader import BatchSamplerShard
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import BatchSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agenticwam.datasets.lerobot.task_sampling import TaskSamplingIndex
from agenticwam.trainer import AgenticWAMTrainer
from agenticwam.utils.samplers import TaskBalancedEpochSampler

spec = importlib.util.spec_from_file_location('configure_small_batch',
    ROOT / 'scripts/configure_custom_code_small_batch.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class BalancedToyDataset(torch.utils.data.Dataset):
    def __init__(self, counts=(16, 24, 32, 24)):
        labels = np.repeat(np.arange(4, dtype=np.int64), counts)
        self.index = TaskSamplingIndex(labels, tuple(f'task-{i}' for i in range(4)),
                                       hashlib.sha256(labels.tobytes()).hexdigest())
        self.enabled = False

    def __len__(self): return len(self.index.task_ids)
    def get_task_sampling_index(self): return self.index
    def enable_task_balanced_sampling(self, index): self.enabled = True
    def __getitem__(self, index):
        return {'x': torch.tensor([index / 100., .5, 1.]),
                'sampling_task_id': int(self.index.task_ids[index])}


class SmallBatchTests(unittest.TestCase):
    def test_old_batch4_order_rng_and_descriptor_are_unchanged(self):
        data = BalancedToyDataset((10, 17, 26, 31))
        sampler = TaskBalancedEpochSampler(data, data.index, 42, 4, 2, 8)
        state = torch.get_rng_state().clone()
        actual = hashlib.sha256(np.asarray(list(sampler), dtype='<i8').tobytes()).hexdigest()
        self.assertEqual(actual, '14da8729f4fa8c7be3994406077b41f9f6d23f49b692205c831480900c1441a4')
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(sampler.balance_scope, 'local')
        self.assertNotIn('balance_scope', sampler.descriptor())

    def test_real_accelerate_shards_balance_every_update_and_rank(self):
        data = BalancedToyDataset((64, 64, 64, 64))
        sampler = TaskBalancedEpochSampler(data, data.index, 42, 2, 2, 16)
        shards = [list(BatchSamplerShard(BatchSampler(sampler, 2, False), 2, rank, False, True))
                  for rank in range(2)]
        for step in range(3):
            updates = [Counter(), Counter()]
            for micro in range(step * 16, (step + 1) * 16):
                left, right = (shards[r][micro] for r in range(2))
                self.assertEqual(len(set(left + right)), 4)
                self.assertEqual(Counter(data.index.task_ids[left + right]), dict.fromkeys(range(4), 1))
                for rank, rows in enumerate((left, right)):
                    updates[rank].update(data.index.task_ids[rows])
            self.assertEqual(updates[0], dict.fromkeys(range(4), 8))
            self.assertEqual(updates[1], dict.fromkeys(range(4), 8))
        self.assertEqual(sampler.descriptor()['balance_scope'], 'global_microbatch_v1')

    def test_resume_epoch_and_unsupported_global_batch(self):
        data = BalancedToyDataset()
        sampler = TaskBalancedEpochSampler(data, data.index, 42, 2, 2, 16)
        sampler.set_epoch(3)
        original = list(sampler)
        resumed = TaskBalancedEpochSampler(data, data.index, 42, 2, 2, 16)
        resumed.set_epoch_offset(3)
        resumed.set_resume_batch_offset(7)
        self.assertEqual(list(resumed), original[7 * 2 * 2:])
        self.assertEqual(len(resumed), len(list(resumed)))
        with self.assertRaisesRegex(ValueError, 'divisible'):
            TaskBalancedEpochSampler(data, data.index, 42, 1, 2, 32)

    def test_actual_trainer_loader_and_resume_recipe(self):
        trainer = AgenticWAMTrainer.__new__(AgenticWAMTrainer)
        trainer.cfg = OmegaConf.create({'task_balanced_batch': True})
        trainer.seed, trainer.batch_size, trainer.num_workers = 42, 2, 0
        trainer.gradient_accumulation_steps = 16
        trainer.accelerator = SimpleNamespace(num_processes=2, device=torch.device('cpu'),
                                               gather=lambda x: torch.cat([x, x]))
        data = BalancedToyDataset()
        loader = trainer._build_loader(data)
        self.assertTrue(data.enabled)
        self.assertEqual(loader.sampler.balance_scope, 'global')
        descriptor = trainer._sampler_descriptor()
        trainer._check_sampler_resume({'sampler': descriptor})
        with self.assertRaisesRegex(ValueError, 'same task mapping'):
            trainer._check_sampler_resume({'sampler': {**descriptor, 'batch_size': 4}})

    def test_read_only_verifier_checks_global_batches_with_real_accelerate(self):
        path = ROOT / 'scripts/verify_task_balanced_batches.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
        namespace = dict(Counter=Counter, islice=islice, BatchSampler=BatchSampler,
                         BatchSamplerShard=BatchSamplerShard, TaskBalancedEpochSampler=TaskBalancedEpochSampler,
                         build_task_sampling_index=lambda dataset: dataset.index)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        report = namespace['verify'](BalancedToyDataset((64, 64, 64, 64)), 2, 2, 16, 3)
        self.assertTrue(report['passed'])
        self.assertTrue(report['resume_prefix_check_passed'])
        self.assertIsNone(report['expected_per_rank_microbatch'])
        self.assertEqual(report['expected_global_microbatch'], dict.fromkeys(range(4), 1))
        self.assertEqual(report['observed_optimizer_batch_counts'], [dict.fromkeys(range(4), 16)] * 3)


class SmallBatchConfigurationTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root / 'recipe').mkdir()
        shared_ds = root / 'shared_ds.json'
        shared_ds.write_text(json.dumps(dict(train_batch_size='auto',
            train_micro_batch_size_per_gpu='auto', gradient_accumulation_steps='auto',
            zero_optimization=dict(stage=1, overlap_comm=False, contiguous_gradients=False,
                                   reduce_bucket_size=1_000_000_000, allgather_bucket_size=500_000_000))))
        accelerator = root / 'shared_accelerator.yaml'
        OmegaConf.save(OmegaConf.create(dict(distributed_type='DEEPSPEED',
            deepspeed_config=dict(deepspeed_config_file=str(shared_ds)))), accelerator)
        cfg = dict(batch_size=4, gradient_accumulation_steps=8, resume=None,
                   foundation_ckpt='/reference/step_008000.pt', learning_rate=1e-5, max_steps=300,
                   model=dict(code_canvas=128, infer_code_tokens=128, use_gradient_checkpointing=True),
                   data=dict(stats='/original/stats.json'), wandb=dict(mode='offline'))
        OmegaConf.save(OmegaConf.create(cfg), root / 'recipe/config.yaml')
        command = shlex.join([sys.executable, '-m', 'accelerate.commands.launch', '--config_file',
                              str(accelerator), '--num_processes', '2', 'scripts/train.py'])
        (root / 'run_code_repair.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + command + '\n')
        return root, shared_ds, accelerator

    def test_configure_preserves_effective_batch_shared_configs_and_backup(self):
        with TemporaryDirectory(prefix='small batch ') as folder:
            root, ds, accelerator = self.fixture(folder)
            old_recipe = (root / 'recipe/config.yaml').read_bytes()
            old_launcher = (root / 'run_code_repair.sh').read_bytes()
            shared = {p: p.read_bytes() for p in (ds, accelerator)}
            report = tool.configure(root)
            self.assertTrue(report['passed'])
            self.assertEqual(report['effective_batch'], 64)
            self.assertEqual(report['gradient_accumulation_steps'], 16)
            self.assertEqual(report['observed_update_counts'], dict.fromkeys(range(4), 16))
            cfg = OmegaConf.load(root / 'recipe/config.yaml')
            self.assertEqual(cfg.batch_size, 2)
            for field in ('foundation_ckpt', 'learning_rate', 'max_steps', 'data', 'model', 'wandb'):
                self.assertEqual(cfg[field], OmegaConf.create(old_recipe.decode())[field])
            for p, content in shared.items(): self.assertEqual(p.read_bytes(), content)
            backup = Path(report['original_files_backup'])
            self.assertEqual((backup / 'config.yaml').read_bytes(), old_recipe)
            self.assertEqual((backup / 'run_code_repair.sh').read_bytes(), old_launcher)
            local_ds = json.loads(Path(report['deepspeed_config']).read_text())
            self.assertEqual(local_ds['zero_optimization']['reduce_bucket_size'], 50_000_000)
            self.assertFalse(local_ds['zero_optimization']['overlap_comm'])
            self.assertIn(report['accelerate_config'], (root / 'run_code_repair.sh').read_text())
            subprocess.run(['bash', '-n', str(root / 'run_code_repair.sh')], check=True)
            with self.assertRaises(FileExistsError): tool.configure(root)

    def test_bad_small_batch_and_started_trial_leave_config_untouched(self):
        with TemporaryDirectory() as folder:
            root, _, _ = self.fixture(folder)
            before = (root / 'recipe/config.yaml').read_bytes()
            with self.assertRaisesRegex(ValueError, 'multiple of four'):
                tool.configure(root, batch_size=1)
            (root / 'checkpoints/weights').mkdir(parents=True)
            (root / 'checkpoints/weights/step_000100.pt').write_bytes(b'no checkpoint tensor is loaded')
            with self.assertRaisesRegex(ValueError, 'saved weights'):
                tool.configure(root)
            self.assertEqual((root / 'recipe/config.yaml').read_bytes(), before)
            self.assertFalse(list(root.glob('small_batch_backup_*')))

    def test_full_generated_repair_launcher_is_patched_and_valid_bash(self):
        from test_custom_code_repair_prepare import RepairPreparationTests, Tokenizer, tool as prepare_tool
        with TemporaryDirectory() as folder:
            args = RepairPreparationTests().fixture(folder)
            prepare_tool.prepare(args, Tokenizer())
            source = args.source_config.read_bytes()
            before = OmegaConf.load(args.out / 'recipe/config.yaml')
            stats = Path(before.data.train.pretrained_norm_stats).read_bytes()
            report = tool.configure(args.out)
            after = OmegaConf.load(args.out / 'recipe/config.yaml')
            for name, value in before.model.items():
                self.assertEqual(after.model[name], value)
            self.assertTrue(after.model.use_gradient_checkpointing)
            self.assertEqual(after.data, before.data)
            self.assertEqual(after.foundation_ckpt, before.foundation_ckpt)
            self.assertEqual(Path(after.data.train.pretrained_norm_stats).read_bytes(), stats)
            self.assertEqual(args.source_config.read_bytes(), source)
            launcher = args.out / 'run_code_repair.sh'
            subprocess.run(['bash', '-n', str(launcher)], check=True)
            self.assertIn(report['accelerate_config'], launcher.read_text())
            for mode in ('baseline)', 'train)', 'eval)'):
                self.assertIn(mode, launcher.read_text())


if __name__ == '__main__':
    unittest.main()
