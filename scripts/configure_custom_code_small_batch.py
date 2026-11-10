#!/usr/bin/env python3
"""Configure an unstarted repair trial for smaller local batches; never train."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import BatchSampler
from accelerate.data_loader import BatchSamplerShard

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agenticwam.datasets.lerobot.task_sampling import TaskSamplingIndex
from agenticwam.utils.samplers import TaskBalancedEpochSampler


def probe(batch_size, world_size, accumulation):
    labels = np.repeat(np.arange(4, dtype=np.int64), 64)
    index = TaskSamplingIndex(labels, tuple(f'task-{i}' for i in range(4)),
                              hashlib.sha256(labels.tobytes()).hexdigest())
    sampler = TaskBalancedEpochSampler(range(len(labels)), index, 42, batch_size,
                                       world_size, accumulation)
    shards = [iter(BatchSamplerShard(BatchSampler(sampler, batch_size, False),
                                    world_size, rank, False, True)) for rank in range(world_size)]
    totals = Counter()
    local_totals = [Counter() for _ in range(world_size)]
    for _ in range(accumulation):
        seen, counts = [], Counter()
        for rank, shard in enumerate(shards):
            rows = next(shard)
            seen.extend(rows)
            current = Counter(int(labels[i]) for i in rows)
            counts.update(current)
            local_totals[rank].update(current)
        if len(set(seen)) != len(seen):
            raise ValueError('duplicate sample across ranks in a global microbatch')
        expected = dict.fromkeys(range(4), batch_size * world_size // 4)
        if counts != expected:
            raise ValueError('replace the task sampler: global microbatch is not balanced')
        totals.update(counts)
    return dict(balance_scope=sampler.balance_scope,
                expected_global_microbatch=expected,
                observed_update_counts=dict(totals),
                observed_per_rank_update_counts=[dict(x) for x in local_totals])


def atomic_write(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def configure(run, batch_size=2, bucket_elements=50_000_000):
    run = Path(run).resolve()
    recipe, launcher = run / 'recipe/config.yaml', run / 'run_code_repair.sh'
    cfg = OmegaConf.load(recipe)
    if cfg.get('resume') not in (None, '', 'null'):
        raise ValueError('this helper targets a fresh weights-initialized repair trial')
    if list((run / 'checkpoints/weights').glob('step_*.pt')):
        raise ValueError('trial has saved weights; create a new trial before changing its sampler')
    text = launcher.read_text()
    commands = [line for line in text.splitlines()
                if ' -m accelerate.commands.launch ' in line and not line.lstrip().startswith('#')]
    if len(commands) != 1:
        raise ValueError('expected one generated Accelerate training command')
    command = shlex.split(commands[0])
    world = int(command[command.index('--num_processes') + 1])
    accelerator_path = Path(command[command.index('--config_file') + 1])
    if not accelerator_path.is_absolute():
        accelerator_path = ROOT / accelerator_path
    previous_effective = int(cfg.batch_size) * int(cfg.gradient_accumulation_steps) * world
    if batch_size < 1 or batch_size * world % 4:
        raise ValueError('batch_size * world_size must be a positive multiple of four tasks')
    if previous_effective % (batch_size * world):
        raise ValueError('cannot preserve the effective batch with integer accumulation')
    accumulation = previous_effective // (batch_size * world)
    if bucket_elements < 1:
        raise ValueError('bucket_elements must be positive (a count of elements)')
    report = probe(batch_size, world, accumulation)
    accelerator_cfg = OmegaConf.load(accelerator_path)
    old_ds_path = Path(str(accelerator_cfg.deepspeed_config.deepspeed_config_file))
    if not old_ds_path.is_absolute():
        old_ds_path = ROOT / old_ds_path
    ds = json.loads(old_ds_path.read_text())
    if int(ds['zero_optimization']['stage']) != 1:
        raise ValueError('this small-batch recipe targets the existing ZeRO-1 trial')
    for field in ('reduce_bucket_size', 'allgather_bucket_size'):
        old = ds['zero_optimization'].get(field, bucket_elements)
        ds['zero_optimization'][field] = min(int(old), bucket_elements) if old != 'auto' else bucket_elements
    new_ds = run / f'recipe/ds_zero1_bs{batch_size}.json'
    new_accelerator = run / f'recipe/accelerate_zero1_bs{batch_size}.yaml'
    if new_ds.exists() or new_accelerator.exists():
        raise FileExistsError('small-batch config already exists; keep its backup and inspect the report')
    accelerator_cfg.deepspeed_config.deepspeed_config_file = str(new_ds)
    command[command.index('--config_file') + 1] = str(new_accelerator)
    updated_launcher = text.replace(commands[0], '    ' + shlex.join(command), 1)
    if 'set -euo pipefail' not in updated_launcher:
        raise ValueError('expected generated Bash launcher')
    updated_launcher = updated_launcher.replace('set -euo pipefail',
        'set -euo pipefail\nexport AGENTICWAM_MOT_NO_CKPT_EVERY=0\n'
        'export AGENTICWAM_DISABLE_MOT_ATTN_CKPT=0\n'
        'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True', 1)
    cfg.batch_size = batch_size
    cfg.gradient_accumulation_steps = accumulation
    cfg.model.use_gradient_checkpointing = True
    cfg.task_balanced_batch = True
    # All validation above is CPU-only and precedes any mutation of the trial.
    backup = run / ('small_batch_backup_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir()
    shutil.copy2(recipe, backup / 'config.yaml')
    shutil.copy2(launcher, backup / 'run_code_repair.sh')
    atomic_write(new_ds, json.dumps(ds, indent=2) + '\n')
    atomic_write(new_accelerator, OmegaConf.to_yaml(accelerator_cfg, resolve=True))
    atomic_write(recipe, OmegaConf.to_yaml(cfg, resolve=True))
    atomic_write(launcher, updated_launcher)
    report.update(passed=True, rank_microbatch=batch_size, world_size=world,
                  gradient_accumulation_steps=accumulation, effective_batch=previous_effective,
                  reduce_bucket_size=ds['zero_optimization']['reduce_bucket_size'],
                  allgather_bucket_size=ds['zero_optimization']['allgather_bucket_size'],
                  config=str(recipe), accelerate_config=str(new_accelerator),
                  deepspeed_config=str(new_ds), original_files_backup=str(backup),
                  task_count_check='synthetic index through real Accelerate batch shards',
                  training_started=False)
    atomic_write(run / 'small_batch_config_report.json', json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--bucket-elements', type=int, default=50_000_000)
    args = parser.parse_args()
    print(json.dumps(configure(args.run, args.batch_size, args.bucket_elements), indent=2))


if __name__ == '__main__':
    main()
