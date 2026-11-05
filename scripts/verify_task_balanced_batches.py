#!/usr/bin/env python3
"""Read-only v3 task index + Accelerate batch-shard verification; no video/model."""
import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path

from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from agenticwam.datasets.lerobot.task_sampling import build_task_sampling_index
from agenticwam.datasets.lerobot.v3.dataset import V3MultiLeRobotDataset
from agenticwam.utils.samplers import TaskBalancedEpochSampler


def verify(dataset, batch_size=4, world_size=2, grad_accum=8, steps=3, seed=42):
    index = build_task_sampling_index(dataset)
    sampler = TaskBalancedEpochSampler(dataset, index, seed, batch_size, world_size, grad_accum)
    count = min(sampler.global_batches, steps * grad_accum)
    shards = [iter(BatchSamplerShard(BatchSampler(sampler, batch_size, drop_last=False),
                                    num_processes=world_size, process_index=rank,
                                    split_batches=False, even_batches=True)) for rank in range(world_size)]
    expected = {task: sampler.per_task_local for task in range(sampler.num_tasks)}
    updates, update_counts = [], Counter()
    for microstep in range(count):
        global_indices = []
        for shard in shards:
            batch = next(shard)
            actual = Counter(int(index.task_ids[item]) for item in batch)
            if actual != expected:
                raise AssertionError(f"Unbalanced rank-local batch: {actual}")
            global_indices.extend(batch)
            update_counts.update(actual)
        if len(global_indices) != len(set(global_indices)):
            raise AssertionError("Duplicate sample across ranks in a global microbatch")
        if (microstep + 1) % grad_accum == 0:
            updates.append(dict(sorted(update_counts.items())))
            update_counts.clear()
    original_tail = list(islice(iter(sampler), grad_accum * batch_size * world_size,
                               (grad_accum + 2) * batch_size * world_size))
    resumed = TaskBalancedEpochSampler(dataset, index, seed, batch_size, world_size, grad_accum)
    resumed.set_resume_batch_offset(grad_accum)
    resumed_tail = list(islice(iter(resumed), 2 * batch_size * world_size))
    if original_tail != resumed_tail:
        raise AssertionError("Resume sequence mismatch")
    raw = index.counts()
    return dict(passed=True, task_names=dict(enumerate(index.task_names)), raw_frame_counts=raw,
                raw_frame_fractions=[value / len(dataset) for value in raw],
                checked_microbatches_per_rank=count, expected_per_rank_microbatch=expected,
                observed_optimizer_batch_counts=updates, resume_prefix_check_passed=True,
                sampler=sampler.descriptor(),
                notes=["Reads selected Parquet task/episode/frame columns only; no video, normalizer or model is loaded.",
                       "Uses the same Accelerate no-split batch sharding as training; does not shard the sampler twice.",
                       "This checks index plans; training logs additionally count actual returned task IDs."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    root = str(args.root.resolve())
    dataset = V3MultiLeRobotDataset([root], episodes={root: args.episodes} if args.episodes else None)
    print(json.dumps(verify(dataset, args.batch_size, args.world_size, args.grad_accum, args.steps, args.seed),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
