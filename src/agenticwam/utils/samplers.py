from typing import Iterator, Sized
from math import ceil

import numpy as np
import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class TaskBalancedEpochSampler(ResumableEpochSampler):
    """Emit full balanced LOCAL batches; Accelerate shards these batches once.

    A logical global microbatch comprises num_processes consecutive local
    batches. Per-task draws are disjoint across that global microbatch. Smaller
    tasks are reshuffled/cycled; epoch length stays near the dataset length.
    """

    def __init__(self, dataset, task_index, seed, batch_size, num_processes, gradient_accumulation_steps=1):
        super().__init__(dataset, seed, batch_size, num_processes)
        self.task_index = task_index
        self.task_names = task_index.task_names
        self.num_tasks = len(self.task_names)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if self.batch_size < 1 or self.num_processes < 1 or self.num_tasks < 1:
            raise ValueError("Batch size, world size and task count must be positive")
        if len(task_index.task_ids) != len(dataset):
            raise ValueError("Task index length does not match the training dataset")
        if not np.array_equal(np.unique(task_index.task_ids), np.arange(self.num_tasks)):
            raise ValueError("Task IDs must cover every named task and be dense from zero")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("Gradient accumulation steps must be positive")
        if self.batch_size % self.num_tasks:
            raise ValueError(f"Per-GPU batch_size={self.batch_size} must be divisible by task count={self.num_tasks}")
        self.per_task_local = self.batch_size // self.num_tasks
        self.per_task_global = self.per_task_local * self.num_processes
        self.groups = [torch.as_tensor(np.flatnonzero(task_index.task_ids == task), dtype=torch.long)
                       for task in range(self.num_tasks)]
        if any(len(group) < self.per_task_global for group in self.groups):
            raise ValueError("Each task needs enough distinct samples for one global microbatch")
        effective_batch = self.batch_size * self.num_processes * self.gradient_accumulation_steps
        self.global_batches = ceil(len(dataset) / effective_batch) * self.gradient_accumulation_steps
        self.epoch_samples = self.global_batches * self.batch_size * self.num_processes

    def descriptor(self):
        return dict(kind="task_balanced_v1", seed=self.seed, batch_size=self.batch_size,
                    world_size=self.num_processes, epoch_samples=self.epoch_samples,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    task_names=list(self.task_names), task_index_sha256=self.task_index.sha256)

    def __iter__(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
        pools = [torch.empty(0, dtype=torch.long) for _ in self.groups]
        positions = [0] * self.num_tasks
        skip = self.resume_batch_offset if self.epoch == 0 else 0
        for global_batch in range(self.global_batches):
            draws = []
            for task, group in enumerate(self.groups):
                if positions[task] + self.per_task_global > len(pools[task]):
                    pools[task] = group[torch.randperm(len(group), generator=generator)]
                    positions[task] = 0
                start = positions[task]
                draws.append(pools[task][start:start + self.per_task_global])
                positions[task] += self.per_task_global
            for rank_slot in range(self.num_processes):
                start = rank_slot * self.per_task_local
                batch = torch.cat([draw[start:start + self.per_task_local] for draw in draws])
                batch = batch[torch.randperm(self.batch_size, generator=generator)]
                # Consume RNG for skipped batches too, so a resumed run has the
                # exact same remaining sequence as an uninterrupted run.
                if global_batch >= skip:
                    yield from batch.tolist()

    def __len__(self):
        skip = self.resume_batch_offset if self.epoch == 0 else 0
        return max(0, self.epoch_samples - skip * self.batch_size * self.num_processes)
