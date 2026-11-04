#!/usr/bin/env python3
"""Read-only old/new loader comparison; model weights and training are not loaded.

Run with CUDA_VISIBLE_DEVICES="" so this CPU validator cannot occupy a training GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch

from agenticwam.datasets.lerobot.decoder_cache import decoder_cache_info
from agenticwam.utils.config_resolvers import register_default_resolvers


def fingerprint(value):
    if torch.is_tensor(value):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {key: fingerprint(item) for key, item in value.items()}
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return {"type": type(value).__name__, "sha256": hashlib.sha256(raw).hexdigest()}


class IndexRecorder:
    def __init__(self, base):
        self.base = base
        self.last_index = None

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    def __getitem__(self, index):
        sample = self.base[index]
        self.last_index = int(sample["idx"])
        return sample


class ComparisonPoints(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.dataset.lerobot_dataset = IndexRecorder(self.dataset.lerobot_dataset)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        seed = 1729 + index
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        sample = self.dataset[index]
        result = {
            "index": index, "actual_sample_index": self.dataset.lerobot_dataset.last_index,
            "fields": fingerprint(sample),
            "rng": {"torch": fingerprint(torch.get_rng_state()),
                    "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                    "numpy": hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()},
            "decoder_cache": decoder_cache_info(),
        }
        return result


def collect(dataset, indices, workers):
    wrapped = ComparisonPoints(dataset, indices)
    loader = torch.utils.data.DataLoader(wrapped, batch_size=None, num_workers=workers)
    return list(loader)


def compare_records(baseline, optimized):
    failures, worker_caches = [], {}
    for left, right in zip(baseline, optimized):
        mismatch = [key for key in set(left["fields"]) | set(right["fields"])
                    if left["fields"].get(key) != right["fields"].get(key)]
        if left["actual_sample_index"] != right["actual_sample_index"]:
            mismatch.append("actual_sample_index")
        if left["rng"] != right["rng"]:
            mismatch.append("rng_state_after_sample")
        if left["index"] != right["index"]:
            mismatch.append("requested_sample_index")
        if mismatch:
            failures.append({"index": left["index"], "mismatched_fields": sorted(mismatch)})
        info = right["decoder_cache"]
        worker_caches[str(info["pid"])] = info
    if len(baseline) != len(optimized):
        failures.append({"error": "different number of samples"})
    return {"passed": not failures, "checked_samples": len(baseline), "failures": failures,
            "optimized_cache_last_snapshot_by_pid": worker_caches,
            "example_field_fingerprints": optimized[0]["fields"] if optimized else {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/real_post_train/data/custom.yaml")
    parser.add_argument("--code-cache", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 0 or args.cache_size < 0:
        parser.error("workers/cache-size must be nonnegative")
    from torchcodec.decoders import VideoDecoder  # fail clearly if this backend is unavailable
    register_default_resolvers()
    cfg = OmegaConf.load(args.config)
    cfg.data.dataset_root = str(args.root.resolve())
    if args.code_cache is not None:
        cfg.data.code_cache = str(args.code_cache.resolve())
    stats_path = Path(cfg.data.train.pretrained_norm_stats)
    if not stats_path.is_file():
        parser.error(f"Existing statistics required: {stats_path}")
    kwargs = dict(video_backend="torchcodec", video_decoder_num_threads=1, save_norm_stats=False)
    dense = instantiate(cfg.data.train, sparse_video_decode=False, video_decoder_cache_size=0, **kwargs)
    fast = instantiate(cfg.data.train, sparse_video_decode=True, video_decoder_cache_size=args.cache_size, **kwargs)
    datasets = dense.lerobot_dataset.multi_dataset._datasets
    if len(datasets) != 1:
        parser.error("This verifier expects one dataset root")
    single = datasets[0]
    episodes = args.episodes or list(dict.fromkeys([
        single.episodes[0], single.episodes[len(single.episodes)//2], single.episodes[-1],
        min(single.episodes, key=single.meta.reader.episode_length),
    ]))
    indices = []
    for episode in episodes:
        if episode not in single.episodes:
            parser.error(f"Episode {episode} is not in the configured training selection")
        slot = single.episodes.index(episode)
        length = single.meta.reader.episode_length(episode)
        indices.extend(single._starts[slot] + position for position in sorted({0, length//2, length-1}))
    baseline = collect(dense, indices, args.workers)
    optimized = collect(fast, indices, args.workers)
    report = compare_records(baseline, optimized)
    report.update(episodes=episodes, workers=args.workers, cache_size=args.cache_size,
                  notes=["Compares hashes of final model inputs, targets, masks, code/text and post-sample RNG states.",
                         "No input data, code cache, statistics or model weights are written.",
                         "workers=0 intentionally does not retain native decoders in the parent process.",
                         "This is a correctness check, not a throughput benchmark."])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
