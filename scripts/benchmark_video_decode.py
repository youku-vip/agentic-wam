#!/usr/bin/env python3
"""Read-only TorchCodec CPU benchmark: dense/sparse queries and decoder reuse.

This is a single-file prototype, not a training loader patch. It measures the
existing decode function, substituting only the decoder factory to compare
fresh construction with a single retained decoder. No GPU decoding is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from agenticwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames_torchcodec


def fingerprint(frames):
    return hashlib.sha256(frames.contiguous().numpy().tobytes()).hexdigest()


def run_benchmark(video, *, windows=16, warmup=2, frames=33, stride=4, threads=1, seed=42):
    if min(windows, frames, stride) < 1 or min(warmup, threads) < 0:
        raise ValueError("windows/frames/stride must be positive; warmup/threads nonnegative")
    from torchcodec.decoders import VideoDecoder
    video = Path(video).resolve(strict=True)
    metadata_decoder = VideoDecoder(str(video), device="cpu", seek_mode="approximate", num_ffmpeg_threads=threads)
    metadata = metadata_decoder.metadata
    fps, count = float(metadata.average_fps), len(metadata_decoder)
    if count < frames:
        raise ValueError(f"Need at least {frames} frames; video contains {count}")
    video_info = dict(path=str(video), codec=metadata.codec, fps=fps, frames=count,
                      width=metadata.width, height=metadata.height)
    del metadata_decoder
    starts = np.random.default_rng(seed).integers(0, count - frames + 1, size=windows + warmup).tolist()
    baseline_hashes = None
    results = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # match the normal PyTorch DataLoader worker setting
    try:
        for sparse, reuse in ((False, False), (True, False), (False, True), (True, True)):
            offsets = list(range(0, frames, stride if sparse else 1))
            retained = None
            constructors = 0
            def factory(*args, **kwargs):
                nonlocal retained, constructors
                if reuse and retained is not None:
                    return retained
                kwargs["num_ffmpeg_threads"] = threads
                decoder = VideoDecoder(*args, **kwargs)
                constructors += 1
                if reuse:
                    retained = decoder
                return decoder
            elapsed, hashes = [], []
            with patch("torchcodec.decoders.VideoDecoder", side_effect=factory):
                for number, start in enumerate(starts):
                    timestamps = [(start + offset) / fps for offset in offsets]
                    begin = time.perf_counter()
                    decoded = decode_video_frames_torchcodec(str(video), timestamps, 1/fps - 1e-4)
                    duration = time.perf_counter() - begin
                    if number >= warmup:
                        elapsed.append(duration)
                        hashes.append(fingerprint(decoded if sparse else decoded[::stride]))
                    del decoded
            if baseline_hashes is None:
                baseline_hashes = hashes
            matches = sum(left == right for left, right in zip(baseline_hashes, hashes))
            results.append(dict(
                case=("reuse" if reuse else "fresh") + ("_sparse" if sparse else "_dense"),
                requested_frames=len(offsets), decoder_constructions_including_warmup=constructors,
                windows=windows, decode_seconds_total=sum(elapsed),
                window_seconds_p50=float(np.median(elapsed)),
                window_seconds_p95=float(np.quantile(elapsed, .95)),
                windows_per_second=windows/sum(elapsed),
                selected_frame_hash_matches=matches,
                selected_frame_hash_checks=windows,
                returned_float32_frame_bytes=len(offsets)*3*video_info["width"]*video_info["height"]*4,
            ))
            retained = None
    finally:
        torch.set_num_threads(previous_threads)
    baseline_seconds = results[0]["decode_seconds_total"]
    for row in results:
        row["speedup_vs_fresh_dense"] = baseline_seconds / row["decode_seconds_total"]
    return dict(
        video=video_info, backend="torchcodec", ffmpeg_threads_per_decoder=threads,
        torch_cpu_threads=1, warmup_windows=warmup, random_seed=seed,
        selected_frames_per_window=len(range(0, frames, stride)), results=results,
        all_selected_frames_identical=all(row["selected_frame_hash_matches"] == windows for row in results),
        limits=[
            "Single CPU process and one video file; retained decoder cases are an upper bound with 100% file-cache hits.",
            "Times cover existing decode calls only; hashes are checked outside timing. No image transforms, text loads, H2D, model or communication.",
            "Warmup favors a hot OS page cache. A random multi-worker/multi-file training loader may have different results.",
            "Sparse output does not avoid all inter-frame/GOP decoding; do not infer 33/9 training speedup.",
            "The benchmark uses the current round(timestamp*average_fps) and timestamp tolerance rules; it is not a timestamp convention change.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    report = run_benchmark(args.video, windows=args.windows, warmup=args.warmup,
                           frames=args.frames, stride=args.stride, threads=args.threads)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_selected_frames_identical"]:
        raise SystemExit("Frame mismatch: do not enable this optimization without resolving alignment")


if __name__ == "__main__":
    main()
