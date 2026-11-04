"""Opt-in, bounded per-rank CPU/CUDA timeline capture for training stalls.

No extra CUDA synchronization or distributed barriers are inserted by this module.
Profiler startup/shutdown does have overhead; do not use its throughput as a benchmark.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import json
import logging
import os
from pathlib import Path
import time

import torch

logger = logging.getLogger(__name__)
_CURRENT = ContextVar("agenticwam_wait_diagnostic", default=None)
PREFIX = "waitdiag/"


def diagnostic_range(name):
    diagnostic = _CURRENT.get()
    return diagnostic.range(name) if diagnostic is not None else nullcontext()


def _union_us(intervals):
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if end is None or left > end:
            total += right - left
        elif right > end:
            total += right - end
        end = max(right, end if end is not None else right)
    return total


def summarize_trace(trace):
    """Chrome timestamps are microseconds. GPU coverage is not SM utilization.

    NCCL kernels can spend time waiting. Report them separately instead of calling
    all kernel time useful compute. With no CUDA events, never report zero GPU use.
    """
    events = trace.get("traceEvents", [])
    windows, kernels, nccl, non_nccl, copies = [], [], [], [], []
    for event in events:
        if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
            continue
        interval = (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
        name, category = event.get("name", ""), event.get("cat", "")
        if name == PREFIX + "microstep":
            windows.append(interval)
        if category == "kernel":
            kernels.append(interval)
            (nccl if "nccl" in name.lower() else non_nccl).append(interval)
        elif category in {"gpu_memcpy", "gpu_memset"}:
            copies.append(interval)
    gpu = kernels + copies
    result = {"gpu_activity_available": bool(gpu)}
    if windows:
        result["host_microstep_window_ms"] = _union_us(windows) / 1000
    if not gpu or not windows:
        return result
    # Async work may outlive the final host range. Include its tail in the window.
    start = min(left for left, _ in windows + gpu)
    end = max(right for _, right in windows + gpu)
    duration = end - start
    result.update(
        trace_window_ms=duration / 1000,
        gpu_activity_covered_ms=_union_us(gpu) / 1000,
        gpu_kernel_covered_ms=_union_us(kernels) / 1000,
        nccl_kernel_covered_ms=_union_us(nccl) / 1000,
        non_nccl_kernel_covered_ms=_union_us(non_nccl) / 1000,
        gpu_copy_or_memset_covered_ms=_union_us(copies) / 1000,
        uncovered_trace_window_fraction=1 - _union_us(gpu) / duration if duration > 0 else 0,
    )
    return result


def _percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left = int(index)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (index - left)


class WaitDiagnostics:
    def __init__(self, output_dir, *, rank=0, device="cpu", start_step=0,
                 active_steps=0, warmup_steps=1, metadata=None):
        if active_steps < 0 or warmup_steps < 0:
            raise ValueError("Wait diagnostic steps and warmup must be nonnegative")
        self.active_steps = active_steps
        self.warmup_steps = warmup_steps
        self.start_step = start_step
        self.begin_step = start_step + warmup_steps
        self.rank = rank
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) / "wait_diagnostics"
        self.metadata = metadata or {}
        self.profile = None
        self.capturing = False
        self.done = active_steps == 0
        self.phases = defaultdict(list)
        self.microsteps = 0
        self._token = None
        self._micro_range = None
        self._last_step = start_step
        self.summary_path = None

    @classmethod
    def from_env(cls, output_dir, **kwargs):
        return cls(output_dir, active_steps=int(os.environ.get("AGENTICWAM_WAIT_DIAG_STEPS", "0")),
                   warmup_steps=int(os.environ.get("AGENTICWAM_WAIT_DIAG_WARMUP", "1")), **kwargs)

    def __enter__(self):
        if not self.done:
            logger.warning("[wait-diag] rank=%d: capture %d optimizer step(s) after %d warmup step(s); profiler adds overhead",
                           self.rank, self.active_steps, self.warmup_steps)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(partial=exc_type is not None)
        return False

    def before_batch(self, step):
        if self.done or step < self.begin_step:
            return
        if not self.capturing:
            try:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if self.device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                self.profile = torch.profiler.profile(
                    activities=activities, record_shapes=False, profile_memory=False, with_stack=False,
                )
                self.profile.start()
                self.capturing = True
                self._token = _CURRENT.set(self)
            except Exception as exc:
                self.done = True
                logger.warning("[wait-diag] rank=%d cannot start profiler: %s", self.rank, exc)
                return
        self._last_step = step
        self._micro_range = torch.profiler.record_function(PREFIX + "microstep")
        self._micro_range.__enter__()

    @contextmanager
    def range(self, name):
        started = time.perf_counter()
        try:
            with torch.profiler.record_function(PREFIX + name):
                yield
        finally:
            self.phases[name].append(1000 * (time.perf_counter() - started))

    def _finish_microstep(self):
        if self._micro_range is not None:
            self._micro_range.__exit__(None, None, None)
            self._micro_range = None
            self.microsteps += 1

    def after_batch(self, step):
        if not self.capturing:
            return
        self._last_step = step
        self._finish_microstep()
        if step >= self.begin_step + self.active_steps:
            self.close()

    def close(self, *, partial=False):
        if not self.capturing:
            return
        self._finish_microstep()
        if self._token is not None:
            _CURRENT.reset(self._token)
            self._token = None
        self.capturing = False
        self.done = True
        try:
            self.profile.stop()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"rank{self.rank}-step{self.begin_step:06d}-{time.time_ns()}"
            trace_path = self.output_dir / (stem + ".trace.json")
            self.profile.export_chrome_trace(str(trace_path))
            trace = json.loads(trace_path.read_text())
            phases = {
                name: {"calls": len(values), "total_ms": sum(values),
                       "p50_ms": _percentile(values, .5), "p95_ms": _percentile(values, .95)}
                for name, values in self.phases.items()
            }
            operators = sorted(self.profile.key_averages(), key=lambda event: event.self_cpu_time_total, reverse=True)
            summary = {
                "rank": self.rank, "device": str(self.device), "start_step": self.begin_step,
                "end_step": self._last_step, "captured_microsteps": self.microsteps,
                "partial": partial or self._last_step < self.begin_step + self.active_steps,
                "metadata": self.metadata, "host_phases": phases,
                "timeline": summarize_trace(trace),
                "top_self_cpu_ops": [{"name": event.key, "self_cpu_ms": event.self_cpu_time_total / 1000,
                                      "calls": event.count} for event in operators[:25]],
                "trace_file": trace_path.name,
                "notes": [
                    "Host phases include CPU work, submission and blocking; they are not isolated GPU compute times.",
                    "Nested phases overlap; do not sum all phase totals.",
                    "GPU timeline coverage is not SM utilization. NCCL kernels can be waiting.",
                    "No CUDA events means GPU activity unavailable, not 0% utilization.",
                    "Profiler changes runtime; compare throughput in a separate run with diagnosis disabled.",
                ],
            }
            self.summary_path = self.output_dir / (stem + ".summary.json")
            self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            logger.warning("[wait-diag] rank=%d wrote %s", self.rank, self.summary_path)
        except Exception as exc:
            logger.warning("[wait-diag] rank=%d cannot finish/export profiler: %s", self.rank, exc)
        finally:
            self.profile = None
