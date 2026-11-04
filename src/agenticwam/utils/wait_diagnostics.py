"""Opt-in host timings/progress, with explicit optional CPU/CUDA profiling.

No extra CUDA synchronization or distributed barriers are inserted by this module.
Host mode never starts torch.profiler; profiler modes must be requested explicitly.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import faulthandler
import json
import logging
import os
from pathlib import Path
import tempfile
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
                 active_steps=0, warmup_steps=1, metadata=None, mode="host", timeout_seconds=90):
        if active_steps < 0 or warmup_steps < 0:
            raise ValueError("Wait diagnostic steps and warmup must be nonnegative")
        if mode not in {"host", "cpu", "cuda"}:
            raise ValueError("Wait diagnostic mode must be host, cpu or cuda")
        if timeout_seconds < 0:
            raise ValueError("Wait diagnostic timeout must be nonnegative")
        self.mode = mode
        self.timeout_seconds = timeout_seconds
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
        self._in_microstep = False
        self._phase_stack = []
        self._last_step = start_step
        self.summary_path = None
        self.progress_path = None
        self.stacks_path = None
        self._stack_stream = None
        self._watchdog_armed = False
        self._watchdog_attempted = False

    @classmethod
    def from_env(cls, output_dir, **kwargs):
        return cls(output_dir, active_steps=int(os.environ.get("AGENTICWAM_WAIT_DIAG_STEPS", "0")),
                   warmup_steps=int(os.environ.get("AGENTICWAM_WAIT_DIAG_WARMUP", "1")),
                   mode=os.environ.get("AGENTICWAM_WAIT_DIAG_MODE", "host").strip().lower(),
                   timeout_seconds=float(os.environ.get("AGENTICWAM_WAIT_DIAG_TIMEOUT", "90")), **kwargs)

    def __enter__(self):
        if not self.done:
            self._token = _CURRENT.set(self)
            try:
                local_dir = Path(tempfile.mkdtemp(
                    prefix=f"agenticwam-waitdiag-rank{self.rank}-pid{os.getpid()}-", dir="/tmp"))
                self.progress_path = local_dir / "progress.json"
                self.stacks_path = local_dir / "stacks.log"
                self._write_progress("waiting_for_first_batch")
            except OSError as exc:
                logger.warning("[wait-diag] rank=%d cannot create local progress files: %s", self.rank, exc)
            logger.warning("[wait-diag] rank=%d mode=%s: capture %d optimizer step(s) after %d warmup step(s); progress=%s stacks=%s",
                           self.rank, self.mode, self.active_steps, self.warmup_steps,
                           self.progress_path, self.stacks_path)
        return self

    def _write_progress(self, event):
        if self.progress_path is None:
            return
        try:
            payload = dict(rank=self.rank, pid=os.getpid(), mode=self.mode, event=event,
                           optimizer_step=self._last_step, phase_stack=list(self._phase_stack),
                           captured_microsteps=self.microsteps, updated_unix_time=time.time())
            temporary = self.progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
            temporary.replace(self.progress_path)
        except OSError:
            # Diagnostics must not turn a failed progress write into a training error.
            pass

    def _start_watchdog(self):
        if self._watchdog_attempted:
            return
        self._watchdog_attempted = True
        if self.stacks_path is None or self.timeout_seconds == 0:
            return
        try:
            self._stack_stream = self.stacks_path.open("a", encoding="utf-8")
            faulthandler.dump_traceback_later(self.timeout_seconds, repeat=True, file=self._stack_stream)
            self._watchdog_armed = True
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("[wait-diag] rank=%d cannot arm Python stack watchdog: %s", self.rank, exc)

    def _stop_watchdog(self):
        if self._watchdog_armed:
            faulthandler.cancel_dump_traceback_later()
            self._watchdog_armed = False
        if self._stack_stream is not None:
            self._stack_stream.close()
            self._stack_stream = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(partial=exc_type is not None)
        return False

    def before_batch(self, step):
        if self.done:
            return
        # DataLoader workers have already been started by iter(loader). Avoid
        # starting the watchdog thread before the initial worker fork.
        self._start_watchdog()
        self._last_step = step
        if step < self.begin_step:
            self._write_progress("warmup")
            return
        if not self.capturing:
            try:
                if self.mode != "host":
                    self._write_progress("profiler_start_begin")
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if self.mode == "cuda" and self.device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    self.profile = torch.profiler.profile(
                        activities=activities, record_shapes=False, profile_memory=False, with_stack=False,
                    )
                    self.profile.start()
                    self._write_progress("profiler_start_end")
                self.capturing = True
                if self._token is None:
                    self._token = _CURRENT.set(self)
            except Exception as exc:
                self.done = True
                self._write_progress("profiler_start_failed")
                if self._token is not None:
                    _CURRENT.reset(self._token)
                    self._token = None
                self._stop_watchdog()
                self.profile = None
                logger.warning("[wait-diag] rank=%d cannot start profiler: %s", self.rank, exc)
                return
        self._in_microstep = True
        if self.profile is not None:
            self._micro_range = torch.profiler.record_function(PREFIX + "microstep")
            self._micro_range.__enter__()
        self._write_progress("microstep_begin")

    @contextmanager
    def range(self, name):
        self._phase_stack.append(name)
        self._write_progress("phase_enter")
        was_capturing = self.capturing
        started = time.perf_counter()
        try:
            scope = torch.profiler.record_function(PREFIX + name) if self.profile is not None else nullcontext()
            with scope:
                yield
        finally:
            if was_capturing:
                self.phases[name].append(1000 * (time.perf_counter() - started))
            self._phase_stack.pop()
            self._write_progress("phase_exit")

    def _finish_microstep(self):
        if self._micro_range is not None:
            self._micro_range.__exit__(None, None, None)
            self._micro_range = None
        if self._in_microstep:
            self.microsteps += 1
            self._in_microstep = False

    def after_batch(self, step):
        if self.done:
            return
        if not self.capturing:
            self._last_step = step
            self._write_progress("warmup_microstep_end")
            return
        self._last_step = step
        self._finish_microstep()
        if step >= self.begin_step + self.active_steps:
            self.close()

    def close(self, *, partial=False):
        if not self.capturing:
            if self._token is not None:
                _CURRENT.reset(self._token)
                self._token = None
            self._stop_watchdog()
            return
        self._finish_microstep()
        if self._token is not None:
            _CURRENT.reset(self._token)
            self._token = None
        self.capturing = False
        self.done = True
        try:
            trace_path = None
            timeline = {"gpu_activity_available": False}
            if self.profile is not None:
                self._write_progress("profiler_stop_begin")
                self.profile.stop()
                self._write_progress("profiler_stop_end")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"rank{self.rank}-step{self.begin_step:06d}-{time.time_ns()}"
            if self.profile is not None:
                trace_path = self.output_dir / (stem + ".trace.json")
                self.profile.export_chrome_trace(str(trace_path))
                # Never reconstruct Kineto FunctionEvents/key_averages here.
                # Real two-GPU captures spent a long time in _parse_kineto_results
                # on BOTH training threads while the GPUs sat idle. Export only;
                # even raw JSON analysis belongs in a separate, offline process.
                timeline = {"gpu_activity_available": None, "analysis_deferred": True}
            phases = {
                name: {"calls": len(values), "total_ms": sum(values),
                       "p50_ms": _percentile(values, .5), "p95_ms": _percentile(values, .95)}
                for name, values in self.phases.items()
            }
            summary = {
                "rank": self.rank, "device": str(self.device), "start_step": self.begin_step,
                "capture_mode": self.mode,
                "end_step": self._last_step, "captured_microsteps": self.microsteps,
                "partial": partial or self._last_step < self.begin_step + self.active_steps,
                "metadata": self.metadata, "host_phases": phases,
                "timeline": timeline,
                "top_self_cpu_ops": [],
                "trace_file": trace_path.name if trace_path else None,
                "progress_file": str(self.progress_path), "stacks_file": str(self.stacks_path),
                "notes": [
                    "Host phases include CPU work, submission and blocking; they are not isolated GPU compute times.",
                    "Nested phases overlap; do not sum all phase totals.",
                    "GPU timeline coverage is not SM utilization. NCCL kernels can be waiting.",
                    "No CUDA events means GPU activity unavailable, not 0% utilization.",
                    "Host mode does not start torch.profiler, collect CUDA events, or synchronize CUDA.",
                    "Raw trace analysis is deferred. Native key_averages is never called in the training process.",
                    "Diagnostics add overhead; compare throughput in a separate run with diagnosis disabled.",
                ],
            }
            self.summary_path = self.output_dir / (stem + ".summary.json")
            self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            self._write_progress("capture_complete")
            logger.warning("[wait-diag] rank=%d wrote %s", self.rank, self.summary_path)
        except Exception as exc:
            logger.warning("[wait-diag] rank=%d cannot finish/export profiler: %s", self.rank, exc)
        finally:
            self.profile = None
            self._stop_watchdog()
