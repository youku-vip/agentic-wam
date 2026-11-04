#!/usr/bin/env python3
"""Read existing Chrome traces offline, without torch.profiler or Kineto parsing.

PYTHONPATH=src python scripts/summarize_wait_trace.py path/to/rank*.trace.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from agenticwam.utils.wait_diagnostics import PREFIX, summarize_trace


def analyze_trace(trace):
    phases = defaultdict(lambda: {"calls": 0, "total_host_ms": 0.0})
    cuda_runtime = defaultdict(lambda: {"calls": 0, "total_host_ms": 0.0})
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X" or "dur" not in event:
            continue
        name = event.get("name", "")
        group = None
        if name.startswith(PREFIX) and name != PREFIX + "microstep":
            name = name[len(PREFIX):]
            group = phases
        elif event.get("cat") == "cuda_runtime":
            group = cuda_runtime
        if group is not None:
            group[name]["calls"] += 1
            group[name]["total_host_ms"] += float(event["dur"]) / 1000
    return {
        "timeline": summarize_trace(trace),
        "host_phases_inclusive": dict(sorted(phases.items(), key=lambda item: item[1]["total_host_ms"], reverse=True)),
        "top_cuda_runtime_host_times": dict(sorted(cuda_runtime.items(), key=lambda item: item[1]["total_host_ms"], reverse=True)[:20]),
        "notes": [
            "Offline JSON analysis only; no native profiler key_averages/FunctionEvent reconstruction.",
            "Host phase times are inclusive and nested; do not sum all phase totals.",
            "A long synchronization API can wait for useful GPU work. Compare with the GPU timeline.",
            "NCCL kernel coverage is not useful compute, and GPU event coverage is not SM utilization.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    args = parser.parse_args()
    captures = []
    for path in args.traces:
        with path.open(encoding="utf-8") as stream:
            trace = json.load(stream)
        captures.append({"file": str(path), **analyze_trace(trace)})
        del trace
    print(json.dumps({"captures": captures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
