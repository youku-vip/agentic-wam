"""Avoid repeated Python parameter scans in affected DeepSpeed ZeRO 1/2 hooks.

Counts come from DeepSpeed's original helper once per active graph task and
stable hook parameter list. Never substitute a fixed number of model parameters.
The cache contains only integer keys/counts, not Parameters or graph nodes.
"""
from collections import OrderedDict
import functools
import inspect
import logging
import threading

import torch

logger = logging.getLogger(__name__)


class GraphTaskCountCache:
    def __init__(self, original, max_entries=64):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.original = original
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = 0
        self.original_calls = 0
        self.lock = threading.Lock()
        functools.update_wrapper(self, original)

    def __call__(self, parameters):
        graph_id = int(torch._C._current_graph_task_id())
        if graph_id < 0 or not isinstance(parameters, (list, tuple)):
            return self.original(parameters)
        # In the matched DeepSpeed hook this list is captured once at hook
        # creation and remains alive/stable for the optimizer's lifetime.
        key = (graph_id, id(parameters), len(parameters))
        with self.lock:
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                return self.entries[key]
        # Do not hold a Python lock across PyTorch/autograd API calls. Concurrent
        # misses may duplicate one scan, but never wait on another graph's work.
        count = self.original(parameters)
        with self.lock:
            self.original_calls += 1
            self.entries[key] = count
            self.entries.move_to_end(key)
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        return count

    def cache_info(self):
        with self.lock:
            return {"original_calls": self.original_calls, "hits": self.hits,
                    "entries": len(self.entries), "max_entries": self.max_entries}


def install_graph_count_cache(zero_stage):
    """Patch the process-local symbol only for the known repeated-scan hook."""
    if int(zero_stage) not in (1, 2):
        logger.warning("[ds-graph-cache] skipped: requires ZeRO stage 1 or 2")
        return None
    from deepspeed.runtime.zero import stage_1_and_2
    original = getattr(stage_1_and_2, "count_used_parameters_in_backward", None)
    if isinstance(original, GraphTaskCountCache):
        return original
    try:
        source = inspect.getsource(stage_1_and_2.DeepSpeedZeroOptimizer.create_gradient_handling_hooks)
    except (OSError, TypeError):
        source = ""
    if (not callable(original)
            or "count_used_parameters_in_backward(all_params_requiring_grad)" not in source
            or not hasattr(torch._C, "_current_graph_task_id")):
        logger.warning("[ds-graph-cache] skipped: installed DeepSpeed hook does not match the supported implementation")
        return None
    cached = GraphTaskCountCache(original)
    stage_1_and_2.count_used_parameters_in_backward = cached
    logger.warning("[ds-graph-cache] enabled: original count cached per backward graph task and stable parameter list; no fixed parameter count")
    return cached
