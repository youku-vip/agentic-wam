import gc
import unittest
from unittest.mock import patch
import weakref

import torch
from torch.utils.checkpoint import checkpoint

from agenticwam.utils.deepspeed_graph_count_cache import GraphTaskCountCache, install_graph_count_cache


class GraphCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from deepspeed.runtime.utils import count_used_parameters_in_backward
        cls.original = staticmethod(count_used_parameters_in_backward)

    def exercise(self, cached):
        parameters = [torch.nn.Parameter(torch.tensor(float(i+1))) for i in range(6)]
        counter = GraphTaskCountCache(self.original) if cached else self.original
        seen = []
        handles = [parameter.register_post_accumulate_grad_hook(lambda _, c=counter: seen.append(c(parameters)))
                   for parameter in parameters]
        losses, gradients = [], []
        for selected in ((0, 2, 4), (1, 3), (0, 1, 2, 3, 4, 5)):
            for parameter in parameters:
                parameter.grad = None
            loss = sum(parameters[index].square() for index in selected)
            losses.append(float(loss.detach()))
            seen.clear()
            loss.backward()
            self.assertEqual(seen, [len(selected)] * len(selected))
            gradients.append([None if parameter.grad is None else float(parameter.grad) for parameter in parameters])
        for handle in handles:
            handle.remove()
        if cached:
            self.assertEqual(counter.cache_info()["original_calls"], 3)
            self.assertEqual(counter.cache_info()["hits"], 8)
        return losses, gradients

    def test_used_counts_dynamic_unused_parameters_and_gradients_match(self):
        self.assertEqual(self.exercise(False), self.exercise(True))

    def test_distinct_parameter_lists_in_same_graph(self):
        left = [torch.nn.Parameter(torch.ones(())) for _ in range(2)]
        right = [torch.nn.Parameter(torch.ones(())) for _ in range(3)]
        counter = GraphTaskCountCache(self.original)
        values = []
        handles = [parameter.register_post_accumulate_grad_hook(lambda _, group=group: values.append(counter(group)))
                   for group in (left, right) for parameter in group]
        sum(parameter.square() for parameter in left+right).backward()
        self.assertEqual(sorted(values), [2, 2, 3, 3, 3])
        self.assertEqual(counter.cache_info()["original_calls"], 2)
        for handle in handles:
            handle.remove()

    def test_reentrant_checkpoint_graphs_and_bounded_cache(self):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        parameters = [parameter]
        counter = GraphTaskCountCache(self.original, max_entries=2)
        seen = []
        handle = parameter.register_post_accumulate_grad_hook(lambda _: seen.append(counter(parameters)))
        for _ in range(4):
            parameter.grad = None
            value = torch.tensor(3.0, requires_grad=True)
            loss = checkpoint(lambda x: x * parameter, value, use_reentrant=True).square()
            loss.backward()
            self.assertEqual(float(parameter.grad), 36)
        self.assertEqual(seen, [1, 1, 1, 1])
        self.assertEqual(counter.cache_info()["entries"], 2)
        handle.remove()

    def test_no_parameter_or_graph_retention_and_outside_backward_behavior(self):
        parameters = [torch.nn.Parameter(torch.ones(()))]
        counter = GraphTaskCountCache(self.original)
        reference = weakref.ref(parameters[0])
        def record_count(_):
            counter(parameters)
        handle = parameters[0].register_post_accumulate_grad_hook(record_count)
        parameters[0].square().backward()
        handle.remove()
        with self.assertRaises(RuntimeError):
            counter(parameters)
        parameters.clear()
        gc.collect()
        self.assertIsNone(reference())

    def test_install_is_guarded_and_idempotent(self):
        from deepspeed.runtime.zero import stage_1_and_2
        original = stage_1_and_2.count_used_parameters_in_backward
        with patch.object(stage_1_and_2, "count_used_parameters_in_backward", original):
            first = install_graph_count_cache(1)
            self.assertIsInstance(first, GraphTaskCountCache)
            self.assertIs(first, install_graph_count_cache(1))
        self.assertIsNone(install_graph_count_cache(3))


if __name__ == "__main__":
    unittest.main()
