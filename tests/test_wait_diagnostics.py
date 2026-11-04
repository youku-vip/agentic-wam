from contextlib import nullcontext
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from agenticwam.utils.wait_diagnostics import WaitDiagnostics, diagnostic_range, summarize_trace


class ToyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, sample):
        loss = self.linear(sample["x"]).square().mean()
        return loss, {"toy_loss": float(loss.detach())}


def make_toy_trainer(directory, accelerator):
    """Exercise the actual train loop without constructing the multi-billion-param model."""
    from agenticwam.trainer import AgenticWAMTrainer
    trainer = AgenticWAMTrainer.__new__(AgenticWAMTrainer)
    trainer.output_dir = directory
    trainer.accelerator = accelerator
    trainer.global_step = 0
    trainer.epoch = 0
    trainer.batch_in_epoch = 0
    trainer.max_steps = 3
    trainer.batch_size = 2
    trainer.gradient_accumulation_steps = 2
    trainer.num_workers = 0
    trainer.mixed_precision = "no"
    trainer.max_grad_norm = 1.0
    trainer.log_every = 0
    trainer.save_every = 0
    trainer.code_eval_every = 0
    trainer.eval_every = 0
    trainer.val_dataset = None
    torch.manual_seed(321)
    model = ToyTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loader = torch.utils.data.DataLoader(
        [{"x": torch.tensor([i/100, .5, 1.0])} for i in range(128)], batch_size=2,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=3)
    trainer.model, trainer.optimizer, trainer.train_loader, trainer.scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler,
    )
    trainer._set_dit_only_train_mode = lambda: trainer.model.train()
    trainer.save_checkpoint = lambda: {"weights_path": "mock-not-written", "state_path": None}
    return trainer


class TraceSummaryTests(unittest.TestCase):
    def event(self, name, cat, start, duration):
        return dict(name=name, cat=cat, ph="X", ts=start, dur=duration)

    def test_overlaps_are_unioned_and_nccl_is_separate(self):
        events = [
            self.event("waitdiag/microstep", "user_annotation", 0, 1000),
            self.event("gemm", "kernel", 100, 200),
            self.event("gemm", "kernel", 200, 200),
            self.event("ncclDevKernel_AllReduce", "kernel", 600, 200),
            self.event("HtoD", "gpu_memcpy", 50, 100),
        ]
        result = summarize_trace({"traceEvents": events})
        self.assertTrue(result["gpu_activity_available"])
        self.assertAlmostEqual(result["gpu_activity_covered_ms"], .55)
        self.assertAlmostEqual(result["non_nccl_kernel_covered_ms"], .3)
        self.assertAlmostEqual(result["nccl_kernel_covered_ms"], .2)
        self.assertAlmostEqual(result["uncovered_trace_window_fraction"], .45)

    def test_async_gpu_tail_is_in_window(self):
        result = summarize_trace({"traceEvents": [
            self.event("waitdiag/microstep", "user_annotation", 100, 100),
            self.event("gemm", "kernel", 150, 100),
        ]})
        self.assertAlmostEqual(result["trace_window_ms"], .15)
        self.assertAlmostEqual(result["uncovered_trace_window_fraction"], 1/3)

    def test_missing_cuda_events_is_unknown_not_zero(self):
        result = summarize_trace({"traceEvents": [
            self.event("waitdiag/microstep", "user_annotation", 0, 100),
            self.event("cudaLaunchKernel", "cuda_runtime", 10, 20),
        ]})
        self.assertFalse(result["gpu_activity_available"])
        self.assertNotIn("uncovered_trace_window_fraction", result)


class CaptureTests(unittest.TestCase):
    def test_actual_trainer_capture_and_disabled_weights_match(self):
        from accelerate import Accelerator
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        with TemporaryDirectory() as baseline_dir, TemporaryDirectory() as profiled_dir:
            with patch.dict(os.environ, {"AGENTICWAM_PHASE_TIMING": "0", "AGENTICWAM_WAIT_DIAG_STEPS": "0"}):
                baseline = make_toy_trainer(baseline_dir, accelerator)
                baseline.train()
            baseline_weights = {key: value.detach().clone() for key, value in baseline.model.state_dict().items()}
            baseline_rng = torch.get_rng_state().clone()
            with patch.dict(os.environ, {"AGENTICWAM_PHASE_TIMING": "0", "AGENTICWAM_WAIT_DIAG_STEPS": "1",
                                         "AGENTICWAM_WAIT_DIAG_WARMUP": "1"}):
                profiled = make_toy_trainer(profiled_dir, accelerator)
                profiled.train()
            self.assertEqual(profiled.global_step, 3)
            self.assertTrue(torch.equal(baseline_rng, torch.get_rng_state()))
            for key, value in profiled.model.state_dict().items():
                self.assertTrue(torch.equal(baseline_weights[key], value))
            summaries = list((Path(profiled_dir) / "wait_diagnostics").glob("*.summary.json"))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text())
            self.assertEqual(summary["captured_microsteps"], 2)
            self.assertFalse(summary["partial"])
            for name in ("data_next_including_h2d", "forward", "backward_including_comm",
                         "clip_and_optimizer_including_comm"):
                self.assertIn(name, summary["host_phases"])

    def test_disabled_does_not_construct_profiler_or_touch_rng(self):
        with TemporaryDirectory() as tmp, patch("torch.profiler.profile", side_effect=AssertionError("unexpected profiler")):
            rng = torch.get_rng_state().clone()
            with WaitDiagnostics(tmp) as diag:
                diag.before_batch(0)
                with diagnostic_range("forward"):
                    torch.ones(2).sum()
                diag.after_batch(1)
            self.assertFalse((Path(tmp) / "wait_diagnostics").exists())
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_bounded_capture_covers_all_accumulated_microsteps(self):
        with TemporaryDirectory() as tmp:
            with WaitDiagnostics(tmp, rank=1, active_steps=1, warmup_steps=1,
                                 metadata={"gradient_accumulation_steps": 2}) as diag:
                diag.before_batch(0)
                self.assertFalse(diag.capturing)
                diag.after_batch(1)
                for index in range(2):
                    diag.before_batch(1)
                    with diagnostic_range("forward"):
                        (torch.ones(16, requires_grad=True) ** 2).sum().backward()
                    diag.after_batch(1 + int(index == 1))
                self.assertTrue(diag.done)
                self.assertFalse(diag.capturing)
                diag.before_batch(2)
                diag.after_batch(3)
            self.assertIsNotNone(diag.summary_path)
            summary = json.loads(diag.summary_path.read_text())
            self.assertEqual(summary["rank"], 1)
            self.assertEqual(summary["captured_microsteps"], 2)
            self.assertEqual(summary["host_phases"]["forward"]["calls"], 2)
            self.assertFalse(summary["partial"])
            self.assertFalse(summary["timeline"]["gpu_activity_available"])
            trace_path = diag.summary_path.parent / summary["trace_file"]
            names = {event.get("name") for event in json.loads(trace_path.read_text())["traceEvents"]}
            self.assertIn("waitdiag/forward", names)
            self.assertEqual(len(list(diag.output_dir.glob("*.trace.json"))), 1)

    def test_exception_exports_partial_and_restores_scope(self):
        with TemporaryDirectory() as tmp:
            diag = WaitDiagnostics(tmp, active_steps=2, warmup_steps=0)
            with self.assertRaisesRegex(RuntimeError, "real training error"):
                with diag:
                    diag.before_batch(0)
                    with diagnostic_range("forward"):
                        torch.ones(4).sum()
                        raise RuntimeError("real training error")
            summary = json.loads(diag.summary_path.read_text())
            self.assertTrue(summary["partial"])
            self.assertIsInstance(diagnostic_range("after_close"), nullcontext)

    def test_capture_preserves_rng_loss_and_parameter_update(self):
        def run(directory, active_steps):
            torch.manual_seed(123)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            losses = []
            with WaitDiagnostics(directory, active_steps=active_steps, warmup_steps=0) as diag:
                for step in range(2):
                    diag.before_batch(step)
                    with diagnostic_range("forward"):
                        loss = model(torch.randn(4, 3)).square().mean()
                    with diagnostic_range("backward"):
                        loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    losses.append(float(loss.detach()))
                    diag.after_batch(step + 1)
            return losses, {key: value.detach().clone() for key, value in model.state_dict().items()}, torch.get_rng_state()
        with TemporaryDirectory() as tmp:
            baseline, base_weights, base_rng = run(tmp, 0)
            profiled, prof_weights, prof_rng = run(tmp, 1)
            self.assertEqual(baseline, profiled)
            self.assertTrue(torch.equal(base_rng, prof_rng))
            for name in base_weights:
                self.assertTrue(torch.equal(base_weights[name], prof_weights[name]))

    def test_invalid_settings_and_environment(self):
        with TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENTICWAM_WAIT_DIAG_STEPS": "1", "AGENTICWAM_WAIT_DIAG_WARMUP": "2"}):
                diag = WaitDiagnostics.from_env(tmp, start_step=100)
                self.assertEqual(diag.begin_step, 102)
                self.assertEqual(diag.active_steps, 1)
            with self.assertRaises(ValueError):
                WaitDiagnostics(tmp, active_steps=-1)


if __name__ == "__main__":
    unittest.main()
