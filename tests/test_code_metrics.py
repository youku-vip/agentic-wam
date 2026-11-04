"""CPU regression tests: PYTHONPATH=src python -m unittest discover -s tests -v."""
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from agenticwam.utils.code_metrics import (
    COUNT_KEYS, count_code_pair, probe_code_batch, summarize_code_counts,
)


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class FakeModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.dit = torch.nn.Linear(1, 1)
        self.frozen = torch.nn.Linear(1, 1).eval()
        self.n_cameras = 3
        self.code_expert = SimpleNamespace(tokenizer=CharacterTokenizer())
        self.video_scheduler = SimpleNamespace(sigmas=torch.tensor([0.25]), marker="training")
        self.action_scheduler = SimpleNamespace(steps=[1, 2])
        self.code_memory_enabled = True
        self._mem_code_history = "original history"
        self._cfg_mode = "training mode"
        self.seen = []
        self.fail = fail

    @property
    def device(self):
        return self.dit.weight.device

    def _resolve_infer_code_tokens(self):
        return 20

    def infer_action_trimodal(self, **kwargs):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert "policy_code" not in kwargs
        assert kwargs["code_cache"] is None
        self.seen.append((kwargs, self._mem_code_history))
        torch.rand(7)
        self.video_scheduler.sigmas.fill_(99)
        self.video_scheduler.extra = "inference"
        self.action_scheduler.steps.append(99)
        self._mem_code_history = "generated history"
        self._cfg_mode = "changed"
        self._last_code_canvas = torch.ones(3)
        if self.fail:
            raise RuntimeError("fake inference failure")
        return torch.zeros(2, 4), "x = 1"


def batch():
    return {
        "video": torch.arange(2*3*9*2*2).reshape(2, 3, 9, 2, 2).float(),
        "proprio": torch.arange(2*4*5).reshape(2, 4, 5).float(),
        "action": torch.ones(2, 4, 5), "prompt_embs": torch.zeros(2, 3, 4),
        "prompt": ["generic one", "generic two"],
        "prompt_code": ["code one", "code two"],
        "prompt_nohist": ["memory one", "memory two"],
        "code_history": ["previous one", "previous two"],
        "policy_code": ["x = 1", "x = 2"],
    }


class CountTests(unittest.TestCase):
    def score(self, pred, target, canvas=100):
        return summarize_code_counts(count_code_pair(pred, target, CharacterTokenizer(), canvas))

    def test_exact_and_outer_whitespace(self):
        scores = self.score("\nx = 1\n", "x = 1")
        self.assertEqual(scores["exact_match"], 1)
        self.assertEqual(scores["token_accuracy"], 1)
        self.assertEqual(scores["syntax_valid_rate"], 1)

    def test_extra_and_missing_tokens_are_penalized(self):
        self.assertEqual(self.score("ab", "abcd")["token_accuracy"], .5)
        self.assertEqual(self.score("abcd", "ab")["token_accuracy"], .5)
        self.assertEqual(self.score("", "ab")["token_accuracy"], 0)

    def test_ast_ignores_format_but_preserves_numbers_and_side(self):
        self.assertEqual(self.score("x=1 # text", "x = 1")["ast_exact_match"], 1)
        self.assertEqual(self.score("x=2", "x=1")["ast_exact_match"], 0)
        self.assertEqual(self.score("robot.grasp(side='left')", "robot.grasp(side='right')")["ast_exact_match"], 0)
        self.assertEqual(self.score("x=1", "x = 1")["exact_match"], 0)

    def test_invalid_empty_comment_only_and_invalid_context(self):
        for pred in ("", "# hello", "def broken(", "return 1"):
            self.assertEqual(self.score(pred, "x=1")["syntax_valid_rate"], 0)
        self.assertEqual(self.score("x=1", "def bad(")["invalid_targets"], 1)

    def test_empty_target_is_excluded_and_undefined_ratios_omitted(self):
        for target in (None, "", " \n"):
            result = self.score("x=1", target)
            self.assertEqual(result["samples"], 0)
            self.assertEqual(result["unlabeled"], 1)
            self.assertNotIn("exact_match", result)

    def test_full_target_is_not_truncated(self):
        result = self.score("x=1", "x=12345", canvas=3)
        self.assertEqual(result["target_over_canvas_rate"], 1)
        self.assertEqual(result["exact_match"], 0)
        self.assertEqual(result["token_slots"], 7)

    def test_sum_counts_before_ratios(self):
        first = count_code_pair("a", "a", CharacterTokenizer(), 100)
        second = count_code_pair("", "bbbbbbbbb", CharacterTokenizer(), 100)
        first.update(second)
        self.assertEqual(summarize_code_counts(first)["token_accuracy"], .1)

    def test_generated_code_is_never_executed(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "must-not-exist"
            code = f"open({str(target)!r}, 'w').write('executed')"
            self.assertEqual(self.score(code, code)["syntax_valid_rate"], 1)
            self.assertFalse(target.exists())


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.oracle_patch = patch.dict(os.environ, {"AGENTICWAM_ORACLE_CODE_FILE": ""})
        self.oracle_patch.start()
        self.addCleanup(self.oracle_patch.stop)

    def test_observation_only_inputs_and_independent_histories(self):
        model, sample = FakeModel(), batch()
        original_video = sample["video"].clone()
        counts, examples = probe_code_batch(model, sample)
        self.assertEqual(counts["samples"], 2)
        self.assertEqual(counts["exact_matches"], 1)
        self.assertEqual(len(examples), 2)
        for i, (kwargs, history) in enumerate(model.seen):
            self.assertTrue(torch.equal(kwargs["obs_video"], original_video[i:i+1, :, [0, 3, 6]]))
            self.assertTrue(torch.equal(kwargs["proprio"], sample["proprio"][i:i+1, :1]))
            self.assertEqual(kwargs["num_video_frames"], 3)
            self.assertEqual(kwargs["action_horizon"], 4)
            self.assertEqual(kwargs["instruction"], sample["prompt_nohist"][i])
            self.assertEqual(history, sample["code_history"][i])
        self.assertTrue(torch.equal(sample["video"], original_video))

    def test_rng_modes_schedulers_and_transient_restore_on_success_and_failure(self):
        for fail in (False, True):
            model = FakeModel(fail=fail)
            flags = [m.training for m in model.modules()]
            rng = torch.get_rng_state().clone()
            if fail:
                with self.assertRaisesRegex(RuntimeError, "fake inference failure"):
                    probe_code_batch(model, batch())
            else:
                probe_code_batch(model, batch())
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual([m.training for m in model.modules()], flags)
            self.assertEqual(model.video_scheduler.sigmas.tolist(), [.25])
            self.assertFalse(hasattr(model.video_scheduler, "extra"))
            self.assertEqual(model.action_scheduler.steps, [1, 2])
            self.assertEqual(model._mem_code_history, "original history")
            self.assertEqual(model._cfg_mode, "training mode")
            self.assertFalse(hasattr(model, "_last_code_canvas"))

    def test_sample_cap_and_prompt_precedence(self):
        model = FakeModel()
        model.code_memory_enabled = False
        counts, _ = probe_code_batch(model, batch(), max_samples=1)
        self.assertEqual(counts["samples"], 1)
        self.assertEqual(model.seen[0][0]["instruction"], "code one")
        sample = batch()
        del sample["prompt_code"]
        model.seen.clear()
        probe_code_batch(model, sample, max_samples=1)
        self.assertEqual(model.seen[0][0]["instruction"], "generic one")

    def test_unlabeled_samples_do_not_generate(self):
        model, sample = FakeModel(), batch()
        sample["policy_code"] = ["", "  "]
        counts, examples = probe_code_batch(model, sample)
        self.assertEqual(counts["unlabeled"], 2)
        self.assertEqual(model.seen, [])
        self.assertEqual(examples, [])

    def test_oracle_is_rejected_even_if_file_missing(self):
        with patch.dict(os.environ, {"AGENTICWAM_ORACLE_CODE_FILE": "/not/here"}):
            with self.assertRaisesRegex(ValueError, "forbid"):
                probe_code_batch(FakeModel(), batch())


class TrainerIntegrationTests(unittest.TestCase):
    def trainer(self, output_dir, remote=None, stage=2):
        from agenticwam.trainer import AgenticWAMTrainer
        trainer = AgenticWAMTrainer.__new__(AgenticWAMTrainer)
        trainer.code_eval_max_samples = 0
        trainer.code_eval_inference_steps = 3
        trainer.global_step = 100
        trainer.output_dir = output_dir
        trainer.wandb_run = None
        def reduce(values, reduction):
            self.assertEqual(reduction, "sum")
            return values if remote is None else values + torch.tensor(remote)
        trainer.accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(
                deepspeed_config={"zero_optimization": {"stage": stage}})),
            process_index=0, device=torch.device("cpu"), is_main_process=True, reduce=reduce,
        )
        return trainer

    @patch.dict(os.environ, {"AGENTICWAM_ORACLE_CODE_FILE": ""})
    def test_global_metrics_and_rank0_artifact(self):
        remote_counts = Counter({key: 0 for key in COUNT_KEYS})
        remote_counts.update(count_code_pair("", "x = 1", CharacterTokenizer(), 20))
        remote = [0] + [remote_counts[key] for key in COUNT_KEYS]
        with TemporaryDirectory() as tmp:
            trainer = self.trainer(tmp, remote=remote)
            result = trainer.evaluate_code_batch(FakeModel(), batch())
            self.assertEqual(result["samples"], 3)
            self.assertAlmostEqual(result["exact_match"], 1/3)
            artifact = json.loads((Path(tmp) / "code_eval.jsonl").read_text())
            self.assertEqual(artifact["metrics"]["samples"], 3)
            self.assertEqual(len(artifact["examples_rank0"]), 2)

    @patch.dict(os.environ, {"AGENTICWAM_ORACLE_CODE_FILE": ""})
    def test_local_or_remote_failure_never_reports_accuracy(self):
        for remote_failure in (False, True):
            with TemporaryDirectory() as tmp:
                remote = [1] + [0] * len(COUNT_KEYS) if remote_failure else None
                trainer = self.trainer(tmp, remote=remote)
                result = trainer.evaluate_code_batch(FakeModel(fail=not remote_failure), batch())
                self.assertIsNone(result)
                self.assertFalse((Path(tmp) / "code_eval.jsonl").exists())

    def test_zero3_is_rejected_before_inference(self):
        with TemporaryDirectory() as tmp:
            model = FakeModel()
            result = self.trainer(tmp, stage=3).evaluate_code_batch(model, batch())
            self.assertIsNone(result)
            self.assertEqual(model.seen, [])


if __name__ == "__main__":
    unittest.main()
