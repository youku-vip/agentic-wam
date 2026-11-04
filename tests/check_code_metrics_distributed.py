"""Two CPU ranks with real Accelerate reduction; run using torchrun, no model weights.

PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=2 \
    tests/check_code_metrics_distributed.py
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from accelerate import Accelerator
import torch.distributed as dist

from agenticwam.trainer import AgenticWAMTrainer
from test_code_metrics import FakeModel, batch


def main():
    os.environ.pop("AGENTICWAM_ORACLE_CODE_FILE", None)
    accelerator = Accelerator(cpu=True)
    assert accelerator.num_processes == 2
    with TemporaryDirectory() as tmp:
        trainer = AgenticWAMTrainer.__new__(AgenticWAMTrainer)
        trainer.accelerator = accelerator
        trainer.code_eval_max_samples = 0
        trainer.code_eval_inference_steps = 3
        trainer.global_step = 100
        trainer.output_dir = tmp
        trainer.wandb_run = None
        sample = batch()
        if accelerator.process_index == 1:
            sample["policy_code"] = ["x = 1", ""]
        result = trainer.evaluate_code_batch(FakeModel(), sample)
        assert result["samples"] == 3, result
        assert result["unlabeled"] == 1, result
        assert abs(result["exact_match"] - 2/3) < 1e-8, result
        assert abs(result["token_accuracy"] - 14/15) < 1e-8, result
        if accelerator.is_main_process:
            assert (Path(tmp) / "code_eval.jsonl").exists()
        # Failure on ONE rank: both ranks must exit collectively without scoring.
        result = trainer.evaluate_code_batch(
            FakeModel(fail=accelerator.process_index == 1), sample,
        )
        assert result is None
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("PASS: 2 CPU ranks, weighted counts, unlabeled exclusion, one-rank failure")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
