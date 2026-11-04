"""Real two-rank CPU/DDP train-loop smoke test; no robot data or model weights."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from accelerate import Accelerator
import torch
import torch.distributed as dist

from test_wait_diagnostics import make_toy_trainer


def main():
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    assert accelerator.num_processes == 2
    temporary = TemporaryDirectory(prefix="wait-diagnostic-ddp-") if accelerator.is_main_process else None
    paths = [temporary.name if temporary else None]
    dist.broadcast_object_list(paths, src=0)
    os.environ["AGENTICWAM_PHASE_TIMING"] = "0"
    os.environ["AGENTICWAM_WAIT_DIAG_STEPS"] = "1"
    os.environ["AGENTICWAM_WAIT_DIAG_WARMUP"] = "1"
    os.environ["AGENTICWAM_WAIT_DIAG_MODE"] = "host"
    trainer = make_toy_trainer(paths[0], accelerator)
    trainer.train()
    assert trainer.global_step == 3
    weights = torch.cat([value.detach().flatten() for value in trainer.model.parameters()])
    gathered = accelerator.gather(weights).reshape(2, -1)
    assert torch.equal(gathered[0], gathered[1])
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        summaries = [json.loads(path.read_text()) for path in (Path(paths[0]) / "wait_diagnostics").glob("*.summary.json")]
        assert sorted(item["rank"] for item in summaries) == [0, 1]
        for item in summaries:
            assert item["captured_microsteps"] == 2
            assert not item["partial"]
            assert item["metadata"]["world_size"] == 2
            assert not item["timeline"]["gpu_activity_available"]
            assert item["capture_mode"] == "host"
            assert item["trace_file"] is None
            assert "clip_and_optimizer_including_comm" in item["host_phases"]
        print("PASS: actual trainer, 2 CPU/DDP ranks, accumulation=2, per-rank traces, synchronized weights")
    accelerator.wait_for_everyone()
    if temporary:
        temporary.cleanup()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
