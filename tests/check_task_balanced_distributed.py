"""Two real CPU/DDP ranks run the actual trainer with balanced batches and GAS=8."""
import os
from tempfile import TemporaryDirectory

from accelerate import Accelerator
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from test_task_balanced_sampling import BalancedToyDataset
from test_wait_diagnostics import make_toy_trainer


def main():
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=8)
    assert accelerator.num_processes == 2
    os.environ["AGENTICWAM_PHASE_TIMING"] = "0"
    os.environ["AGENTICWAM_WAIT_DIAG_STEPS"] = "0"
    with TemporaryDirectory(prefix="balanced-ddp-") as temporary:
        trainer = make_toy_trainer(temporary, accelerator)
        trainer.cfg = OmegaConf.create({"task_balanced_batch": True})
        trainer.seed, trainer.batch_size, trainer.gradient_accumulation_steps = 42, 4, 8
        trainer.log_every, trainer.wandb_run = 1, None
        payloads = []
        trainer._wandb_log = payloads.append
        trainer.train_dataset = BalancedToyDataset()
        trainer.train_loader = accelerator.prepare_data_loader(trainer._build_loader(trainer.train_dataset))
        trainer.train()
        assert trainer.global_step == 3
        weights = torch.cat([value.detach().flatten() for value in trainer.model.parameters()])
        all_weights = accelerator.gather(weights).reshape(2, -1)
        assert torch.equal(all_weights[0], all_weights[1])
        if accelerator.is_main_process:
            for payload in payloads:
                assert [payload[f"sampling/task_{i}_samples"] for i in range(4)] == [16]*4
            assert len(payloads) == 3
            print("PASS: 2 CPU/DDP ranks, actual trainer, GAS=8, epoch wrap, each update has 16 samples/task, synchronized weights")
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
