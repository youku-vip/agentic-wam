"""Two CPU/DDP ranks run the actual trainer: batch=2, GAS=16, balanced tasks."""
import os
from tempfile import TemporaryDirectory

from accelerate import Accelerator
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from test_small_task_balanced_batches import BalancedToyDataset
from test_wait_diagnostics import make_toy_trainer


def main():
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=16)
    assert accelerator.num_processes == 2
    os.environ['AGENTICWAM_PHASE_TIMING'] = '0'
    os.environ['AGENTICWAM_WAIT_DIAG_STEPS'] = '0'
    with TemporaryDirectory(prefix='small-balanced-ddp-') as folder:
        trainer = make_toy_trainer(folder, accelerator)
        trainer.cfg = OmegaConf.create({'task_balanced_batch': True})
        trainer.seed, trainer.batch_size, trainer.gradient_accumulation_steps = 42, 2, 16
        trainer.log_every, trainer.wandb_run = 1, None
        payloads = []
        trainer._wandb_log = payloads.append
        trainer.train_dataset = BalancedToyDataset()
        trainer.train_loader = accelerator.prepare_data_loader(trainer._build_loader(trainer.train_dataset))
        trainer.train()
        assert trainer.global_step == 3
        weights = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
        gathered = accelerator.gather(weights).reshape(2, -1)
        assert torch.equal(gathered[0], gathered[1])
        if accelerator.is_main_process:
            for payload in payloads:
                assert [payload[f'sampling/task_{i}_samples'] for i in range(4)] == [16] * 4
            assert len(payloads) == 3
            print('PASS: actual trainer, 2 CPU/DDP ranks, batch=2/GAS=16, 16 samples/task/update, epoch wrap, synchronized weights')
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
