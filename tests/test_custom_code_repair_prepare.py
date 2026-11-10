import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('repair_recipe',ROOT/'scripts/custom_code_repair.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
from agenticwam.models.experts.code_expert import ByteTokenizer


class Tokenizer(ByteTokenizer):
    mask_token_id=258
    def __len__(self):return 259


class RepairPreparationTests(unittest.TestCase):
    def fixture(self, folder):
        root=Path(folder)
        stats=root/'dataset_stats.json';stats.write_text('{"test_only": true}\n')
        prompt=root/'prompts.json'
        prompt.write_text(json.dumps({x:'Available robot APIs:\n' for x in ['open','pick','turn','close']}))
        cache=root/'codes.pt'
        program="def policy_chunk(obs, state, robot, skills):\n    return robot.grasp('cup', side='left')\n"
        torch.save(dict(codes=['',program],code_id_by_idx=torch.tensor([1,1,0,1])),cache)
        checkpoint=root/'step_008000.pt';checkpoint.write_bytes(b'test existence only; checkpoint is never loaded')
        source=root/'config.yaml'
        OmegaConf.save(OmegaConf.create(dict(
            model=dict(_target_='agenticwam.models.cosmos_mot.build.create_agenticwam_cosmos',
                       code_model_name='test-tokenizer',code_objective='joint',code_memory=False,
                       max_code_tokens=512,infer_code_tokens=64),
            data=dict(train=dict(skill_prompt_map=str(prompt),skill_prompt_only=False,
                                 policy_code_idx_path=str(cache),pretrained_norm_stats=str(stats)),
                      val=dict(pretrained_norm_stats=str(stats))),
            batch_size=4,gradient_accumulation_steps=8,mixed_precision='bf16',
            wandb=dict(enabled=False,name='old',group='old'))),source)
        return SimpleNamespace(source_config=source,checkpoint=checkpoint,stats=None,out=root/'pilot output',
            steps=300,lr=1e-5,canvas='auto',nproc=2,python=sys.executable,
            review_episodes=[3,1,2,3258],review_frame=5)

    def test_prepare_audits_and_generates_hydra_config_without_gpu_or_training(self):
        from hydra import compose, initialize_config_dir
        with tempfile.TemporaryDirectory(prefix='code repair ') as folder:
            args=self.fixture(folder)
            source_bytes=args.source_config.read_bytes()
            checkpoint_bytes=args.checkpoint.read_bytes()
            with patch('torch.cuda._lazy_init',side_effect=AssertionError('CUDA must not initialize')), \
                    patch.dict(os.environ,{'AGENTICWAM_CODE_MASK_FULLP':'.99','AGENTICWAM_ORACLE_CODE_FILE':'unused'}):
                report=tool.prepare(args,Tokenizer())
                self.assertEqual(os.environ['AGENTICWAM_CODE_MASK_FULLP'],'.99')
                self.assertEqual(os.environ['AGENTICWAM_ORACLE_CODE_FILE'],'unused')
            self.assertTrue(report['passed'])
            self.assertTrue(report['full_mask_probe_passed'])
            self.assertEqual(report['rope_dtype'],'torch.complex128')
            self.assertEqual(report['canvas'],128)
            self.assertEqual(report['effective_batch'],64)
            self.assertEqual(report['expected_samples_per_task_per_update'],16)
            self.assertEqual(report['unlabeled_frames'],1)
            self.assertEqual(args.source_config.read_bytes(),source_bytes)
            self.assertEqual(args.checkpoint.read_bytes(),checkpoint_bytes)
            config=args.out/'recipe/config.yaml'
            with initialize_config_dir(config_dir=str(config.parent),version_base='1.3'):
                cfg=compose(config_name='config')
            self.assertIsNone(cfg.resume)
            self.assertEqual(cfg.foundation_ckpt,str(args.checkpoint))
            self.assertEqual(cfg.model.code_rope_mode,'complex')
            self.assertEqual(cfg.model.code_training_options.canvas_tail,'masked')
            self.assertFalse(cfg.save_optimizer_state)
            self.assertEqual(Path(cfg.data.train.pretrained_norm_stats).read_bytes(),(Path(folder)/'dataset_stats.json').read_bytes())
            subprocess.run(['bash','-n',str(args.out/'run_code_repair.sh')],check=True)
            self.assertIn('export AGENTICWAM_MOT_NO_CKPT_EVERY=0',
                          (args.out/'run_code_repair.sh').read_text())
            self.assertIn('export AGENTICWAM_DISABLE_MOT_ATTN_CKPT=0',
                          (args.out/'run_code_repair.sh').read_text())
            self.assertTrue(tool.audit(cfg,Tokenizer())['passed'])
            with self.assertRaises(FileExistsError):
                tool.prepare(args,Tokenizer())

    def test_overflow_is_refused_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(folder);args.canvas='32'
            with self.assertRaisesRegex(ValueError,'exceeds'):
                tool.prepare(args,Tokenizer())
            self.assertFalse(args.out.exists())

    def test_invalid_target_api_is_not_executed_or_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(folder)
            cfg=OmegaConf.load(args.source_config)
            torch.save(dict(codes=['', "raise RuntimeError('never execute')"],
                            code_id_by_idx=torch.tensor([1])),cfg.data.train.policy_code_idx_path)
            with self.assertRaisesRegex(ValueError,'invalid target robot APIs'):
                tool.prepare(args,Tokenizer())
            self.assertFalse(args.out.exists())

    def test_explicit_missing_statistics_do_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(folder);args.stats=Path(folder)/'missing.json'
            with self.assertRaises(FileNotFoundError):
                tool.prepare(args,Tokenizer())
            self.assertFalse(args.out.exists())

    def test_prepare_accepts_global_task_balance_with_two_samples_per_rank(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.fixture(folder)
            cfg=OmegaConf.load(args.source_config)
            cfg.batch_size=2
            cfg.gradient_accumulation_steps=16
            OmegaConf.save(cfg,args.source_config)
            report=tool.prepare(args,Tokenizer())
            self.assertEqual(report['effective_batch'],64)
            self.assertEqual(report['expected_samples_per_task_per_update'],16)


if __name__=='__main__':
    unittest.main()
