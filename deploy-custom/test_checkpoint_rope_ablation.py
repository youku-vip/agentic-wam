import importlib.util
from pathlib import Path
import sys
import unittest
import warnings
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('rope_patch', ROOT/'deploy-custom/patch_checkpoint_rope_ablation.py')
patcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(patcher)


class RopeAblationTests(unittest.TestCase):
    def model(self, theta=1000000.):
        from agenticwam.models.experts.code_expert import CodeExpert
        model = nn.Module()
        model.code_expert = CodeExpert(blocks=nn.ModuleList([]), embed_tokens=nn.Embedding(32,8),
                                      final_norm=nn.Identity(), lm_head=nn.Linear(8,32,bias=False),
                                      hidden_dim=8, num_heads=1, attn_head_dim=8, vocab_size=32,
                                      max_code_tokens=32, rope_theta=theta)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.code_expert.to(dtype=torch.bfloat16)
        return model

    def apply_block(self, model, mode):
        import json, os
        namespace = dict(model=model, torch=torch, json=json, os=os)
        import textwrap
        with patch.dict(os.environ, {'AGENTICWAM_EVAL_CODE_ROPE':mode,
                                     'AGENTICWAM_EVAL_CODE_ROPE_THETA':'1000000'}):
            exec(compile(textwrap.dedent(patcher.BLOCK), '<rope block>', 'exec'), namespace)
        return namespace['code_rope_info']

    def test_original_is_bitwise_unchanged(self):
        model = self.model(); before = model.code_expert.freqs
        info = self.apply_block(model, 'original')
        self.assertIs(model.code_expert.freqs, before)
        self.assertFalse(info['after_complex'])
        self.assertTrue(info['quantized_cosine_matches'])
        self.assertFalse(info['rebuilt'])

    def test_restores_rotation_without_changing_checkpoint_tensors(self):
        from agenticwam.models.experts.wan_video_dit import rope_apply
        model = self.model()
        weights = {k:v.clone() for k,v in model.state_dict().items()}
        info = self.apply_block(model, 'complex')
        self.assertTrue(info['rebuilt'])
        self.assertTrue(info['after_complex'])
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v, weights[k]), k)
        x = torch.ones(1,16,8)
        y = rope_apply(x, model.code_expert.freqs[:16].view(16,1,-1), 1)
        torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1))

    def test_mismatched_theta_is_rejected(self):
        model = self.model(theta=10000.)
        before = model.code_expert.freqs
        with self.assertRaisesRegex(ValueError, 'cosine mismatch'):
            self.apply_block(model, 'complex')
        self.assertIs(model.code_expert.freqs, before)

    def test_patch_is_idempotent_and_valid(self):
        source = (ROOT/'deploy-custom/evaluate_custom34_checkpoint.py').read_text()
        result = patcher.patch_source(source)
        self.assertIn('code_rope_audit=code_rope_info', result)
        self.assertEqual(patcher.patch_source(result), result)


if __name__ == '__main__':
    unittest.main()
