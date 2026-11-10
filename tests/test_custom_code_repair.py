import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
import warnings
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from agenticwam.models.experts.code_expert import ByteTokenizer, CodeExpert


ENV = dict(AGENTICWAM_CODE_ROPE_MODE='complex', AGENTICWAM_CODE_CANVAS='64',
           AGENTICWAM_CODE_CANVAS_TAIL='masked', AGENTICWAM_PAD_EOS_WINDOW='4',
           AGENTICWAM_PAD_EOS_WEIGHT='1', AGENTICWAM_CODE_MASK_TMIN='0.5',
           AGENTICWAM_CODE_MASK_FULLP='1', AGENTICWAM_CODE_NO_INSTRUCTION='0')


def tiny_expert(**options):
    return CodeExpert(blocks=nn.ModuleList([]), embed_tokens=nn.Embedding(259, 8),
                      final_norm=nn.Identity(), lm_head=nn.Linear(8, 259, bias=False),
                      hidden_dim=8, num_heads=1, attn_head_dim=8, vocab_size=259,
                      tokenizer=ByteTokenizer(), code_objective='joint', mask_token_id=258,
                      rope_theta=1000000, **options)


class CodeRepairTests(unittest.TestCase):
    def test_saved_config_options_work_without_environment_flags(self):
        with patch.dict(os.environ, ENV):
            for key in ENV:
                os.environ.pop(key, None)
            expert=tiny_expert(code_rope_mode='complex', code_canvas=64,
                code_training_options=dict(canvas_tail='masked', mask_tmin=.5,
                                           mask_fullp=1., pad_eos_window=4))
            expert.to(dtype=torch.bfloat16)
            self.assertTrue(expert.freqs.is_complex())
            ids,labels=expert.build_code_batch(['turn'],['return 1'],'cpu')
            plen=len(expert.tokenizer.encode('# Instruction: turn\n'))
            self.assertTrue((ids[0,plen:]==258).all())
            self.assertEqual(int((labels==257).sum()),5)
            self.assertIn('acc_noneos',expert.code_loss(torch.zeros(1,ids.shape[1],259),labels))

    def test_sdar_factory_passes_persisted_configuration_before_cast(self):
        from types import SimpleNamespace
        from agenticwam.models.experts import code_expert as source
        class Tokenizer(ByteTokenizer):
            mask_token_id=258
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args,**kwargs): return Tokenizer()
        attention=SimpleNamespace(**{name:nn.Linear(8,8) for name in ('q_proj','k_proj','v_proj','o_proj')})
        layer=SimpleNamespace(input_layernorm=nn.Identity(),post_attention_layernorm=nn.Identity(),
                              mlp=nn.Linear(8,8),self_attn=attention)
        model=SimpleNamespace(model=SimpleNamespace(layers=[layer],embed_tokens=nn.Embedding(259,8),norm=nn.Identity()),
                              lm_head=nn.Linear(8,259,bias=False))
        with patch.dict(os.environ, ENV):
            for key in ENV: os.environ.pop(key,None)
            with patch.dict(sys.modules,{'transformers':SimpleNamespace(AutoTokenizer=AutoTokenizer)}), \
                    patch.object(source,'_load_sdar_backbone',return_value=model):
                expert=source.build_code_expert(model_name='tiny-sdar',num_layers=1,num_heads=1,
                    attn_head_dim=8,n_extra_layers=0,objective='joint',device='cpu',
                    freeze_backbone=False,code_canvas=64,code_rope_mode='complex',
                    code_training_options=dict(canvas_tail='masked',mask_fullp=1.))
            self.assertEqual(expert.code_rope_mode,'complex')
            self.assertEqual(expert.code_training_options['canvas_tail'],'masked')
            self.assertTrue(expert.freqs.is_complex())
            self.assertTrue(all(p.dtype==torch.bfloat16 for p in expert.parameters()))
    def test_parent_cast_preserves_complex_cache_and_checkpoint_layout(self):
        with patch.dict(os.environ, ENV):
            expert=tiny_expert(); cache=expert.freqs.clone(); keys=set(expert.state_dict())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                parent=nn.Sequential(expert).to(dtype=torch.bfloat16)
                parent.float().bfloat16()
            self.assertTrue(expert.freqs.is_complex())
            self.assertTrue(torch.equal(expert.freqs, cache))
            self.assertEqual(keys, set(expert.state_dict()))
            self.assertNotIn('freqs', expert.state_dict())
            self.assertTrue(all(p.dtype==torch.bfloat16 for p in expert.parameters()))
            self.assertFalse(any('Casting complex' in str(w.message) for w in caught))

    def test_legacy_mode_retains_old_cast_behavior(self):
        with patch.dict(os.environ, {**ENV, 'AGENTICWAM_CODE_ROPE_MODE':'legacy'}):
            expert=tiny_expert()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');expert.to(dtype=torch.bfloat16)
            self.assertFalse(expert.freqs.is_complex())

    def test_forward_refuses_lost_imaginary_component(self):
        with patch.dict(os.environ, ENV):
            expert=tiny_expert(); expert.freqs=expert.freqs.real
            with self.assertRaisesRegex(RuntimeError, 'lost its imaginary'):
                expert.pre_dit(torch.tensor([[1, 2]]))

    def test_full_mask_matches_inference_canvas_and_bounds_eos_labels(self):
        with patch.dict(os.environ, ENV):
            expert=tiny_expert(); instruction='turn'; code='return 1'
            ids,labels=expert.build_code_batch([instruction],[code],'cpu')
            plen=len(expert.tokenizer.encode('# Instruction: turn\n'))
            self.assertEqual(ids.shape, (1, plen+64))
            self.assertTrue((ids[0,plen:]==258).all())
            self.assertTrue((labels[0,:plen]==-100).all())
            self.assertEqual(int((labels==257).sum()), 5)
            self.assertEqual(int((labels!=-100).sum()), len(code.encode())+5)
            self.assertTrue((labels[0,plen+len(code.encode())+5:]==-100).all())

    def test_empty_target_keeps_fixed_canvas_and_zero_finite_loss(self):
        with patch.dict(os.environ, ENV):
            expert=tiny_expert(); ids,labels=expert.build_code_batch(['x'],[''],'cpu')
            plen=len(expert.tokenizer.encode('# Instruction: x\n'))
            self.assertEqual(ids.shape[1],plen+64)
            self.assertTrue((ids[0,plen:]==258).all())
            self.assertTrue((labels==-100).all())
            logits=torch.zeros(1,ids.shape[1],259,requires_grad=True)
            loss=expert.code_loss(logits,labels)['loss']
            self.assertEqual(float(loss),0);loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_target_overflow_is_refused(self):
        with patch.dict(os.environ, ENV):
            with self.assertRaisesRegex(ValueError,'refusing truncation'):
                tiny_expert().build_code_batch(['x'],['x'*64],'cpu')

    def test_prompt_overflow_is_refused(self):
        with patch.dict(os.environ, ENV):
            with self.assertRaisesRegex(ValueError,'exceeds training cap'):
                tiny_expert().build_code_batch(['x'],['y'],'cpu',max_length=70)

    def test_legacy_batches_and_rng_match_prechange_golden(self):
        golden=json.loads((ROOT/'tests/fixtures/custom_code_legacy_batch.json').read_text())
        for row in golden['cases']:
            env={**ENV,'AGENTICWAM_CODE_ROPE_MODE':'legacy','AGENTICWAM_CODE_CANVAS_TAIL':'legacy',
                 'AGENTICWAM_CODE_CANVAS':str(row['canvas']),'AGENTICWAM_PAD_EOS_WINDOW':'0',
                 'AGENTICWAM_CODE_MASK_TMIN':'0','AGENTICWAM_CODE_MASK_FULLP':'0'}
            with self.subTest(canvas=row['canvas']),patch.dict(os.environ,env):
                expert=tiny_expert();torch.manual_seed(123)
                ids,labels=expert.build_code_batch(row['instructions'],row['codes'],'cpu')
                self.assertEqual(ids.tolist(),row['input_ids'])
                self.assertEqual(labels.tolist(),row['labels'])
                self.assertEqual(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),row['rng_sha256'])

    def test_invalid_mask_probabilities_fail(self):
        with patch.dict(os.environ,{**ENV,'AGENTICWAM_CODE_MASK_FULLP':'1.1'}):
            with self.assertRaisesRegex(ValueError,'FULLP'):
                tiny_expert().build_code_batch(['x'],['y'],'cpu')


if __name__=='__main__':
    unittest.main()
