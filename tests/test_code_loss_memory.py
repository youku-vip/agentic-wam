"""Check CE values/gradients and retained tensor sizes without a GPU/model download."""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from agenticwam.models.experts.code_expert import ByteTokenizer, CodeExpert


def expert(objective='joint', weight=1.):
    return CodeExpert(blocks=nn.ModuleList([]), embed_tokens=nn.Embedding(259, 8),
                      final_norm=nn.Identity(), lm_head=nn.Linear(8, 259, bias=False),
                      hidden_dim=8, num_heads=1, attn_head_dim=8, vocab_size=259,
                      tokenizer=ByteTokenizer(), code_objective=objective,
                      code_training_options={'pad_eos_weight': weight, 'pad_eos_window': 8})


class CodeLossMemoryTests(unittest.TestCase):
    def test_token_values_and_weighted_gradients_match_dense_fp32_ce(self):
        torch.manual_seed(182)
        labels = torch.randint(0, 43, (3, 29))
        labels[:, :9] = -100
        labels[1, :] = -100
        weights = torch.rand(labels.shape)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.randn(3, 29, 43).to(dtype).requires_grad_()
                y = x.detach().clone().requires_grad_()
                got = CodeExpert.token_cross_entropy(x, labels, chunk_tokens=7)
                ref = F.cross_entropy(y.float().reshape(-1, 43), labels.reshape(-1),
                                      ignore_index=-100, reduction='none').view_as(labels)
                torch.testing.assert_close(got, ref)
                (got * weights).sum().backward()
                (ref * weights).sum().backward()
                torch.testing.assert_close(x.grad, y.grad)

    def test_code_loss_preserves_ar_shift_eos_weight_and_metrics(self):
        torch.manual_seed(91)
        labels = torch.randint(0, 259, (3, 11))
        labels[:, :4] = -100
        labels[:, -2:] = 257
        labels[1] = -100
        with patch.dict(os.environ, {'AGENTICWAM_PAD_EOS_WEIGHT': '0.3'}):
            for objective in ('ar', 'joint', 'blockdiff'):
                with self.subTest(objective=objective):
                    ce = expert(objective)
                    x = torch.randn(3, 11, 259, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    a, b = (y[:, :-1], labels[:, 1:]) if objective == 'ar' else (y, labels)
                    a, b = a.reshape(-1, 259), b.reshape(-1)
                    raw = F.cross_entropy(a.float(), b, ignore_index=-100, reduction='none')
                    weights = torch.ones_like(raw)
                    weights[b == 257] = .3
                    weights[b == -100] = 0
                    ref = (raw * weights).sum() / weights.sum().clamp(min=1e-6)
                    got = ce.code_loss(x, labels)
                    torch.testing.assert_close(got['loss'], ref)
                    for name, valid in [('acc', b != -100), ('acc_noneos', (b != -100) & (b != 257))]:
                        metric = (a.argmax(-1)[valid] == b[valid]).float().mean()
                        torch.testing.assert_close(got[name], metric)
                    got['loss'].backward()
                    ref.backward()
                    torch.testing.assert_close(x.grad, y.grad)

    def test_empty_labels_keep_model_parameters_in_graph_with_zero_gradients(self):
        ce = expert()
        tokens = ce.embed_tokens(torch.tensor([[1, 2, 3], [4, 5, 6]]))
        logits = ce.post_dit(tokens)
        labels = torch.full((2, 3), -100, dtype=torch.long)
        loss = ce.code_loss(logits, labels)['loss']
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        for module in (ce.embed_tokens, ce.lm_head):
            for p in module.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertEqual(int(torch.count_nonzero(p.grad)), 0)

    def test_fp32_vocabulary_buffers_are_bounded_and_not_retained(self):
        labels = torch.full((2, 700), -100, dtype=torch.long)
        labels[0, -70:] = torch.arange(70)
        logits = torch.randn(2, 700, 259, dtype=torch.bfloat16, requires_grad=True)
        original = F.cross_entropy
        calls, saved = [], []

        def checked(source, target, **kwargs):
            calls.append(tuple(source.shape))
            self.assertEqual(source.dtype, torch.float32)
            self.assertLessEqual(source.shape[0], 32)
            return original(source, target, **kwargs)

        def pack(tensor):
            saved.append((tensor.dtype, tuple(tensor.shape)))
            return tensor

        with patch.object(F, 'cross_entropy', side_effect=checked), \
                torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = CodeExpert.token_cross_entropy(logits, labels)
            self.assertEqual(calls, [(32, 259), (32, 259), (6, 259)])
            self.assertFalse(any(dt == torch.float32 and len(shape) == 2 and shape[-1] == 259
                                 for dt, shape in saved))
            result.sum().backward()
        self.assertGreater(len(calls), 3)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_no_grad_and_noncontiguous_inputs_match_dense(self):
        x = torch.randn(2, 8, 17).transpose(0, 1)
        labels = torch.randint(0, 17, (8, 2))
        labels[3] = -100
        with torch.no_grad():
            got = CodeExpert.token_cross_entropy(x, labels, chunk_tokens=3)
            ref = F.cross_entropy(x.reshape(-1, 17), labels.reshape(-1), ignore_index=-100,
                                  reduction='none').view_as(labels)
        torch.testing.assert_close(got, ref)

    def test_actual_model_code_branch_keeps_sigma_and_rl_weighting(self):
        # Compile the actual code-loss block in isolation; avoid building the 5B model.
        path = ROOT / 'src/agenticwam/models/cosmos_mot/agenticwam_model.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_training_loss_impl')
        block = next(n for n in fn.body if isinstance(n, ast.If)
                     and ast.unparse(n.test).startswith('code_pre is not None and code_labels'))
        wrapper = ast.parse('def run(self, code_labels, out, _rw_on, w_cond):\n'
                            '    code_pre = {}\n    loss = 0.\n    loss_dict = {}\n'
                            '    return loss, loss_dict\n').body[0]
        wrapper.body.insert(-1, block)
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        namespace = {'torch': torch, 'F': F, 'diagnostic_range': lambda _: nullcontext()}
        exec(compile(module, str(path), 'exec'), namespace)
        torch.manual_seed(63)
        labels = torch.randint(0, 259, (3, 9))
        labels[:, :2] = -100
        labels[1] = -100
        factors = torch.tensor([.4, 1.3, 2.])
        for mode in ('sigma', 'rl', 'ordinary'):
            with self.subTest(mode=mode):
                ce = expert()
                x = torch.randn(3, 9, 259, dtype=torch.bfloat16, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                ce.post_dit = lambda tokens: tokens
                model = SimpleNamespace(code_expert=ce, loss_lambda_code=.5,
                                        _rl_weights=factors if mode == 'rl' else None)
                got, _ = namespace['run'](model, labels, {'code': x}, mode == 'sigma', factors)
                raw = F.cross_entropy(y.float().reshape(-1, 259), labels.reshape(-1),
                                      ignore_index=-100, reduction='none').view_as(labels)
                counts = (labels != -100).sum(1).clamp(min=1)
                if mode == 'sigma':
                    ref = (raw.sum(1) / counts * factors).mean() * .5
                elif mode == 'rl':
                    ref = (raw.sum(1) / counts * factors).sum() * .5
                else:
                    ref = raw.sum() / (labels != -100).sum() * .5
                torch.testing.assert_close(got, ref)
                got.backward()
                ref.backward()
                torch.testing.assert_close(x.grad, y.grad)


if __name__ == '__main__':
    unittest.main()
