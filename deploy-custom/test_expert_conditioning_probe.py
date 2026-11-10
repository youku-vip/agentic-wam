import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deploy-custom'))
import evaluate_custom34_expert_conditioning as probe
from test_checkpoint_code_eval import TARGET, full_pipeline_fixture


class ExpertConditioningProbeTests(unittest.TestCase):
    def build(self, folder):
        expert = Path(folder)/'expert.py'
        expert.write_text(TARGET)
        return probe.build_probe(ROOT/'deploy-custom/evaluate_custom34_checkpoint.py', expert), expert

    def run_report(self, mode, patched=False):
        with tempfile.TemporaryDirectory() as folder:
            module, expert = self.build(folder)
            evaluator = ROOT/'deploy-custom/evaluate_custom34_checkpoint.py'
            if patched:
                from patch_checkpoint_rope_ablation import patch_source
                source = patch_source(evaluator.read_text())
                source = source.replace('code_rope_info = dict(mode=rope_mode',
                                        'code_rope_info = dict(\n        mode=rope_mode')
                evaluator = Path(folder)/'formatted_evaluator.py'
                evaluator.write_text(source)
            with full_pipeline_fixture(module=module, plots=False) as (module, args):
                import hydra.utils
                instantiate = hydra.utils.instantiate
                def checked_instantiate(cfg, **kwargs):
                    model = instantiate(cfg, **kwargs)
                    if hasattr(model, 'code_expert'):
                        model.code_expert.code_objective = 'joint'
                        from agenticwam.models.experts.wan_video_dit import precompute_freqs_cis
                        model.code_expert.freqs = precompute_freqs_cis(8, theta=1000000).real.to(torch.bfloat16)
                        model.original_rope = model.code_expert.freqs
                        inference = model.infer_action_trimodal
                        def checked_inference(**kw):
                            self.assertEqual(os.environ['AGENTICWAM_ORACLE_CODE_FILE'], str(expert))
                            return inference(**kw)
                        model.infer_action_trimodal = checked_inference
                    return model
                with patch('hydra.utils.instantiate', side_effect=checked_instantiate), patch.dict(
                        os.environ, {'AGENTICWAM_EVAL_CODE_ROPE':mode}):
                    # Mode is read when the probe is built, before any model is loaded.
                    module = probe.build_probe(evaluator, expert)
                    self.assertEqual(module.run(args), 0)
                self.assertNotIn('AGENTICWAM_ORACLE_CODE_FILE', os.environ)
                report = json.loads((args.out/'summary.json').read_text())
                self.assertEqual(report['completed'], 1)
                self.assertFalse(report['provenance']['code_generation_evaluated'])
                self.assertEqual(report['provenance']['code_conditioning'], 'provided expert code')
                rope = report['provenance']['code_rope_probe_audit']
                self.assertEqual(rope['status'], 'verified')
                self.assertEqual(rope['rebuilt'], mode == 'complex' and not patched)
                self.assertEqual(rope['after_complex'], mode == 'complex')
                self.assertEqual(rope['after_inference_complex'], mode == 'complex')
                self.assertEqual(json.loads((args.out/'run.json').read_text())['code_rope_probe_audit'], rope)
                result = report['results'][0]
                self.assertIsNone(result['code_metrics']['exact_match'])
                self.assertIsNone(result['code_metrics']['api_valid'])
                self.assertIsNone(result['code_metrics']['token_accuracy'])
                self.assertTrue(result['code_metrics']['provided_code_api_valid'])
                self.assertLess(result['physical']['left']['position_error_mm']['rmse'], .001)

    def test_original_mode_complete_cpu_report(self):
        self.run_report('original')

    def test_complex_mode_complete_cpu_report_without_evaluator_patch(self):
        self.run_report('complex')

    def test_existing_reformatted_evaluator_patch_is_compatible(self):
        self.run_report('complex', patched=True)

    def test_rejects_mismatched_target_before_inference(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            module, expert = self.build(folder)
            model = SimpleNamespace(code_expert=SimpleNamespace(code_objective='joint'))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                module.infer_once(model, {'policy_code':['different target']}, steps=10, canvas=128, seed=42)
            self.assertNotIn('AGENTICWAM_ORACLE_CODE_FILE', os.environ)

    def test_existing_complex_cache_is_preserved(self):
        from types import SimpleNamespace
        cache = torch.tensor([[1+2j]], dtype=torch.complex128)
        model = SimpleNamespace(code_expert=SimpleNamespace(freqs=cache))
        audit = probe.prepare_code_rope(model, 'complex', 1000000)
        self.assertIs(model.code_expert.freqs, cache)
        self.assertFalse(audit['rebuilt'])

    def test_theta_mismatch_rejects_without_mutation(self):
        from types import SimpleNamespace
        from agenticwam.models.experts.wan_video_dit import precompute_freqs_cis
        cache = precompute_freqs_cis(8, theta=10000).real.to(torch.bfloat16)
        model = SimpleNamespace(code_expert=SimpleNamespace(freqs=cache))
        with self.assertRaisesRegex(ValueError, 'cosine mismatch'):
            probe.prepare_code_rope(model, 'complex', 1000000)
        self.assertIs(model.code_expert.freqs, cache)


if __name__ == '__main__':
    unittest.main()
