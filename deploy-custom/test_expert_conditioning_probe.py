import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deploy-custom'))
import evaluate_custom34_expert_conditioning as probe
from test_checkpoint_code_eval import TARGET, full_pipeline_fixture


class ExpertConditioningProbeTests(unittest.TestCase):
    def build(self, folder):
        expert = Path(folder)/'expert.py'
        expert.write_text(TARGET)
        return probe.build_probe(ROOT/'deploy-custom/evaluate_custom34_checkpoint.py', expert), expert

    def test_complete_cpu_report_marks_provided_code_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            module, expert = self.build(folder)
            with full_pipeline_fixture(module=module, plots=False) as (module, args):
                import hydra.utils
                instantiate = hydra.utils.instantiate
                def checked_instantiate(cfg, **kwargs):
                    model = instantiate(cfg, **kwargs)
                    if hasattr(model, 'code_expert'):
                        model.code_expert.code_objective = 'joint'
                        inference = model.infer_action_trimodal
                        def checked_inference(**kw):
                            self.assertEqual(os.environ['AGENTICWAM_ORACLE_CODE_FILE'], str(expert))
                            return inference(**kw)
                        model.infer_action_trimodal = checked_inference
                    return model
                with patch('hydra.utils.instantiate', side_effect=checked_instantiate):
                    self.assertEqual(module.run(args), 0)
                self.assertNotIn('AGENTICWAM_ORACLE_CODE_FILE', os.environ)
                report = json.loads((args.out/'summary.json').read_text())
                self.assertEqual(report['completed'], 1)
                self.assertFalse(report['provenance']['code_generation_evaluated'])
                self.assertEqual(report['provenance']['code_conditioning'], 'provided expert code')
                result = report['results'][0]
                self.assertIsNone(result['code_metrics']['exact_match'])
                self.assertIsNone(result['code_metrics']['api_valid'])
                self.assertIsNone(result['code_metrics']['token_accuracy'])
                self.assertTrue(result['code_metrics']['provided_code_api_valid'])
                self.assertLess(result['physical']['left']['position_error_mm']['rmse'], .001)

    def test_rejects_mismatched_target_before_inference(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            module, expert = self.build(folder)
            model = SimpleNamespace(code_expert=SimpleNamespace(code_objective='joint'))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                module.infer_once(model, {'policy_code':['different target']}, steps=10, canvas=128, seed=42)
            self.assertNotIn('AGENTICWAM_ORACLE_CODE_FILE', os.environ)

    def test_complex_mode_requires_installed_rope_ablation(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {'AGENTICWAM_EVAL_CODE_ROPE':'complex'}):
                with self.assertRaisesRegex(ValueError, 'install the RoPE ablation'):
                    self.build(folder)


if __name__ == '__main__':
    unittest.main()
