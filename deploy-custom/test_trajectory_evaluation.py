import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trajectory_evaluation',
    ROOT / 'deploy-custom/evaluate_custom34_trajectories.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
from test_checkpoint_code_eval import full_pipeline_fixture, TARGET


class StageCaseTests(unittest.TestCase):
    def fixture(self):
        class Reader:
            tasks = {0: 'open', 2: 'knob'}
            def episode_length(self, ep): return 40
            def read_episode(self, ep, columns=None):
                return pd.DataFrame({'task_index': [0 if ep == 8 else 2] * 40})
        single = SimpleNamespace(episodes=[8, 2], _starts=[0, 40], meta=SimpleNamespace(reader=Reader()))
        programs = ['', TARGET,
            "def policy_chunk(obs, state, robot, skills):\n    return robot.move_arm(side='left', target_pos=[.1,.2,.3], target_rpy=[1,2,3])\n",
            "def policy_chunk(obs, state, robot, skills):\n    return robot.move_arm(side='left', target_pos=[.4,.5,.6], target_rpy=[4,5,6])\n"]
        ids = np.tile(np.repeat([0, 1, 2, 3], [2, 10, 20, 8]), 2)
        return single, programs, ids

    def test_stage_runs_use_loader_order_and_keep_distinct_move_targets(self):
        single, codes, ids = self.fixture()
        cases, skipped = tool.select_cases(single, codes, ids, [2, 8])
        self.assertEqual([c['frame'] for c in cases], [6, 21, 35] * 2)
        self.assertEqual([c['global_index'] for c in cases], [46, 61, 75, 6, 21, 35])
        self.assertEqual([c['stage_index'] for c in cases], [1, 2, 3] * 2)
        self.assertEqual([c['valid_action_steps'] for c in cases[:3]], [32, 19, 5])
        self.assertNotEqual(cases[1]['target_sha256'], cases[2]['target_sha256'])
        self.assertEqual(len(skipped), 2)

    def test_explicit_frames_and_full_horizon_filter(self):
        single, codes, ids = self.fixture()
        cases, _ = tool.select_cases(single, codes, ids, [8], frames=[5])
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]['frame'], 5)
        cases, skipped = tool.select_cases(single, codes, ids, [8], min_valid_steps=32)
        self.assertEqual([c['frame'] for c in cases], [6])
        self.assertEqual(len(skipped), 3)
        with self.assertRaisesRegex(ValueError, 'out of range'):
            tool.select_cases(single, codes, ids, [8], frames=[41])

    def test_label_programs_are_never_executed(self):
        single, codes, ids = self.fixture()
        codes[1] = "def policy_chunk(obs, state, robot, skills):\n    return __import__('os').getcwd()"
        with self.assertRaisesRegex(ValueError, 'unknown robot API'):
            tool.select_cases(single, codes, ids, [8])

    def test_explicit_plan_rejects_duplicate_or_changed_enumeration(self):
        module = tool.evaluator()
        single, _, _ = self.fixture()
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'cases.json'
            args = SimpleNamespace(cases_file=path, padding='repeat')
            path.write_text(json.dumps({'cases': [dict(episode=2, frame=5, global_index=99)]}))
            with self.assertRaisesRegex(ValueError, 'enumeration'):
                module.evaluation_plans(single, single.meta.reader, args, 32, 1)
            path.write_text(json.dumps({'cases': [dict(episode=2, frame=5)] * 2}))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                module.evaluation_plans(single, single.meta.reader, args, 32, 1)


class TrajectoryPlotTests(unittest.TestCase):
    def test_cpu_plot_every_dimension_csv_units_and_padding_exclusion(self):
        with TemporaryDirectory() as folder:
            case = Path(folder) / 'case'; case.mkdir()
            gt = np.zeros((32, 20), dtype=np.float32)
            gt[:, [3, 7, 12, 16]] = 1
            gt[:, 18:] = 1
            prediction = gt.copy(); prediction[:, 0] = .01
            prediction[10:, 0] = 1000
            mask = np.arange(32) < 10
            np.savez_compressed(case / 'chunks.npz', pred_action20=prediction,
                                gt_action20=gt, valid_mask=mask, sample_fps=np.asarray(10.))
            (case / 'target.py').write_text(TARGET)
            (case / 'prediction.py').write_text(TARGET)
            output = Path(folder) / 'plots'
            report = tool.render_case(case, output)
            self.assertEqual(report['valid_steps'], 10)
            self.assertAlmostEqual(report['physical']['left']['position_error_mm']['rmse'], 10., places=5)
            self.assertEqual(len(list((output / 'dimensions20').glob('*.png'))), 20)
            for name in ('index.html', 'physical_errors.png', 'action20_all_dimensions.png', 'per_dimension_metrics.csv'):
                self.assertTrue((output / name).is_file())
            with (output / 'action20_comparison.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 32)
            self.assertEqual(rows[31]['time_s'], '3.1')
            self.assertEqual(rows[10]['valid'], '0')
            self.assertAlmostEqual(float(rows[0]['left_x_m_error']), .01, places=7)
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                tool.render_case(case, Path(folder) / 'bad-fps', fps=20)

    def test_old_npz_requires_explicit_fps(self):
        with TemporaryDirectory() as folder:
            case = Path(folder)
            with self.assertRaisesRegex(ValueError, 'FPS is missing'):
                tool.case_fps(case, {}, None)
            self.assertEqual(tool.case_fps(case, {}, 10), 10)

    def test_full_checkpoint_case_plan_and_render_use_real_conversion(self):
        with full_pipeline_fixture(plots=False, mixed=True, readonly=True) as (module, args):
            cases = args.out.parent / 'cases.json'
            cases.write_text(json.dumps({'cases': [dict(episode=0, frame=5,
                global_index=5, stage_index=1, stage_name='grasp:left', task_name='open',
                target_sha256=hashlib.sha256(TARGET.strip().encode()).hexdigest())]}))
            options = SimpleNamespace(train_config=args.train_config, checkpoint=args.checkpoint,
                stats=args.stats, device='cpu', cases_file=cases, out=args.out,
                inference_steps=10, canvas_tokens=256, seeds=[42], padding='repeat', relative64=False)
            with patch.object(tool, 'evaluator', return_value=module):
                self.assertEqual(tool.run(options), 0)
            summary = json.loads((args.out / 'evaluation/summary.json').read_text())
            result = summary['results'][0]
            self.assertTrue(result['code_metrics']['exact_match'])
            self.assertEqual(result['case_selection']['stage_name'], 'grasp:left')
            case = args.out / 'evaluation' / result['folder']
            with np.load(case / 'chunks.npz', allow_pickle=False) as z:
                self.assertEqual(float(z['sample_fps']), 10.)
            self.assertTrue((args.out / 'plots' / result['folder'] / 'index.html').is_file())
            self.assertTrue((args.out / 'stage_metrics.csv').is_file())

    def test_wrong_target_hash_is_rejected_before_any_prediction(self):
        with full_pipeline_fixture(plots=False) as (module, args):
            cases = args.out.parent / 'cases.json'
            cases.write_text(json.dumps({'cases': [dict(episode=0, frame=5, target_sha256='bad')]}))
            args.cases_file = cases
            with patch.object(module, 'infer_once', side_effect=AssertionError('must not run')) as infer:
                with self.assertRaisesRegex(ValueError, 'target disagree'):
                    module.run(args)
            infer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
