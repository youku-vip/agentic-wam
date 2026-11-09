import importlib.util
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('checkpoint_eval', ROOT/'deploy-custom/evaluate_custom34_checkpoint.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

TARGET = "def policy_chunk(obs, state, robot, skills):\n    return robot.grasp('air fryer door handle', side='left')\n"


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class Model:
    device = 'cpu'
    n_cameras = 3

    def __init__(self, fail=False):
        self.fail = fail
        self.video_scheduler = SimpleNamespace(sigmas=torch.tensor([1., 0.]))
        self.action_scheduler = SimpleNamespace(sigmas=torch.tensor([1., 0.]))

    def infer_action_trimodal(self, **kw):
        assert kw['return_video'] is False
        assert kw['proprio'].shape == (1, 1, 64)
        assert kw['num_video_frames'] == 9
        assert os.environ['AGENTICWAM_INFER_CODE_TOKENS'] == str(kw['max_code_tokens'])
        self.video_scheduler.sigmas.zero_()
        self.action_scheduler.new_field = 'changed'
        if self.fail:
            raise RuntimeError('test failure')
        return torch.randn(32, 64), TARGET


def sample():
    return dict(video=torch.arange(27.).reshape(1, 1, 27, 1, 1).expand(1, 3, 27, 2, 2),
                proprio=torch.zeros(1, 32, 64), action=torch.zeros(1, 32, 64),
                prompt_embs=torch.zeros(1, 2, 4), prompt=['wrapped task'], prompt_code=['skill plus task'])


class CheckpointEvalTests(unittest.TestCase):
    def test_reference_lists_become_arrays_before_metrics(self):
        from agenticwam.utils.custom34_review import compare_actions
        action = [[.1, .2, .3, 1, 0, 0, 0, 1, 0, -.1, -.2, -.3, 1, 0, 0, 0, 1, 0, 0, 1]]*32
        raw = dict(action20=action, valid_mask=[True]*32, fps=10, source={})
        with self.assertRaises(TypeError):
            compare_actions(np.asarray(action), raw)
        reference = tool.action_reference(action, [True]*32, 10, {})
        metrics = compare_actions(np.asarray(action), reference)
        self.assertEqual(metrics['valid_steps'], 32)
        self.assertEqual(metrics['physical']['left']['position_error_mm']['rmse'], 0)

    def test_mixed_builder_dtypes_are_unified_without_corrupting_rope(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2).bfloat16())
        rope = torch.tensor([1+2j, 3-4j], dtype=torch.complex64)
        model.register_buffer('rope', rope.clone())
        model.register_buffer('indices', torch.tensor([1, 2], dtype=torch.int64))
        with self.assertRaisesRegex(RuntimeError, 'dtype'):
            model(torch.ones(1, 4))
        with warnings.catch_warnings(record=True) as captured:
            profile = tool.prepare_inference_model(model, 'cpu', torch.bfloat16)
        self.assertFalse(any('complex' in str(w.message) for w in captured))
        self.assertEqual(set(profile['after']['parameters']), {'torch.bfloat16'})
        self.assertEqual(model.rope.dtype, torch.complex64)
        self.assertTrue(torch.equal(model.rope, rope))
        self.assertEqual(model.indices.dtype, torch.int64)
        self.assertEqual(model(torch.ones(1, 4, dtype=torch.bfloat16)).dtype, torch.bfloat16)

    def test_fp32_mode_also_unifies_builder_dtypes(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4).bfloat16(), torch.nn.Linear(4, 2))
        profile = tool.prepare_inference_model(model, 'cpu', torch.float32)
        self.assertEqual(set(profile['after']['parameters']), {'torch.float32'})
        self.assertEqual(model(torch.ones(1, 4)).dtype, torch.float32)

    def test_first_frame_of_each_camera(self):
        obs, frames = tool.observation_frames(sample()['video'], 3)
        self.assertEqual(frames, 9)
        self.assertEqual(obs[0, 0, :, 0, 0].tolist(), [0, 9, 18])

    def test_exact_target(self):
        result = tool.code_metrics(TARGET, TARGET, Tokenizer(), 256)
        self.assertTrue(result['exact_match'])
        self.assertTrue(result['api_valid'])
        self.assertTrue(result['primitive_sequence_match'])

    def test_syntax_valid_is_not_api_valid(self):
        prediction = "def policy_chunk(obs, state, robot, skills):\n    robot.move_arm(side='left', target_pos=[.2,.1,-.3], target_rpy=[179.56,-82.64])"
        result = tool.code_metrics(prediction, TARGET, Tokenizer(), 256)
        self.assertTrue(result['syntax_valid'])
        self.assertFalse(result['api_valid'])
        self.assertFalse(result['primitive_sequence_match'])
        self.assertIn('target_rpy', result['api_error'])

    def test_wrong_arm_is_not_same_stage(self):
        result = tool.code_metrics(TARGET.replace("side='left'", "side='right'"), TARGET, Tokenizer(), 256)
        self.assertTrue(result['api_valid'])
        self.assertFalse(result['primitive_sequence_match'])

    def test_empty_target_is_not_scored_as_zero_accuracy(self):
        result = tool.code_metrics(TARGET, '', Tokenizer(), 256)
        self.assertFalse(result['labeled'])
        self.assertIsNone(result['exact_match'])
        self.assertIsNone(result['primitive_sequence_match'])

    def test_generated_calls_are_not_executed(self):
        prediction = "def policy_chunk(obs, state, robot, skills):\n    return __import__('os').getcwd()"
        result = tool.code_metrics(prediction, TARGET, Tokenizer(), 256)
        self.assertFalse(result['api_valid'])

    def test_seed_and_state_restoration(self):
        model = Model()
        original_rng = torch.get_rng_state().clone()
        with patch.dict(os.environ, {'AGENTICWAM_INFER_CODE_TOKENS': '999'}):
            first = tool.infer_once(model, sample(), steps=10, canvas=64, seed=42)[0]
            second = tool.infer_once(model, sample(), steps=20, canvas=128, seed=42)[0]
            torch.testing.assert_close(first, second)
            self.assertEqual(os.environ['AGENTICWAM_INFER_CODE_TOKENS'], '999')
        torch.testing.assert_close(model.video_scheduler.sigmas, torch.tensor([1., 0.]))
        self.assertFalse(hasattr(model.action_scheduler, 'new_field'))
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))

    def test_failure_restores_scheduler_and_env(self):
        model = Model(fail=True)
        with patch.dict(os.environ, {'AGENTICWAM_INFER_CODE_TOKENS': '999'}):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                tool.infer_once(model, sample(), steps=10, canvas=64, seed=42)
            self.assertEqual(os.environ['AGENTICWAM_INFER_CODE_TOKENS'], '999')
        torch.testing.assert_close(model.video_scheduler.sigmas, torch.tensor([1., 0.]))
        self.assertFalse(hasattr(model.action_scheduler, 'new_field'))

    def test_silent_loader_fallback_is_rejected(self):
        class Dataset:
            def __getitem__(self, index):
                return {'idx': index+1}
        recorder = tool.IndexRecorder(Dataset())
        with self.assertRaisesRegex(RuntimeError, 'refusing to score'):
            tool.checked_sample(recorder, recorder, 207)

    def test_absolute_conversion_uses_base_and_one_inverse_normalization(self):
        from agenticwam.datasets.lerobot.transforms.custom_unified_eef import CustomUnifiedEEFTransform
        transform = CustomUnifiedEEFTransform(['default'])
        left = [1, 2, 3, 0, 1, 0, -1, 0, 0]
        right = [-1, -2, -3, 1, 0, 0, 0, 1, 0]
        state = torch.tensor([0.]*14+left+right+[1., 0.])
        gt = torch.tensor([left+right+[0., 1.]]).repeat(32, 1)
        gt[:, 1] += .1
        relative = transform._action64(gt[None], state[None])[0]
        class Normalizer:
            def backward(self, x):
                return (x+1)/2
        actual_relative, absolute = tool.absolute_prediction(relative*2-1, state, transform, Normalizer(), 'default')
        torch.testing.assert_close(actual_relative, relative, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(absolute, gt.numpy(), atol=2e-7)


@contextmanager
def full_pipeline_fixture(module=tool, *, start=5, plots=True):
    """Real conversion, normalization, metrics, PNGs and plots; model/data source are CPU fakes."""
    import pandas as pd
    from omegaconf import OmegaConf
    from agenticwam.datasets.lerobot.transforms.custom_unified_eef import CustomUnifiedEEFTransform
    from agenticwam.datasets.lerobot.utils.normalizer import LinearNormalizer
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        transform = CustomUnifiedEEFTransform(['default'])
        shape = {'images': [{'key': k} for k in ['head', 'left', 'right']],
                 'state': [{'key': 'default', 'shape': 64}], 'action': [{'key': 'default', 'shape': 64}]}
        stats = {}
        for field in ['state', 'action']:
            stats[field] = {'default': {}}
            for prefix, dims in [('global', (64,)), ('stepwise', (32, 64) if field=='action' else (1, 64))]:
                for key, value in [('min', -2), ('max', 2), ('q01', -2), ('q99', 2), ('mean', 0), ('std', 1)]:
                    stats[field]['default'][f'{prefix}_{key}'] = torch.full(dims, float(value))
        norm = LinearNormalizer(shape_meta=shape, use_stepwise_action_norm=True, default_mode='q01/q99',
                                exception_mode=None, stats=stats)
        processor = SimpleNamespace(shape_meta=shape, normalizer=norm, action_state_transforms=[transform])
        rows = []
        for i in range(40):
            left = [.18+i*.001, .15, -.39, 1, 0, 0, 0, 1, 0]
            right = [.14, -.22, -.44, 1, 0, 0, 0, 1, 0]
            rows.append({'observation.state': np.array([0.]*14+left+right+[1.,1.], dtype=np.float32),
                         'action': np.array(left+right+[float(i<23),1.], dtype=np.float32), 'task_index': 0})
        frame = pd.DataFrame(rows)
        class Reader:
            fps = 10
            def episode_length(self, episode): return len(frame)
            def read_episode(self, episode, columns=None): return frame.copy()
        reader = Reader(); reader.root = root
        single = SimpleNamespace(episodes=[0], _starts=[0], meta=SimpleNamespace(reader=reader))
        class Base:
            def __len__(self): return len(frame)
            def __getitem__(self, index): return {'idx': index}
        base = Base(); base.processor = processor; base.multi_dataset = SimpleNamespace(_datasets=[single])
        class Dataset:
            def __init__(self): self.lerobot_dataset = base
            def __getitem__(self, index):
                actual = self.lerobot_dataset[index]['idx']
                idx = actual+np.arange(32); clipped = np.minimum(idx, len(frame)-1)
                state = torch.from_numpy(frame.iloc[actual]['observation.state'])
                action = torch.from_numpy(np.stack(frame.iloc[clipped]['action']))
                relative = transform._action64(action[None], state[None])[0]
                states = torch.from_numpy(np.stack(frame.iloc[clipped]['observation.state']))
                return dict(video=torch.zeros(3,27,32,32),
                            proprio=norm.normalizers['state']['default'].forward(transform._state64(states)),
                            action=norm.normalizers['action']['default'].forward(relative),
                            action_is_pad=torch.from_numpy(idx>=len(frame)), prompt='task', prompt_code='skill plus task',
                            prompt_embs=torch.zeros(2,4), policy_code=TARGET)
        dataset = Dataset()
        target_action = dataset[start]['action']
        class CheckpointModel(torch.nn.Module):
            n_cameras = 3
            def __init__(self):
                super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.video_scheduler = SimpleNamespace(sigmas=torch.tensor([1.,0.]))
                self.action_scheduler = SimpleNamespace(sigmas=torch.tensor([1.,0.]))
                self.code_expert = SimpleNamespace(tokenizer=Tokenizer(), max_code_tokens=512)
            @property
            def device(self): return self.anchor.device
            def _trainable_state_dict(self): return self.state_dict()
            def _resolve_infer_code_tokens(self): return 256
            def infer_action_trimodal(self, **kw):
                assert kw['return_video'] is False
                return target_action.clone(), TARGET
        model = CheckpointModel()
        checkpoint = root/'weights.pt'
        torch.save({'model':model.state_dict(), 'step':8000}, checkpoint)
        stats_path = root/'dataset_stats.json'
        stats_path.write_text(json.dumps({f:{'default':{k:v.tolist() for k,v in d['default'].items()}} for f,d in stats.items()}))
        config_path = root/'config.yaml'
        config = OmegaConf.create(dict(mixed_precision='bf16', model={'_target_':'fixture.model','code_memory':False},
                                      data={'train':{'_target_':'fixture.dataset','pretrained_norm_stats':str(stats_path),
                                                     'num_frames':33, 'global_sample_stride':1}}))
        OmegaConf.save(config, config_path)
        args = SimpleNamespace(train_config=config_path, checkpoint=checkpoint, stats=stats_path,
                               out=root/'output', episodes=[0], start_frames=[start], device='cpu',
                               padding='repeat' if start>8 else 'error', plots=plots,
                               canvas_tokens=[256], inference_steps=[10], seeds=[42])
        def instantiate(cfg, **kwargs):
            return dataset if cfg._target_=='fixture.dataset' else model
        with patch('hydra.utils.instantiate', side_effect=instantiate):
            yield module, args


class FullReportTests(unittest.TestCase):
    def test_full_pipeline_exports_metrics_inputs_arrays_and_plots(self):
        with full_pipeline_fixture() as (module, args):
            self.assertEqual(module.run(args), 0)
            summary = json.loads((args.out/'summary.json').read_text())
            self.assertEqual(summary['completed'], 1)
            self.assertEqual(summary['failures'], [])
            item = summary['results'][0]
            self.assertTrue(item['code_metrics']['exact_match'])
            self.assertEqual(item['valid_action_steps'], 32)
            self.assertLess(item['physical']['left']['position_error_mm']['rmse'], 1e-3)
            case = args.out/item['folder']
            for name in ['metrics.json','code_metrics.json','chunks.npz','prediction.py',
                         'observation_model_input/contact_sheet.png','action20_all_dimensions.png']:
                self.assertTrue((case/name).is_file(), name)

    def test_plot_error_does_not_discard_completed_metrics(self):
        with full_pipeline_fixture() as (module, args):
            with patch('agenticwam.utils.custom34_review.write_comparison_plots', side_effect=RuntimeError('plot failure')):
                self.assertEqual(module.run(args), 2)
            summary = json.loads((args.out/'summary.json').read_text())
            self.assertEqual(summary['completed'], 1)
            self.assertEqual(summary['failures'], [])
            self.assertEqual(summary['artifact_warning_cases'], 1)
            case = args.out/summary['results'][0]['folder']
            self.assertTrue((case/'metrics.json').is_file())
            self.assertTrue((case/'plot_export_error.log').is_file())

    def test_tail_padding_is_excluded_from_metrics(self):
        with full_pipeline_fixture(start=30, plots=False) as (module, args):
            self.assertEqual(module.run(args), 0)
            summary = json.loads((args.out/'summary.json').read_text())
            item = summary['results'][0]
            self.assertEqual(item['valid_action_steps'], 10)
            metrics = json.loads((args.out/item['folder']/'metrics.json').read_text())
            self.assertEqual(metrics['valid_steps'], 10)
            self.assertEqual(metrics['padded_steps'], 22)


if __name__ == '__main__':
    unittest.main()
