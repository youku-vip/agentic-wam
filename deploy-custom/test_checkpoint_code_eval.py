import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
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


if __name__ == '__main__':
    unittest.main()
