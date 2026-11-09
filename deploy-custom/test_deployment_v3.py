"""Deployment configuration, prompt parity and rejected-response tests without robot SDKs."""
import ast
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from agenticwam.utils.deployment_text_cache import CachedCosmosTextEncoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('deployment_v3_server', HERE / 'custom34_server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
client = runpy.run_path(str(HERE / 'test_custom34_rtc_adapter.py'))
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def args(**values):
    return SimpleNamespace(**dict(dict(train_config=None, override=[], task=None, dataset=None,
                                      skill_prompt_map=None, dataset_stats=None), **values))


def test_saved_run_config_loads_without_task_group_files(tmp_path):
    snapshot = tmp_path / 'config.yaml'
    OmegaConf.save(OmegaConf.create({'model': {'action_dim': 64}, 'data': {'train': {
        'pretrained_norm_stats': '/old/path/stats.json', 'skill_prompt_map': None}},
        'paths': {'code_model': '/old/weights'}}), snapshot)
    options = args(train_config=str(snapshot), override=['paths.code_model=/new/weights',
                                                       'model.n_cameras=3'])
    cfg = server.load_training_config(options)
    assert cfg.paths.code_model == '/new/weights' and cfg.model.n_cameras == 3
    assert OmegaConf.load(snapshot).paths.code_model == '/old/weights'
    options.override = ['data=custom1']
    with pytest.raises(ValueError, match='dotted'):
        server.load_training_config(options)


def test_exact_run_stats_precede_dataset_stats(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('{}')
    frozen = tmp_path / 'dataset_stats.json'
    frozen.write_text('{}')
    explicit = tmp_path / 'explicit.json'
    explicit.write_text('{}')
    cfg = OmegaConf.create({'data': {'train': {'pretrained_norm_stats': '/gone/dataset_stats.json'}}})
    assert server.select_stats_file(args(train_config=str(config)), cfg) == frozen
    assert server.select_stats_file(args(train_config=str(config), dataset_stats=str(explicit)), cfg) == explicit


@pytest.mark.parametrize('only', [False, True])
def test_code_prompt_matches_training_per_task(tmp_path, only):
    path = tmp_path / 'skills.json'
    path.write_text(json.dumps({'open fryer': 'OPEN APIs\n', 'pick cake': 'PICK APIs\n'}))
    cfg = OmegaConf.create({'skill_prompt_map': str(path), 'skill_prompt_only': only})
    builder = server.make_code_prompt_builder(cfg, DEFAULT_PROMPT)
    for task, skill in json.loads(path.read_text()).items():
        assert builder(task) == skill + ('' if only else DEFAULT_PROMPT.format(task=task))
    with pytest.raises(ValueError, match='absent'):
        builder('unknown task')
    cfg.code_instruction_mode = 'episode'
    assert server.make_code_prompt_builder(cfg, DEFAULT_PROMPT)('pick cake') == DEFAULT_PROMPT.format(task='pick cake')


def test_history_recipe_is_not_silently_dropped():
    with pytest.raises(ValueError, match='history'):
        server.make_code_prompt_builder(OmegaConf.create({'code_slices': True}), DEFAULT_PROMPT)


def test_embedding_cache_matches_training_hash_and_is_bounded(tmp_path):
    prompts = ['open fryer', 'pick cake', 'turn knob']
    tensors = []
    for i, prompt in enumerate(prompts):
        emb = torch.arange(12).reshape(3, 4).to(torch.bfloat16) + i
        tensors.append(emb)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        torch.save({'prompt_embs': emb}, tmp_path / f'{digest}.cosmos_reason1.pt')
    cache = CachedCosmosTextEncoder(tmp_path, max_entries=2)
    for prompt, expected in zip(prompts, tensors):
        actual = cache.encode(prompt)
        assert actual.device.type == 'cpu' and torch.equal(actual, expected[None])
    assert len(cache._cache) == 2 and prompts[0] not in cache._cache
    with pytest.raises(FileNotFoundError, match='exact training instruction'):
        cache.encode('unknown task')
    torch.save({'prompt_embs': torch.zeros(3, 4)}, tmp_path / f'{hashlib.sha256(b"bad").hexdigest()}.cosmos_reason1.pt')
    with pytest.raises(ValueError, match='all-zero'):
        cache.encode('bad')


def test_actual_policy_method_uses_config_prompt_despite_legacy_env(monkeypatch):
    tree = ast.parse((ROOT / 'robotwin/deploy/deploy_policy.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AgenticWAMCosmosRobotWinPolicy')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_infer_action_chunk')
    namespace = dict(Dict=Dict, Any=Any, torch=torch, np=np, os=os, DEFAULT_PROMPT=DEFAULT_PROMPT)
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'actual_deployment_method', 'exec'), namespace)
    seen = {}
    def infer(**kwargs):
        seen.update(kwargs)
        return torch.zeros(1, 32, 64), 'return robot.pick()'
    def encode(prompt):
        seen['text_prompt'] = prompt
        return torch.ones(1, 2, 3)
    policy = SimpleNamespace(
        _build_obs_video=lambda _: torch.zeros(1), _normalize_state=lambda _: torch.zeros(1, 64),
        device='cpu', code_prompt_builder=lambda task: 'CONFIG ' + task,
        text=SimpleNamespace(encode=encode), video_log_dir=None, step_count=1,
        model=SimpleNamespace(use_code_expert=True, infer_action_trimodal=infer),
        action_horizon=32, num_video_frames=9, num_inference_steps=10, sigma_shift=None,
        code_cache={}, _denormalize_action=lambda a: a.numpy())
    monkeypatch.setenv('AGENTICWAM_SKILL_PROMPT_ONLY', '1')
    monkeypatch.setenv('AGENTICWAM_SLICE_SUMMARY', '1')
    output = namespace[method.name](policy, {'joint_action': {'vector': np.zeros(34)}}, 'pick cake')
    assert output.shape == (32, 64)
    assert seen['instruction'] == 'CONFIG pick cake'
    assert seen['text_prompt'] == DEFAULT_PROMPT.format(task='pick cake')
    assert not hasattr(policy.model, '_mem_code_history')


def test_gripper_defaults_and_repeated_prompt_revision(monkeypatch):
    monkeypatch.delenv('AGENTICWAM_AUTO_GRIPPER_RECOVERY', raising=False)
    monkeypatch.delenv('AGENTICWAM_GRIPPER_LOCK_STEPS', raising=False)
    assert client['deployment_control_options']() == (False, 0)
    monkeypatch.setenv('AGENTICWAM_AUTO_GRIPPER_RECOVERY', '1')
    monkeypatch.setenv('AGENTICWAM_GRIPPER_LOCK_STEPS', '15')
    assert client['deployment_control_options']() == (True, 15)
    shared = client['SharedPrompt']('pick cake')
    before = shared.snapshot()
    shared.set('pick cake')
    assert shared.snapshot() == ('pick cake', before[1] + 1)


def test_observation_cannot_overwrite_protocol_state():
    obs = client['fake_obs']()
    obs.update(state=np.zeros(20), images={'evil': 'replacement'})
    result = client['convert_obs'](obs)
    assert result['state'].shape == (34,) and set(result['images']) == {'cam_high', 'cam_left_wrist', 'cam_right_wrist'}


def test_changed_backend_rejects_action_then_rehandshakes():
    c = client['RTCClient']('http://example.com', 'pick cake', True, s_min=20)
    state = {'id': 'first'}
    paths = []
    async def request(method, path, body=None):
        paths.append(path)
        base = dict(status='success', server_id=state['id'])
        if path == '/health':
            return dict(base, state_dim=34, action_dim=20, model_dim=64, chunk_size=32,
                        protocol_version=2, action_type='absolute_eef20', rotation6d='first_two_columns', gripper_open_value=1)
        if path == '/reset':
            return base
        actions = np.zeros((32, 20), np.float32)
        actions[:, 3:9] = actions[:, 12:18] = [1, 0, 0, 0, 1, 0]
        return dict(base, action=actions.tolist(), request_id=body['request_id'])
    c._request = request
    async def scenario():
        await c.connect()
        state['id'] = 'second'
        with pytest.raises(ConnectionError, match='instance changed'):
            await c.guided_inference(client['fake_obs']())
        assert c._needs_handshake
        out = await c.guided_inference(client['fake_obs']())
        assert out['action'].shape == (32, 20) and c._server_id == 'second'
        await c.stop()
    asyncio.run(scenario())
    assert paths == ['/health', '/reset', '/act', '/health', '/reset', '/act']


@pytest.mark.parametrize('bad', ['rotation', 'request_id', 'shape', 'http'])
def test_bad_response_never_returns_an_action(bad):
    c = client['RTCClient']('http://example.com', 'pick cake', True, s_min=20)
    c._protocol_version = 2
    c._server_id = 'same'
    async def request(method, path, body=None):
        if bad == 'http':
            raise ConnectionError('HTTP timeout')
        actions = np.zeros((32, 20), np.float32)
        actions[:, 3:9] = actions[:, 12:18] = [1, 0, 0, 0, 1, 0]
        if bad == 'rotation': actions[1, 3:9] = 0
        if bad == 'shape': actions = actions[:1]
        return dict(status='success', server_id='same', action=actions.tolist(),
                    request_id='other' if bad == 'request_id' else body['request_id'])
    c._request = request
    async def scenario():
        with pytest.raises(ConnectionError):
            await c.guided_inference(client['fake_obs']())
        assert c.inference_count == 0 and c._needs_handshake
    asyncio.run(scenario())
