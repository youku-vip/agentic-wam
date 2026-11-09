"""CPU tests for generated-view export, native action errors and packed v3 expert alignment."""
import ast
import asyncio
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any, Dict
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from agenticwam.utils.custom34_review import (compare_actions, physical_errors, rotation_matrices,
                                             save_video_frames, validate_reference, write_review)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'deploy-custom'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load('test_client_server_contract')
server = contract.server
review = load('review_custom34_chunks')


def yaw6d(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return [c, s, 0, -s, c, 0]


def actions(n=32):
    a = np.zeros((n, 20), np.float32)
    a[:, :3] = [.25, .1, .3]
    a[:, 9:12] = [-.2, .1, .4]
    a[:, 3:9] = a[:, 12:18] = yaw6d(0)
    a[:, 18:] = [.25, .75]
    return a


def video(nframes=9):
    value = torch.full((1, 3, 3, nframes, 16, 32), -1.)
    for role in range(3): value[0, role, role] = 1
    return value.reshape(1, 3, 3*nframes, 16, 32)


def test_errors_use_units_same_timesteps_and_exclude_padding():
    target = actions()
    pred = target.copy()
    pred[:, :3] += [.001, -.002, .002]
    pred[:, 3:9] = yaw6d(90)
    pred[:, 18] = .35
    pred[:, 19] = .25
    pred[30:, 0] += 1000
    reference = validate_reference(dict(action20=target, valid_mask=[True]*30+[False]*2, fps=10), 32)
    result = compare_actions(pred, reference)
    assert result['valid_steps'] == 30 and result['padded_steps'] == 2
    assert result['per_dimension'][0]['mae'] == pytest.approx(.001, abs=1e-7)
    assert result['physical']['left']['position_error_mm']['mean'] == pytest.approx(3, abs=1e-4)
    assert result['physical']['left']['rotation_error_deg']['mean'] == pytest.approx(90)
    assert result['physical']['left']['gripper_accuracy'] == 1
    assert result['physical']['right']['gripper_accuracy'] == 0


def test_rotation_errors_do_not_subtract_wrapped_euler_angles():
    pred, target = actions(), actions()
    pred[:, 3:9] = yaw6d(-179)
    target[:, 3:9] = yaw6d(179)
    assert np.allclose(physical_errors(pred, target)['left']['rotation_error_deg'], 2, atol=1e-5)


@pytest.mark.parametrize('change', ['shape', 'mask', 'empty_mask', 'rotation', 'fps'])
def test_bad_reference_is_rejected(change):
    reference = dict(action20=actions().tolist())
    if change == 'shape': reference['action20'] = reference['action20'][:3]
    if change == 'mask': reference['valid_mask'] = [1]*32
    if change == 'empty_mask': reference['valid_mask'] = [False]*32
    if change == 'rotation': reference['action20'][0][3:9] = [0]*6
    if change == 'fps': reference['fps'] = float('nan')
    with pytest.raises(ValueError): validate_reference(reference, 32)


def test_camera_major_frame_export_and_rgb_are_exact(tmp_path):
    result = save_video_frames(video(2), tmp_path/'generated', ('left', 'head', 'right'), step_stride=4)
    assert result['frames_per_camera'] == 2 and result['total_images'] == 6
    assert np.array_equal(np.array(Image.open(tmp_path/'generated/left/frame_000.png'))[0, 0], [255, 0, 0])
    assert np.array_equal(np.array(Image.open(tmp_path/'generated/head/frame_001.png'))[0, 0], [0, 255, 0])
    frames = json.loads((tmp_path/'generated/frames.json').read_text())
    assert frames[0]['kind'] == 'conditioned_observation'
    assert frames[1]['kind'] == 'predicted_future' and frames[1]['state_step_offset'] == 4


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_frames_are_diagnosed_and_never_saved_as_fake_black_images(tmp_path, bad_value):
    pixels = video(2)
    pixels[0, 0, 1, 0, 0] = bad_value
    out = tmp_path/'generated'
    report = save_video_frames(pixels, out, ('head', 'left', 'right'))
    assert report['status'] == 'partial' and report['invalid_frame_count'] == 1
    assert report['total_images'] == 5 and not (out/'head/frame_001.png').exists()
    assert (out/'head/frame_000.png').is_file()
    assert sum(report[key] for key in ('nan_count', 'posinf_count', 'neginf_count')) == 1
    with np.load(out/'nonfinite_video.npz', allow_pickle=False) as saved:
        assert not np.isfinite(saved['video'][0, 0, 1, 0, 0])
    frames = json.loads((out/'frames.json').read_text())
    assert frames[1]['status'] == 'nonfinite' and frames[1]['file'] is None


def test_wrong_video_shape_has_a_distinct_error(tmp_path):
    with pytest.raises(ValueError, match='video shape error'):
        save_video_frames(np.zeros((1, 3, 28, 16, 32)), tmp_path/'bad', ('head', 'left', 'right'))


def test_actual_policy_requests_generated_frames_on_every_inference():
    tree = ast.parse((ROOT/'robotwin/deploy/deploy_policy.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AgenticWAMCosmosRobotWinPolicy')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_infer_action_chunk')
    namespace = dict(Dict=Dict, Any=Any, torch=torch, np=np, os=os,
                     DEFAULT_PROMPT='Task: {task}')
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'actual_policy_capture', 'exec'), namespace)
    calls = []
    def infer(**kwargs):
        calls.append(kwargs)
        return torch.zeros(1, 32, 64), 'code', video()
    obs = video(1)
    policy = SimpleNamespace(capture_generated_video=True, _build_obs_video=lambda _: obs,
                             _normalize_state=lambda _: torch.zeros(1, 64), device='cpu',
                             code_prompt_builder=lambda text: text,
                             text=SimpleNamespace(encode=lambda _: torch.ones(1, 2, 3)),
                             video_log_dir=None, step_count=10, action_horizon=32, num_video_frames=9,
                             num_inference_steps=10, sigma_shift=None, code_cache={},
                             model=SimpleNamespace(use_code_expert=True, infer_action_trimodal=infer),
                             _denormalize_action=lambda value: value.numpy())
    for _ in range(2):
        namespace[method.name](policy, {'joint_action': {'vector': np.zeros(34)}}, 'open fryer')
        assert torch.equal(policy._last_obs_video, obs)
        assert torch.equal(policy._last_generated_video, video())
    assert len(calls) == 2 and all(call['return_video'] for call in calls)


def runtime(tmp_path):
    rt = contract.runtime()
    rt.server_id = 'a'*32
    rt.review_dir = tmp_path
    rt.stats_sha256 = 'mock-stats'
    rt.checkpoint = 'MOCK MODEL - not a real checkpoint'
    rt.video_size = [16, 32]
    rt.sample_stride = 1
    rt.video_action_ratio = 4
    original = rt.policy._infer_action_chunk
    def infer(ob, instruction):
        assert set(ob) == {'observation', 'joint_action'}  # Expert actions never enter inference.
        out = original(ob, instruction)
        rt.policy._last_generated_video = video()
        rt.policy._last_obs_video = video(1)
        return out
    rt.policy._infer_action_chunk = infer
    return rt


def test_server_review_and_relative64_use_same_observation_anchor(tmp_path):
    rt = runtime(tmp_path)
    body = contract.request_body()
    body['request_id'] = 'test-reference'
    body['reference'] = dict(action20=actions().tolist(), fps=10, source=dict(kind='expert_dataset_action'))
    result, _, _ = rt.infer(body)
    assert rt.last_review['status'] == 'saved'
    out = Path(rt.last_review['path'])
    with np.load(out/'chunks.npz', allow_pickle=False) as data:
        assert np.array_equal(data['pred_action20'], result)
        assert np.allclose(data['pred_relative64'], data['gt_relative64'], atol=1e-7)
        assert np.allclose(data['gt_relative64'][:, 0], .05)
    report = json.loads((out/'metrics.json').read_text())
    assert len(report['per_dimension']) == 20 and len(report['relative64']['per_dimension']) == 64
    assert report['physical']['left']['position_error_mm']['maximum'] < 1e-4
    assert (out/'action20_all_dimensions.png').is_file()
    assert len(list((out/'dimensions20').glob('*.png'))) == 20
    assert (out/'relative64_all_dimensions.png').is_file()
    assert len(list((out/'generated').glob('*/frame_*.png'))) == 27
    metadata = json.loads((out/'metadata.json').read_text())
    assert metadata['comparison_uses_reference_as_model_input'] is False


def test_review_write_failure_does_not_discard_computed_action(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    def fail(*args, **kwargs): raise OSError('disk full')
    monkeypatch.setattr('agenticwam.utils.custom34_review.write_review', fail)
    result, _, _ = rt.infer(contract.request_body())
    assert result.shape == (32, 20) and rt.calls == 1
    assert rt.last_review['status'] == 'error' and 'disk full' in rt.last_review['message']


def test_nonfinite_video_preserves_real_action_comparison_figures_and_arrays(tmp_path):
    rt = runtime(tmp_path)
    original = rt.policy._infer_action_chunk
    def infer(ob, instruction):
        result = original(ob, instruction)
        rt.policy._last_generated_video = torch.full_like(video(), float('nan'))
        return result
    rt.policy._infer_action_chunk = infer
    body = contract.request_body()
    body['reference'] = dict(action20=actions().tolist())
    prediction, _, _ = rt.infer(body)
    result = rt.last_review
    assert result['status'] == 'partial' and result['comparison_ready'] and not result['images_valid']
    out = Path(result['path'])
    assert (out/'action20_all_dimensions.png').is_file()
    assert (out/'relative64_all_dimensions.png').is_file()
    assert len(list((out/'dimensions20').glob('*.png'))) == 20
    with np.load(out/'chunks.npz', allow_pickle=False) as values:
        assert np.array_equal(values['pred_action20'], prediction)
        assert np.isfinite(values['gt_action20']).all()
    assert not list((out/'generated').glob('*/frame_*.png'))
    assert (out/'generated/nonfinite_video.npz').is_file()


def test_plot_failure_preserves_arrays_and_marks_review_incomplete(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    def fail(*args, **kwargs): raise RuntimeError('plot unavailable')
    monkeypatch.setattr('agenticwam.utils.custom34_review.write_comparison_plots', fail)
    body = contract.request_body()
    body['reference'] = dict(action20=actions().tolist())
    result, _, _ = rt.infer(body)
    assert result.shape == (32, 20) and rt.last_review['status'] == 'error'
    with np.load(Path(rt.last_review['path'])/'chunks.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['pred_action20'], result)
        assert 'gt_action20' in saved


def make_dataset(root, source=None):
    import av
    root.mkdir()
    (root/'meta/episodes/chunk-000').mkdir(parents=True)
    (root/'data/chunk-000').mkdir(parents=True)
    keys = ['observation.images.'+role for role in ('head', 'left', 'right')]
    count = 80 if source is None else 35
    info = dict(codebase_version='v3.0', fps=10, total_episodes=2 if source is None else 1,
                total_frames=count, data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
                features={key: dict(dtype='video') for key in keys})
    (root/'meta/info.json').write_text(json.dumps(info))
    (root/'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0, task='open fryer'))+'\n')
    rows, episodes = [], []
    for episode in range(2 if source is None else 1):
        length = 40 if source is None else 35
        start = episode*40 if source is None else 0
        metadata = dict(episode_index=episode, dataset_from_index=start, dataset_to_index=start+length,
                        length=length, **{'data/chunk_index': 0, 'data/file_index': 0})
        for key in keys:
            metadata.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': 0,
                             f'videos/{key}/from_timestamp': episode*4 if source is None else 4.5})
        episodes.append(metadata)
        for frame in range(length):
            a = actions(1)[0]
            a[0] += (frame if source is None else frame+5)*.001
            state = contract.raw_state()
            rows.append(dict(episode_index=episode, frame_index=frame, task_index=0,
                             timestamp=1234+frame*.2, **{'observation.state': state, 'action': a}))
    pd.DataFrame(rows).to_parquet(root/'data/chunk-000/file-000.parquet', index=False)
    pd.DataFrame(episodes).to_parquet(root/'meta/episodes/chunk-000/file-000.parquet', index=False)
    for role, key in enumerate(keys):
        path = root/f'videos/{key}/chunk-000/file-000.mp4'
        path.parent.mkdir(parents=True)
        if source is not None:
            path.symlink_to(source/f'videos/{key}/chunk-000/file-000.mp4')
            continue
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('libx264', rate=10)
            stream.width, stream.height, stream.pix_fmt = 32, 16, 'yuv420p'
            for i in range(80):
                rgb = np.zeros((16, 32, 3), np.uint8)
                rgb[:, :, role] = 60+i
                frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
    return root


def test_raw_and_curated_roots_use_packed_video_offsets_and_action_t(tmp_path):
    from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
    raw = make_dataset(tmp_path/'raw')
    clean = make_dataset(tmp_path/'clean', source=raw)
    for root, episode, expected_pts, expected_x in ((raw, 1, 4., .25), (clean, 0, 4.5, .255)):
        item = review.sample_window(V3EpisodeReader(root), episode, 0)
        assert item['reference']['action20'][0][0] == pytest.approx(expected_x)
        assert item['state'][14] == pytest.approx(.2)  # It is NOT substituted for action[t].
        assert item['timing']['head']['decoded_timestamp'] == pytest.approx(expected_pts)
        assert item['timing']['head']['target_timestamp'] == pytest.approx(expected_pts)
        assert len(item['reference']['action20']) == 32


def test_tail_padding_is_marked_and_cannot_cross_episode(tmp_path):
    from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
    reader = V3EpisodeReader(make_dataset(tmp_path/'raw'))
    with pytest.raises(ValueError, match='only 31 real steps'):
        review.sample_window(reader, 0, 9)
    item = review.sample_window(reader, 0, 9, padding='repeat')
    assert item['reference']['valid_mask'] == [True]*31+[False]
    assert item['reference']['source']['action_frame_indices'][-2:] == [39, 39]


@pytest.mark.parametrize('nonfinite', [False, True])
def test_offline_client_downloads_actual_http_review_bundle(tmp_path, monkeypatch, nonfinite):
    # Plotting itself is exercised above. Keep the HTTP/packed-video roundtrip lightweight.
    monkeypatch.setattr('agenticwam.utils.custom34_review.write_comparison_plots', lambda *a, **k: None)
    rt = runtime(tmp_path/'server_reviews')
    if nonfinite:
        original = rt.policy._infer_action_chunk
        def infer(ob, instruction):
            result = original(ob, instruction)
            rt.policy._last_generated_video = torch.full_like(video(), float('nan'))
            return result
        rt.policy._infer_action_chunk = infer
    holder, ready = [], threading.Event()
    class TestServer(server.InferenceServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder.append(self)
            ready.set()
    monkeypatch.setattr(server, 'InferenceServer', TestServer)
    monkeypatch.setattr(server, 'Custom34Runtime', lambda _: rt)
    monkeypatch.setattr(sys, 'argv', ['custom34_server.py', '--ckpt', 'mock', '--port', '0'])
    thread = threading.Thread(target=server.main, daemon=True)
    thread.start()
    assert ready.wait(5)
    base = f'http://127.0.0.1:{holder[0].server_address[1]}'
    try:
        args = SimpleNamespace(root=make_dataset(tmp_path/'raw'), server=base, timeout=60,
                               camera_key=[], episodes=[0], start_frames=[0], padding='error', out=tmp_path/'results')
        assert review.run(args) == (2 if nonfinite else 0)
        image_file = 'nonfinite_video.npz' if nonfinite else 'head/frame_008.png'
        assert (args.out/'episode000000_frame000000/generated'/image_file).is_file()
        assert json.loads((args.out/'summary.json').read_text())['valid_action_steps'] == 32
        assert json.loads((args.out/'summary.json').read_text())['image_warning_chunks'] == int(nonfinite)
        assert (args.out/'index.html').is_file()
        with pytest.raises(Exception):
            urlopen(base+'/reviews/../../etc/passwd/bundle.zip')
    finally:
        holder[0].shutdown()
        thread.join(5)
    assert not thread.is_alive()
