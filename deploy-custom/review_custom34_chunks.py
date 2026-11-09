#!/usr/bin/env python3
"""Compare server predictions with expert action windows from raw or curated LeRobot v3 data."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def request_json(base, path, payload=None, timeout=180):
    raw = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = Request(base.rstrip('/') + path, data=raw,
                      headers={} if raw is None else {'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except HTTPError as exc:
        raise RuntimeError(f'{path}: HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc
    if not isinstance(value, dict) or value.get('status') != 'success':
        raise RuntimeError(f'{path}: server rejected request: {value}')
    return value


def camera_keys(video_keys, overrides=()):
    aliases = dict(head={'head', 'head_camera', 'cam_high', 'top'},
                   left={'left', 'left_camera', 'cam_left_wrist'},
                   right={'right', 'right_camera', 'cam_right_wrist'})
    given = {}
    for item in overrides:
        role, sep, key = item.partition('=')
        if not sep or role not in aliases or key not in video_keys:
            raise ValueError(f'invalid --camera-key {item!r}; expected role=existing.video.key')
        given[role] = key
    for role, names in aliases.items():
        if role not in given:
            matches = [key for key in video_keys if key.rsplit('.', 1)[-1] in names]
            if len(matches) != 1:
                raise ValueError(f'cannot resolve {role} camera from {video_keys}; use --camera-key {role}=...')
            given[role] = matches[0]
    if len(set(given.values())) != 3:
        raise ValueError('head/left/right must resolve to three distinct cameras')
    return given


def decode_rgb(path, timestamp, tolerance):
    """Nearest PTS around the encoded-video timestamp, including the packed episode offset."""
    import av
    best = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((timestamp-tolerance) / float(stream.time_base))), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * frame.time_base)
            error = abs(pts-timestamp)
            if best is None or error < best[0]:
                best = error, pts, frame.to_ndarray(format='rgb24')
            if pts >= timestamp:
                break
    if best is None or best[0] > tolerance:
        raise ValueError(f'video alignment failed: {path}, target={timestamp}, nearest={None if best is None else best[:2]}')
    return best[2], dict(target_timestamp=timestamp, decoded_timestamp=best[1], error_seconds=best[0])


def sample_window(reader, episode, start, *, horizon=32, stride=1, padding='error', keys=None):
    frame = reader.read_episode(episode, columns=['observation.state', 'action', 'task_index'])
    if not 0 <= start < len(frame):
        raise ValueError(f'episode {episode}: start {start} outside [0,{len(frame)})')
    rows = start + np.arange(horizon) * stride
    valid = rows < len(frame)
    if not valid.all() and padding == 'error':
        raise ValueError(f'episode {episode} frame {start}: only {int(valid.sum())} real steps; '
                         'choose an earlier start or --padding repeat (padding excluded from metrics)')
    clipped = np.minimum(rows, len(frame)-1)
    state = np.asarray(frame.iloc[start]['observation.state'], dtype=np.float32)
    actions = np.stack(frame.iloc[clipped]['action']).astype(np.float32)
    task_index = int(frame.iloc[start]['task_index'])
    if not np.all(frame.iloc[clipped]['task_index'].to_numpy() == task_index):
        raise ValueError('a chunk must not cross a task boundary')
    if state.shape != (34,) or actions.shape != (horizon, 20) or not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError('dataset must provide finite raw state34 and absolute action20')
    keys = keys or camera_keys(reader.video_keys)
    images, timing = {}, {}
    for role, key in keys.items():
        target = reader.video_timestamps(episode, key, [start])[0]
        path = reader.video_file(episode, key)
        images[role], timing[role] = decode_rgb(path, target, 1/reader.fps-1e-4)
        timing[role].update(video_key=key, file=str(path))
    source = dict(kind='expert_dataset_action', root=str(reader.root), episode=int(episode),
                  start_frame=int(start), task_index=task_index, stride=stride,
                  action_frame_indices=clipped.tolist(), padding=padding, source_fps=reader.fps,
                  info_sha256=hashlib.sha256((reader.root/'meta/info.json').read_bytes()).hexdigest(),
                  note='Expert commands from action, not measured future observation.state.')
    reference = dict(action20=actions.tolist(), valid_mask=valid.tolist(), fps=reader.fps/stride, source=source)
    return dict(state=state, images=images, instruction=reader.tasks[task_index], reference=reference, timing=timing)


def encode_png(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def download_review(base, path, output, timeout):
    if not path.startswith('/reviews/') or not path.endswith('/bundle.zip'):
        raise ValueError('server returned an unexpected review download path')
    with urlopen(base.rstrip('/')+path, timeout=timeout) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or '..' in relative.parts or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('unsafe review archive member')
        if archive.testzip() is not None:
            raise ValueError('review archive checksum failed')
        archive.extractall(output)


def write_index(output, results, failures):
    image_warnings = [r for r in results if not r.get('images_valid', True)]
    summary = dict(evaluation='offline expert-observation chunk prediction, not closed-loop task success',
                   chunks=len(results), failed_chunks=len(failures),
                   image_warning_chunks=len(image_warnings),
                   valid_action_steps=sum(r['valid_steps'] for r in results), results=results, failures=failures)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    page = '<!doctype html><meta charset="utf-8"><title>Custom34 expert comparison</title>'
    page += '<style>body{font-family:sans-serif;margin:24px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>'
    page += '<h1>Custom34: prediction vs expert action chunks</h1><p>Offline predictions from expert observations; padded steps excluded.</p>'
    page += '<table><tr><th>Chunk</th><th>Task</th><th>Valid steps</th><th>Left position RMSE (mm)</th><th>Right position RMSE (mm)</th></tr>'
    for result in results:
        page += (f'<tr><td><a href="{result["folder"]}/index.html">{result["folder"]}</a></td>'
                 f'<td>{html.escape(result["instruction"])}</td><td>{result["valid_steps"]}</td>'
                 f'<td>{result["physical"]["left"]["position_error_mm"]["rmse"]:.3f}</td>'
                 f'<td>{result["physical"]["right"]["position_error_mm"]["rmse"]:.3f}</td></tr>')
    page += '</table>'
    if failures: page += '<h2>Failures</h2><pre>'+html.escape(json.dumps(failures, indent=2, ensure_ascii=False))+'</pre>'
    if image_warnings:
        page += '<p style="color:red">Some image exports failed or contain NaN/Inf. The action comparisons above remain available; see each chunk diagnostics.</p>'
    (output/'index.html').write_text(page)


def run(args):
    from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
    base = args.server.rstrip('/')
    health = request_json(base, '/health', timeout=args.timeout)
    if not health.get('review_enabled'):
        raise ValueError('install the chunk-review server update and start it without --no-review')
    if int(health.get('chunk_size', -1)) != 32 or int(health.get('action_dim', -1)) != 20:
        raise ValueError('this evaluation expects server horizon=32 and absolute action20')
    if health.get('action_type') != 'absolute_eef20' or health.get('rotation6d') != 'first_two_columns':
        raise ValueError('server must return de-normalized absolute EEF20 with first-two-column rotation6D')
    stride = int(health.get('sample_stride', 1))
    if stride < 1:
        raise ValueError('server sample_stride must be positive')
    reader = V3EpisodeReader(args.root)
    keys = camera_keys(reader.video_keys, args.camera_key)
    episodes = list(dict.fromkeys(args.episodes))
    starts = list(dict.fromkeys(args.start_frames))
    for episode in episodes: reader.episode(episode)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results, failures = [], []
    for episode in episodes:
        for start in starts:
            folder = f'episode{episode:06d}_frame{start:06d}'
            try:
                sample = sample_window(reader, episode, start, stride=stride, padding=args.padding, keys=keys)
                reset = request_json(base, '/reset', {}, args.timeout)
                if reset.get('server_id') != health.get('server_id'):
                    raise RuntimeError('server instance changed before inference; restart evaluation')
                request_id = 'eval-' + os.urandom(16).hex()
                payload = dict(instruction=sample['instruction'], state=sample['state'].tolist(), request_id=request_id,
                               images={role: encode_png(image) for role, image in sample['images'].items()},
                               reference=sample['reference'])
                response = request_json(base, '/act', payload, args.timeout)
                if response.get('server_id') != health.get('server_id') or response.get('request_id') != request_id:
                    raise RuntimeError('mismatched server instance or request ID; refusing comparison')
                review = response.get('review') or {}
                if review.get('status') not in ('saved', 'partial') or not review.get('has_reference'):
                    raise RuntimeError(f'server did not save images/comparison: {review}')
                if review.get('status') == 'partial' and not review.get('comparison_ready'):
                    raise RuntimeError(f'action comparison is not complete: {review}')
                case = output/folder
                case.mkdir()
                download_review(base, review['download_url'], case, args.timeout)
                with np.load(case/'chunks.npz', allow_pickle=False) as arrays:
                    if not np.array_equal(arrays['pred_action20'], np.asarray(response['action'])):
                        raise ValueError('downloaded prediction differs from the action returned for this request')
                    if not np.array_equal(arrays['gt_action20'], np.asarray(sample['reference']['action20'])):
                        raise ValueError('downloaded expert chunk differs from the selected dataset window')
                metadata = json.loads((case/'metadata.json').read_text())
                if metadata.get('request_id') != request_id:
                    raise ValueError('downloaded review belongs to another request')
                (case/'video_alignment.json').write_text(json.dumps(sample['timing'], indent=2)+'\n')
                report = json.loads((case/'metrics.json').read_text())
                result = dict(folder=folder, episode=episode, start_frame=start, instruction=sample['instruction'],
                              images_valid=metadata.get('images_valid', True),
                              image_warnings=metadata.get('warnings', []),
                              valid_steps=report['valid_steps'], physical=report['physical'], server_review=review)
                results.append(result)
                print(f'[review] {folder}: {report["valid_steps"]}/32 valid steps; saved {case}', flush=True)
                if not result['images_valid']:
                    print(f'[review] IMAGE WARNING {folder}: action curves saved; inspect metadata.json and generated/video_diagnostics.json',
                          file=sys.stderr, flush=True)
            except Exception as exc:
                failures.append(dict(episode=episode, start_frame=start, error=f'{type(exc).__name__}: {exc}'))
                print(f'[review] FAILED {folder}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
            write_index(output, results, failures)
    partial = sum(not r['images_valid'] for r in results)
    print(json.dumps(dict(output=str(output), compared_chunks=len(results), passed_chunks=len(results)-partial,
                          image_warning_chunks=partial, failed_chunks=len(failures)), indent=2))
    return 1 if failures else (2 if partial else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='raw OR curated single LeRobot v3 root')
    parser.add_argument('--server', default='http://127.0.0.1:8765')
    parser.add_argument('--episodes', nargs='+', type=int, required=True)
    parser.add_argument('--start-frames', nargs='+', type=int, default=[0])
    parser.add_argument('--padding', choices=('error', 'repeat'), default='error')
    parser.add_argument('--camera-key', action='append', default=[], help='optional head=observation.images.head, etc.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
