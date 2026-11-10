#!/usr/bin/env python3
"""Stage-aware checkpoint tests and CPU-only per-dimension expert/action plots."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluator():
    path = Path(__file__).with_name('evaluate_custom34_checkpoint.py')
    spec = importlib.util.spec_from_file_location('trajectory_checkpoint_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'evaluation_plans'):
        raise ValueError('install the accompanying evaluate_custom34_checkpoint.py v1.4')
    return module


def select_cases(single, codes, ids, episodes, *, frame_offset=0, horizon=32, stride=1,
                 points=('middle',), frames=None, min_valid_steps=1):
    """Select exact contiguous code-ID runs from the actual loader enumeration."""
    module = evaluator()
    reader = single.meta.reader
    ids = np.asarray(ids)
    cases, skipped = [], []
    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
        raise ValueError('code_id_by_idx must be a one-dimensional integer array')
    if frame_offset < 0 or horizon != 32 or stride < 1 or not 1 <= min_valid_steps <= horizon:
        raise ValueError('this test expects a 32-step action horizon and positive stride')
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError('provide distinct episode IDs')
    for episode in episodes:
        slot = single.episodes.index(episode)
        length = reader.episode_length(episode)
        begin = int(single._starts[slot])
        local = ids[frame_offset + begin:frame_offset + begin + length]
        if len(local) != length:
            raise ValueError('code cache does not cover this loader episode')
        raw = reader.read_episode(episode, columns=['task_index'])
        tasks = np.unique(raw['task_index'].to_numpy())
        if len(raw) != length or len(tasks) != 1:
            raise ValueError('expected one task per complete episode')
        task = int(tasks[0])
        edges = [0, *(np.flatnonzero(local[1:] != local[:-1]) + 1).tolist(), length]
        stage = 0
        for start, stop in zip(edges[:-1], edges[1:]):
            cid = int(local[start])
            if not 0 <= cid < len(codes) or not isinstance(codes[cid], str):
                raise ValueError('invalid code ID/program in the selected episode')
            target = codes[cid]
            if not target.strip():
                skipped.append(dict(episode=episode, start_frame=start, end_frame=stop,
                                    reason='unlabeled code run'))
                continue
            stage += 1
            calls = module.robot_calls(target)  # AST validation only; never execute code.
            positions = ({'start': start, 'middle': (start + stop - 1) // 2, 'end': stop - 1})
            chosen = sorted(set(f for f in frames if start <= f < stop)) if frames is not None else sorted(set(positions[p] for p in points))
            for frame in chosen:
                valid = min(horizon, (length - 1 - frame) // stride + 1)
                if valid < min_valid_steps:
                    skipped.append(dict(episode=episode, frame=frame, stage_index=stage,
                                        reason='fewer than min_valid_steps', valid_action_steps=valid))
                    continue
                cases.append(dict(episode=episode, frame=int(frame), global_index=begin + int(frame),
                    task_index=task, task_name=reader.tasks[task], stage_index=stage,
                    stage_name=' -> '.join(name + ':' + side for name, side in calls),
                    stage_start_frame=start, stage_end_frame_exclusive=stop, code_id=cid,
                    target_sha256=hashlib.sha256(target.strip().encode()).hexdigest(),
                    target_calls=calls, valid_action_steps=valid))
        if frames is not None:
            for frame in frames:
                if frame < 0 or frame >= length:
                    raise ValueError(f'episode {episode}: frame {frame} is out of range')
                if not any(c['episode'] == episode and c['frame'] == frame for c in cases):
                    raise ValueError(f'episode {episode}, frame {frame}: no labeled eligible case')
    if not cases:
        raise ValueError('no eligible stage cases; inspect episode IDs/min_valid_steps')
    return cases, skipped


def make_plan(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.train_config)
    stats = args.stats or args.train_config.with_name('dataset_stats.json')
    if not stats.is_file() and args.stats is None:
        stats = Path(str(cfg.data.train.pretrained_norm_stats))
    if not stats.is_file():
        raise FileNotFoundError(stats)
    if cfg.model.get('code_memory', False) or cfg.data.train.get('code_slices', False):
        raise ValueError('this tool expects ordinary custom34 code without history/slices')
    cfg.data.train.pretrained_norm_stats = str(stats.resolve())
    cfg.data.train.save_norm_stats = False
    cfg.data.train.skip_padding_as_possible = False
    print('[trajectory-test] reading training enumeration and code stages; no model or video decode', flush=True)
    dataset = instantiate(cfg.data.train)
    sources = dataset.lerobot_dataset.multi_dataset._datasets
    if len(sources) != 1 or dataset.code_id_by_idx is None:
        raise ValueError('expected one custom v3 root and an indexed policy code cache')
    single = sources[0]
    offset = int(dataset.frame_offset)
    if offset == 0 and len(dataset.code_id_by_idx) != len(dataset):
        raise ValueError('code cache length and training enumeration differ; keep the original training data selection')
    cases, skipped = select_cases(single, dataset.codes, dataset.code_id_by_idx.cpu().numpy(),
        args.episodes, frame_offset=offset, horizon=int(cfg.data.train.num_frames) - 1,
        stride=int(cfg.data.train.global_sample_stride), points=args.points,
        frames=args.frames, min_valid_steps=args.min_valid_steps)
    return dict(version='custom34-stage-plan-v1', root=str(single.meta.reader.root.resolve()),
        config_sha256=sha(args.train_config), stats_sha256=sha(stats),
        code_cache_sha256=sha(cfg.data.train.policy_code_idx_path),
        case_selection='code-ID runs in actual training loader order; target labels are scoring data only',
        cases=cases, skipped=skipped)


def case_fps(case, arrays, supplied):
    stored = float(arrays['sample_fps']) if 'sample_fps' in arrays else None
    for name in ('reference.json', 'metadata.json'):
        path = case / name
        if stored is None and path.is_file():
            metadata = json.loads(path.read_text())
            stored = metadata.get('fps') or metadata.get('reference', {}).get('fps')
    if supplied is not None and stored is not None and not np.isclose(supplied, stored):
        raise ValueError('--fps disagrees with the saved sample rate')
    fps = stored if supplied is None else supplied
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError('FPS is missing; provide --fps (e.g. 10 for the current stride-1 dataset)')
    return float(fps)


def render_case(case, output, *, fps=None, relative64=False):
    from agenticwam.utils.custom34_review import (ACTION20_NAMES, RELATIVE64_NAMES,
        validate_reference, compare_actions, write_comparison_plots)
    case, output = Path(case).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'choose a new plot directory: {output}')
    source = case / 'chunks.npz'
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: np.array(z[k], copy=True) for k in z.files}
    pred, target, mask = (arrays[k] for k in ('pred_action20', 'gt_action20', 'valid_mask'))
    if pred.ndim != 2 or pred.shape != target.shape or pred.shape[1] != 20:
        raise ValueError('expected matching prediction/expert [H,20] chunks')
    rate = case_fps(case, arrays, fps)
    reference = validate_reference(dict(action20=target, valid_mask=mask, fps=rate,
        source=dict(case=str(case), chunks_sha256=sha(source))), len(target))
    extra = {}
    if relative64:
        if not all(k in arrays for k in ('pred_relative64', 'gt_relative64')):
            raise ValueError('both relative64 arrays are required for --relative64')
        extra = dict(pred_relative64=arrays['pred_relative64'], gt_relative64=arrays['gt_relative64'])
    report = compare_actions(pred, reference, **extra)
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'source.json').write_text(json.dumps(dict(case=str(case), chunks_sha256=sha(source),
        fps=rate, alignment=report['alignment'], horizon=len(target), valid_steps=int(mask.sum())), indent=2) + '\n')
    shutil.copy2(source, output / 'chunks.npz')
    for field, names, p, g in [('action20', ACTION20_NAMES, pred, target),
            *([('relative64', RELATIVE64_NAMES, arrays['pred_relative64'], arrays['gt_relative64'])] if relative64 else [])]:
        with (output / (field + '_comparison.csv')).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'time_s', 'valid', *[name + '_' + kind for name in names for kind in ('pred', 'expert', 'error')]])
            for step in range(len(pred)):
                row = [step, step / rate, int(mask[step])]
                for dim in range(len(names)):
                    row.extend((float(p[step, dim]), float(g[step, dim]), float(p[step, dim] - g[step, dim])))
                writer.writerow(row)
    with (output / 'per_dimension_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(report['per_dimension'][0]))
        writer.writeheader(); writer.writerows(report['per_dimension'])
    write_comparison_plots(output, pred, reference, report, **extra)
    page = '<!doctype html><meta charset="utf-8"><title>Expert / prediction trajectories</title>'
    page += '<style>body{font-family:sans-serif;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}td{vertical-align:top;padding:10px}</style>'
    page += f'<h1>{html.escape(case.name)}</h1><p>{int(mask.sum())}/{len(mask)} valid steps; {rate:g} Hz. Blue: expert; orange: prediction. Padding is excluded from metrics.</p>'
    page += '<table><tr>'
    for name in ('target.py', 'prediction.py'):
        text = (case / name).read_text() if (case / name).is_file() else '(not present)'
        (output / name).write_text(text)
        page += '<td><b>' + name + '</b><pre>' + html.escape(text) + '</pre></td>'
    page += '</tr></table>'
    sheet = case / 'observation_model_input/contact_sheet.png'
    if sheet.is_file():
        shutil.copy2(sheet, output / 'observation.png')
        page += '<h2>Conditioning observation</h2><img src="observation.png">'
    for name in ('action20_all_dimensions.png', 'physical_errors.png', 'relative64_all_dimensions.png'):
        if (output / name).is_file(): page += f'<h2>{name}</h2><img src="{name}">'
    page += '<h2>Individual dimensions</h2>'
    for path in sorted((output / 'dimensions20').glob('*.png')):
        page += f'<p><a href="dimensions20/{path.name}">{path.name}</a></p>'
    (output / 'index.html').write_text(page)
    return dict(valid_steps=int(mask.sum()), horizon=len(mask), fps=rate, physical=report['physical'])


def overview(output, summary, plot_failures):
    groups = defaultdict(list)
    for item in summary['results']:
        selection = item.get('case_selection', {})
        groups[(selection.get('task_name', ''), selection.get('stage_index', ''), selection.get('stage_name', ''))].append(item)
    rows = []
    for (task, stage, name), items in groups.items():
        row = dict(task=task, stage=stage, target_calls=name, cases=len(items),
            observation_windows=len({(r['episode'], r['frame']) for r in items}))
        for metric in ('syntax_valid', 'api_valid', 'primitive_sequence_match', 'ast_exact_match'):
            row[metric] = sum(r['code_metrics'][metric] is True for r in items) / len(items)
        rows.append(row)
    with (output / 'stage_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    page = '<!doctype html><meta charset="utf-8"><title>Custom34 stage evaluation</title>'
    page += '<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}pre{white-space:pre-wrap}</style>'
    page += '<h1>Custom34 stage / trajectory tests</h1><p>Offline expert-observation predictions. Stage labels come from the code cache, not visual verification. Primitive matching compares API names and sides; inspect arguments separately.</p>'
    page += '<table><tr><th>Case / plots</th><th>Task / stage</th><th>Valid steps</th><th>API valid</th><th>API/side sequence match</th><th>AST exact</th><th>Left RMSE mm</th><th>Right RMSE mm</th></tr>'
    failed = {r['folder'] for r in plot_failures}
    for r in summary['results']:
        folder = r['folder']; selected = r.get('case_selection', {})
        label = html.escape(selected.get('task_name', '') + ' / stage ' + str(selected.get('stage_index', ''))
                            + ': ' + selected.get('stage_name', ''))
        link = html.escape(folder) if folder in failed else f'<a href="plots/{html.escape(folder)}/index.html">{html.escape(folder)}</a>'
        page += f'<tr><td>{link}</td><td>{label}</td><td>{r["valid_action_steps"]}/32</td><td>{r["code_metrics"]["api_valid"]}</td><td>{r["code_metrics"]["primitive_sequence_match"]}</td><td>{r["code_metrics"]["ast_exact_match"]}</td><td>{r["physical"]["left"]["position_error_mm"]["rmse"]:.2f}</td><td>{r["physical"]["right"]["position_error_mm"]["rmse"]:.2f}</td></tr>'
    page += '</table><h2>Failures / artifact warnings</h2><pre>' + html.escape(json.dumps(dict(
        inference_failures=summary['failures'], plot_failures=plot_failures,
        image_warning_cases=summary['artifact_warning_cases']), ensure_ascii=False, indent=2)) + '</pre>'
    (output / 'index.html').write_text(page)


def run(args):
    output = args.out.resolve()
    if output.exists(): raise FileExistsError(f'choose a new output directory: {output}')
    module = evaluator()
    plan = json.loads(args.cases_file.read_text()) if args.cases_file else make_plan(args)
    output.mkdir(parents=True)
    plan_path = output / 'cases.json'
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    status = module.run(argparse.Namespace(train_config=args.train_config, checkpoint=args.checkpoint,
        stats=args.stats, device=args.device, episodes=[], start_frames=[], cases_file=plan_path,
        inference_steps=[args.inference_steps], canvas_tokens=[args.canvas_tokens],
        seeds=args.seeds, padding=args.padding, out=output / 'evaluation', plots=False))
    summary = json.loads((output / 'evaluation/summary.json').read_text())
    failures = []
    for item in summary['results']:
        folder = item['folder']
        if Path(folder).name != folder: raise ValueError('invalid case folder')
        try:
            render_case(output / 'evaluation' / folder, output / 'plots' / folder, relative64=args.relative64)
        except Exception as exc:
            failures.append(dict(folder=folder, error=f'{type(exc).__name__}: {exc}'))
    overview(output, summary, failures)
    (output / 'plot_failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2) + '\n')
    print('[trajectory-test] index:', output / 'index.html', flush=True)
    return status if status else (2 if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for mode in ('plan', 'run'):
        p = commands.add_parser(mode)
        p.add_argument('--train-config', type=Path, required=True)
        p.add_argument('--stats', type=Path)
        p.add_argument('--episodes', type=int, nargs='+')
        p.add_argument('--frames', type=int, nargs='+', help='Explicit frames in each selected episode; otherwise select each code stage')
        p.add_argument('--points', nargs='+', choices=('start', 'middle', 'end'), default=['middle'])
        p.add_argument('--min-valid-steps', type=int, default=1)
        p.add_argument('--out', type=Path, required=True)
        if mode == 'run':
            p.add_argument('--cases-file', type=Path)
            p.add_argument('--checkpoint', type=Path, required=True)
            p.add_argument('--inference-steps', type=int, default=10)
            p.add_argument('--canvas-tokens', type=int, default=128)
            p.add_argument('--seeds', type=int, nargs='+', default=[42])
            p.add_argument('--device', default='cuda:0')
            p.add_argument('--padding', choices=('repeat', 'error'), default='repeat')
            p.add_argument('--relative64', action='store_true')
    p = commands.add_parser('plot')
    p.add_argument('--case-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fps', type=float)
    p.add_argument('--relative64', action='store_true')
    args = parser.parse_args()
    if args.command == 'plot':
        report = render_case(args.case_dir, args.out, fps=args.fps, relative64=args.relative64)
        print(json.dumps(report, indent=2)); return 0
    if not getattr(args, 'cases_file', None) and not args.episodes:
        parser.error('--episodes is required unless --cases-file is provided')
    if args.frames is not None and len(set(args.frames)) != len(args.frames):
        parser.error('--frames must be distinct')
    if args.command == 'plan':
        if args.out.exists(): raise FileExistsError(args.out)
        plan = make_plan(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        print(f'[trajectory-test] {len(plan["cases"])} cases saved: {args.out}'); return 0
    if args.inference_steps < 1 or args.canvas_tokens < 1 or any(s < 0 for s in args.seeds):
        parser.error('inference steps/canvas must be positive; seeds nonnegative')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
