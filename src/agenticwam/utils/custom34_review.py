"""Export Custom34 model frames and compare de-normalized, time-aligned action chunks."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


AXES = ('x', 'y', 'z')
POSE_NAMES = [*(f'{a}_m' for a in AXES), *(f'rot6d_c{c}_{a}' for c in (0, 1) for a in AXES)]
ACTION20_NAMES = [*(f'left_{n}' for n in POSE_NAMES), *(f'right_{n}' for n in POSE_NAMES),
                  'left_gripper', 'right_gripper']
RELATIVE64_NAMES = [*(f'left_local_{n}' for n in POSE_NAMES),
                    *(f'left_hand_placeholder_{i}' for i in range(22)), 'left_gripper',
                    *(f'right_local_{n}' for n in POSE_NAMES),
                    *(f'right_hand_placeholder_{i}' for i in range(22)), 'right_gripper']
ACTIVE64 = [*range(9), 31, *range(32, 41), 63]


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def check_actions(value, horizon, dimension=20):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (horizon, dimension) or not np.isfinite(result).all():
        raise ValueError(f'expected finite [{horizon},{dimension}] actions, got {result.shape}')
    return result


def rotation_matrices(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 6 or not np.isfinite(value).all():
        raise ValueError('rotation6D requires six finite values')
    a, b = value[..., :3], value[..., 3:]
    n1, n2 = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if (n1 < 1e-6).any() or (n2 < 1e-6).any():
        raise ValueError('invalid zero rotation6D column')
    a = a / n1[..., None]
    b = b / n2[..., None]
    b = b - (a * b).sum(-1, keepdims=True) * a
    norm = np.linalg.norm(b, axis=-1)
    if (norm < 1e-4).any():
        raise ValueError('invalid parallel rotation6D columns')
    b = b / norm[..., None]
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def validate_reference(reference, horizon):
    if not isinstance(reference, dict):
        raise ValueError('reference must be an object containing action20')
    actions = check_actions(reference.get('action20'), horizon)
    rotation_matrices(actions[:, 3:9])
    rotation_matrices(actions[:, 12:18])
    mask = np.asarray(reference.get('valid_mask', [True] * horizon))
    if mask.shape != (horizon,) or mask.dtype.kind != 'b' or not mask.any():
        raise ValueError('reference.valid_mask must be a nonempty boolean [H] mask')
    fps = reference.get('fps')
    if fps is not None and (isinstance(fps, bool) or not np.isfinite(float(fps)) or float(fps) <= 0):
        raise ValueError('reference.fps must be positive')
    source = reference.get('source', {})
    if not isinstance(source, dict):
        raise ValueError('reference.source must be a metadata object')
    return dict(action20=actions, valid_mask=mask, fps=float(fps) if fps is not None else None, source=source)


def dimensional_metrics(prediction, target, mask, names, active=None):
    difference = prediction[mask] - target[mask]
    active = set(range(len(names))) if active is None else set(active)
    return [dict(dimension=i, name=name, active=i in active,
                 unit='m' if name.endswith('_m') else 'dimensionless',
                 mae=float(np.abs(difference[:, i]).mean()),
                 rmse=float(np.sqrt(np.square(difference[:, i]).mean())),
                 max_abs_error=float(np.abs(difference[:, i]).max()),
                 bias=float(difference[:, i].mean())) for i, name in enumerate(names)]


def physical_errors(prediction, target):
    result = {}
    for side, offset, grip in (('left', 0, 18), ('right', 9, 19)):
        rp = rotation_matrices(prediction[:, offset+3:offset+9])
        rg = rotation_matrices(target[:, offset+3:offset+9])
        # trace(Rg.T @ Rp) == sum(Rg * Rp). Avoid discontinuous Euler-angle subtraction.
        cosine = ((rg * rp).sum((-2, -1)) - 1) / 2
        result[side] = dict(position_error_mm=np.linalg.norm(prediction[:, offset:offset+3] - target[:, offset:offset+3], axis=-1) * 1000,
                            rotation_error_deg=np.rad2deg(np.arccos(np.clip(cosine, -1, 1))),
                            gripper_correct=(prediction[:, grip] > .5) == (target[:, grip] > .5))
    return result


def compare_actions(prediction, reference, *, pred_relative64=None, gt_relative64=None):
    horizon = len(reference['action20'])
    prediction = check_actions(prediction, horizon)
    target, mask = reference['action20'], reference['valid_mask']
    errors = physical_errors(prediction, target)
    groups = {}
    for side, values in errors.items():
        groups[side] = {}
        for key in ('position_error_mm', 'rotation_error_deg'):
            valid = values[key][mask]
            groups[side][key] = dict(mean=float(valid.mean()), rmse=float(np.sqrt(np.square(valid).mean())),
                                      p95=float(np.quantile(valid, .95)), maximum=float(valid.max()))
        groups[side]['gripper_accuracy'] = float(values['gripper_correct'][mask].mean())
    result = dict(horizon=horizon, valid_steps=int(mask.sum()), padded_steps=int((~mask).sum()),
                  alignment='observation.state[t] -> dataset action[t + step * stride]; no action shift',
                  action_space='de-normalized absolute EEF20; grippers remain absolute',
                  per_dimension=dimensional_metrics(prediction, target, mask, ACTION20_NAMES), physical=groups)
    if pred_relative64 is not None or gt_relative64 is not None:
        pred64 = check_actions(pred_relative64, horizon, 64)
        gt64 = check_actions(gt_relative64, horizon, 64)
        result['relative64'] = dict(action_space='de-normalized local delta EEF64, anchored to the SAME observation.state[t]',
                                    active_dimensions=ACTIVE64,
                                    per_dimension=dimensional_metrics(pred64, gt64, mask, RELATIVE64_NAMES, ACTIVE64))
    return result


def save_video_frames(video, output, roles, *, step_stride=1):
    """Split [1,3,nc*T,H,W] into camera-major PNGs without changing the RGB channel order."""
    video = array(video)
    nc = len(roles)
    if (not nc or video.ndim != 5 or video.shape[0:2] != (1, 3) or video.shape[2] % nc
            or video.shape[2] == 0):
        raise ValueError(f'video shape error: expected [1,3,nc*T,H,W] with nc={nc}, got {video.shape}')
    nframes, height, width = video.shape[2] // nc, video.shape[3], video.shape[4]
    finite = np.isfinite(video)
    bad_frames = (~finite[0]).reshape(3, nc, nframes, height, width).any(axis=(0, 3, 4))
    selected = video[finite]
    diagnostic = dict(shape=list(video.shape), dtype=str(video.dtype), elements=int(video.size),
                      nan_count=int(np.isnan(video).sum()), posinf_count=int(np.isposinf(video).sum()),
                      neginf_count=int(np.isneginf(video).sum()), invalid_frame_count=int(bad_frames.sum()),
                      finite_min=float(selected.min()) if selected.size else None,
                      finite_max=float(selected.max()) if selected.size else None)
    views = video[0].reshape(3, nc, nframes, height, width)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'video_diagnostics.json').write_text(json.dumps(diagnostic, indent=2, allow_nan=False)+'\n')
    if bad_frames.any():
        # Preserve the actual tensor for diagnosis; never replace invalid pixels with fake images.
        np.savez_compressed(output / 'nonfinite_video.npz', video=video)
    sheet = Image.new('RGB', (nframes * width, nc * (height + 22)), 'white')
    draw = ImageDraw.Draw(sheet)
    records = []
    for camera, role in enumerate(roles):
        folder = output / role
        folder.mkdir()
        for step in range(nframes):
            name = f'frame_{step:03d}.png'
            invalid = bool(bad_frames[camera, step])
            if invalid:
                left, top = step*width, camera*(height+22)+22
                draw.rectangle((left, top, left+width-1, top+height-1), fill='#ffe5e5', outline='red')
                draw.text((left+3, top+3), 'INVALID NaN/Inf', fill='red')
            else:
                pixels = views[:, camera, step].transpose(1, 2, 0)
                rgb = np.rint((np.clip(pixels, -1, 1)+1)*127.5).astype(np.uint8)
                image = Image.fromarray(rgb)
                image.save(folder / name)
                sheet.paste(image, (step * width, camera * (height+22) + 22))
            draw.text((step*width+3, camera*(height+22)+3), f'{role} +{step*step_stride}', fill='black')
            records.append(dict(camera=role, file=None if invalid else f'{role}/{name}', frame=step,
                                status='nonfinite' if invalid else 'saved',
                                state_step_offset=step*step_stride,
                                kind='conditioned_observation' if step == 0 else 'predicted_future'))
    sheet.save(output / 'contact_sheet.png')
    (output / 'frames.json').write_text(json.dumps(records, indent=2) + '\n')
    return dict(status='partial' if bad_frames.any() else 'saved',
                cameras=list(roles), frames_per_camera=nframes, total_images=nc*nframes-int(bad_frames.sum()),
                frame_slots=nc*nframes, **diagnostic,
                image_size=[height, width], step_stride=step_stride,
                note='The first decoded frame is the conditioned observation reconstruction.')


def write_comparison_plots(output, prediction, reference, report, *, pred_relative64=None, gt_relative64=None):
    # Plot only in review/evaluation; ordinary live /act calls do not need Matplotlib.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target, mask = reference['action20'], reference['valid_mask']
    x = np.arange(len(prediction), dtype=float)
    if reference['fps'] is not None:
        x /= reference['fps']
    xlabel = 'Time from observation (s)' if reference['fps'] is not None else 'Chunk step'

    def draw(ax, p, g, name):
        ax.plot(x, g, color='tab:blue', label='Expert', linewidth=1.4)
        ax.plot(x, p, color='tab:orange', label='Prediction', linewidth=1.3)
        if (~mask).any():
            ax.scatter(x[~mask], g[~mask], marker='x', color='gray', label='Padding (excluded)', s=13)
        ax.set_title(name, fontsize=8)
        ax.grid(alpha=.25)

    fig, axes = plt.subplots(5, 4, figsize=(18, 17), sharex=True)
    for i, ax in enumerate(axes.flat):
        draw(ax, prediction[:, i], target[:, i], f'{i:02d} {ACTION20_NAMES[i]}')
        ax.text(.02, .96, f"MAE={report['per_dimension'][i]['mae']:.4g}", transform=ax.transAxes, va='top', fontsize=7)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]: ax.set_xlabel(xlabel)
    fig.suptitle(f"Absolute EEF20: prediction vs expert ({int(mask.sum())}/{len(mask)} valid steps)")
    fig.tight_layout()
    fig.savefig(output / 'action20_all_dimensions.png', dpi=140)
    plt.close(fig)
    per_dim = output / 'dimensions20'
    per_dim.mkdir(exist_ok=True)
    for i, name in enumerate(ACTION20_NAMES):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        draw(ax, prediction[:, i], target[:, i], f'{i:02d} {name}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(report['per_dimension'][i]['unit'])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(per_dim / f'{i:02d}_{name}.png', dpi=140)
        plt.close(fig)
    errors = physical_errors(prediction, target)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for column, (side, grip) in enumerate((('left', 18), ('right', 19))):
        for row, metric in enumerate(('position_error_mm', 'rotation_error_deg')):
            y = np.where(mask, errors[side][metric], np.nan)
            axes[row, column].plot(x, y)
            axes[row, column].set_title(f'{side}: {metric}')
            axes[row, column].grid(alpha=.25)
        draw(axes[2, column], prediction[:, grip], target[:, grip], f'{side} gripper')
        axes[2, column].axhline(.5, color='gray', linestyle='--', linewidth=.7)
        axes[2, column].set_xlabel(xlabel)
    axes[2, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'physical_errors.png', dpi=140)
    plt.close(fig)
    if pred_relative64 is not None:
        fig, axes = plt.subplots(8, 8, figsize=(26, 22), sharex=True)
        for i, ax in enumerate(axes.flat):
            draw(ax, pred_relative64[:, i], gt_relative64[:, i], f'{i:02d} {RELATIVE64_NAMES[i]}')
            if i not in ACTIVE64: ax.set_facecolor('#eeeeee')
        fig.suptitle('Local relative EEF64 (gray panels: unused hand placeholders)')
        fig.tight_layout()
        fig.savefig(output / 'relative64_all_dimensions.png', dpi=110)
        plt.close(fig)


def write_review(output, *, raw_images, model_input, generated_video, camera_order, state34,
                 prediction20, prediction64, metadata, reference=None, target64=None,
                 video_step_stride=4, make_plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(metadata)
    metadata['review_version'] = 'custom34-chunk-review-v2'
    metadata['warnings'] = []
    values = dict(state34=np.asarray(state34), pred_action20=np.asarray(prediction20),
                  pred_relative64=np.asarray(prediction64))
    if reference is not None:
        report = compare_actions(prediction20, reference, pred_relative64=prediction64, gt_relative64=target64)
        (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        values.update(gt_action20=reference['action20'], valid_mask=reference['valid_mask'], gt_relative64=target64)
        metadata['reference'] = dict(source=reference['source'], fps=reference['fps'], valid_steps=report['valid_steps'])
        with (output / 'per_dimension_metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report['per_dimension'][0]))
            writer.writeheader()
            writer.writerows(report['per_dimension'])
        with (output / 'action20_comparison.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['step', 'valid', *[f'{name}_{kind}' for name in ACTION20_NAMES for kind in ('pred', 'expert', 'error')]])
            for step in range(len(prediction20)):
                row = [step, int(reference['valid_mask'][step])]
                for dimension in range(20):
                    p, g = float(prediction20[step, dimension]), float(reference['action20'][step, dimension])
                    row.extend((p, g, p-g))
                writer.writerow(row)
    np.savez_compressed(output / 'chunks.npz', **values)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    # Core action arrays/metrics are already durable. Pixel export cannot suppress them.
    raw = output / 'observation_raw'
    raw.mkdir()
    for role, image in raw_images.items():
        Image.fromarray(image).save(raw / f'{role}.png')
    for key, value, folder, stride in (
        ('model_input_video', model_input, 'observation_model_input', 0),
        ('generated_video', generated_video, 'generated', video_step_stride),
    ):
        if value is None:
            item = dict(status='missing', message=f'{key} tensor is absent')
        else:
            try:
                item = save_video_frames(value, output/folder, camera_order, step_stride=stride)
            except Exception as exc:
                item = dict(status='error', message=f'{type(exc).__name__}: {exc}')
        metadata[key] = item
        if item['status'] != 'saved':
            metadata['warnings'].append(dict(component=key, **item))
    if reference is not None and make_plots:
        write_comparison_plots(output, prediction20, reference, report,
                               pred_relative64=prediction64, gt_relative64=target64)
    metadata['comparison_ready'] = reference is not None
    metadata['images_valid'] = not metadata['warnings']
    metadata['review_status'] = 'partial' if metadata['warnings'] else 'saved'
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    images = ['generated/contact_sheet.png', 'observation_model_input/contact_sheet.png',
              'action20_all_dimensions.png', 'physical_errors.png', 'relative64_all_dimensions.png']
    page = '<!doctype html><meta charset="utf-8"><title>Custom34 chunk review</title>'
    page += '<style>body{font-family:sans-serif;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
    page += '<h1>Custom34 chunk review</h1><pre>' + html.escape(json.dumps(metadata, indent=2, ensure_ascii=False)) + '</pre>'
    if metadata['warnings']:
        page += '<p style="color:red">Image export contains missing or invalid frames. Action comparisons are separate; see diagnostics.</p>'
    for name in images:
        if (output / name).is_file(): page += f'<h2>{name}</h2><a href="{name}"><img src="{name}"></a>'
    for folder in ('dimensions20',):
        if (output / folder).is_dir():
            page += '<h2>Per-dimension curves</h2>'
            for image in sorted((output / folder).glob('*.png')):
                path = image.relative_to(output).as_posix()
                page += f'<p><a href="{path}">{image.name}</a></p>'
    page += '<p><a href="chunks.npz">Chunk arrays</a> | <a href="metadata.json">Metadata</a></p>'
    if reference is not None:
        page += '<p><a href="metrics.json">Metrics</a> | <a href="action20_comparison.csv">Per-step CSV</a></p>'
    (output / 'index.html').write_text(page)
    return metadata
