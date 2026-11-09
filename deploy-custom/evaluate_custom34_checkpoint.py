#!/usr/bin/env python3
"""Re-evaluate a custom34 checkpoint through the original training data pipeline.

No trainer, optimizer, DeepSpeed, HTTP server, robot, or VAE pixel decoder is used.
Generated Python is inspected with AST only and is never executed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
TOOL_VERSION = '1.2'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_fingerprint(value):
    value = value.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def dtype_profile(model):
    return dict(parameters=dict(Counter(str(p.dtype) for p in model.parameters())),
                buffers=dict(Counter(str(b.dtype) for b in model.buffers())))


def prepare_inference_model(model, device, dtype):
    """Builders leave some modules FP32; training's DeepSpeed normally casts them.

    Module.bfloat16/half/float cast floating tensors while preserving complex RoPE
    and integer buffers. Module.to(dtype=...) would also cast complex buffers.
    """
    before = dtype_profile(model)
    model.to(device=device)
    method = {torch.bfloat16: 'bfloat16', torch.float16: 'half', torch.float32: 'float'}[dtype]
    getattr(model, method)()
    model.eval()
    if any(p.is_floating_point() and p.dtype != dtype for p in model.parameters()):
        raise RuntimeError('model still has floating parameters with inconsistent inference precision')
    return dict(before=before, after=dtype_profile(model),
                policy='cast floating tensors; preserve complex and integer tensors')


def ast_tree(code):
    try:
        tree = ast.parse(code.strip())
        return ast.dump(tree, include_attributes=False) if tree.body else None
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


def robot_calls(code):
    """Validate literal API arguments, including the three required RPY numbers."""
    signatures = dict(move_arm=['side', 'target_pos', 'target_rpy'],
                      grasp=['object_name', 'side'], pick=['object_name', 'side'],
                      place=['object_name', 'target_name', 'side'],
                      rotate=['object_name', 'angle_deg', 'side'], release=['side'])
    tree = ast.parse(code.strip())
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('expected one policy_chunk function')
    fn = tree.body[0]
    if (fn.name != 'policy_chunk' or fn.decorator_list or fn.args.vararg or fn.args.kwarg
            or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.defaults
            or [a.arg for a in fn.args.args] != ['obs', 'state', 'robot', 'skills']):
        raise ValueError('wrong policy_chunk signature')
    calls = []
    for node in fn.body:
        if not isinstance(node, (ast.Expr, ast.Return)) or not isinstance(node.value, ast.Call):
            raise ValueError('expected straight-line robot API calls')
        c = node.value
        if not (isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name)
                and c.func.value.id == 'robot' and c.func.attr in signatures):
            raise ValueError('unknown robot API')
        name, params = c.func.attr, signatures[c.func.attr]
        if len(c.args) > len(params):
            raise ValueError('too many arguments')
        values = dict(zip(params, [ast.literal_eval(v) for v in c.args]))
        for kw in c.keywords:
            if kw.arg not in params or kw.arg in values:
                raise ValueError('unknown or repeated argument')
            values[kw.arg] = ast.literal_eval(kw.value)
        if set(values) != set(params) or values['side'] not in ('left', 'right'):
            raise ValueError('missing argument or invalid side')
        if name == 'move_arm':
            for key in ('target_pos', 'target_rpy'):
                x = values[key]
                if (not isinstance(x, (tuple, list)) or len(x) != 3
                        or any(not isinstance(n, (int, float)) or not np.isfinite(n) for n in x)):
                    raise ValueError(f'{key} must contain three finite numbers')
        if name == 'rotate' and (not isinstance(values['angle_deg'], (int, float))
                                 or not np.isfinite(values['angle_deg'])):
            raise ValueError('angle_deg must be finite')
        for key in ('object_name', 'target_name'):
            if key in values and (not isinstance(values[key], str) or not values[key]):
                raise ValueError(f'{key} must be a nonempty string')
        calls.append((name, values['side']))
    return calls


def code_metrics(prediction, target, tokenizer, canvas):
    p, t = prediction.strip(), target.strip()
    pi, ti = tokenizer.encode(p, add_special_tokens=False), tokenizer.encode(t, add_special_tokens=False)
    pt, tt = ast_tree(p), ast_tree(t)
    pc, tc, problem = None, None, None
    try:
        pc = robot_calls(p)
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        problem = str(exc)
    try:
        tc = robot_calls(t)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        pass
    slots = max(len(pi), len(ti))
    return dict(labeled=bool(t), exact_match=pi == ti if t else None,
                ast_exact_match=(tt is not None and pt == tt) if t else None,
                syntax_valid=pt is not None, api_valid=pc is not None, api_error=problem,
                primitive_sequence_match=(tc is not None and pc == tc) if t else None,
                target_calls=tc, predicted_calls=pc, target_tokens=len(ti),
                predicted_tokens=len(pi), canvas_tokens=canvas, target_over_canvas=len(ti) > canvas,
                token_matches=sum(a == b for a, b in zip(pi, ti)), token_slots=slots,
                token_accuracy=sum(a == b for a, b in zip(pi, ti)) / slots if slots else None)


def observation_frames(video, cameras):
    if cameras < 1 or video.ndim != 5 or video.shape[1] != 3 or video.shape[2] % cameras:
        raise ValueError('expected [B,3,cameras*frames,H,W]')
    count = video.shape[2] // cameras
    if count < 1:
        raise ValueError('empty video')
    return torch.cat([video[:, :, v*count:v*count+1] for v in range(cameras)], dim=2), count


def infer_once(model, batch, *, steps, canvas, seed):
    """Same inference entrypoint as training's code probe; isolate RNG/schedulers."""
    obs, frames = observation_frames(batch['video'], model.n_cameras)
    prompt = batch.get('prompt_code', batch['prompt'])[0]
    device = torch.device(model.device)
    schedulers = [(s, deepcopy(s.__dict__)) for s in (model.video_scheduler, model.action_scheduler)]
    previous = os.environ.get('AGENTICWAM_INFER_CODE_TOKENS')
    os.environ['AGENTICWAM_INFER_CODE_TOKENS'] = str(canvas)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            output = model.infer_action_trimodal(
                obs_video=obs.to(device), proprio=batch['proprio'][:, :1].to(device),
                prompt_embs=batch['prompt_embs'].to(device), instruction=prompt,
                action_horizon=batch['action'].shape[1], num_video_frames=frames,
                num_inference_steps=steps, max_code_tokens=canvas,
                code_cache=None, return_video=False)
        prediction = output[0].detach().float().cpu()
        if prediction.shape != batch['action'].shape[1:] or not torch.isfinite(prediction).all():
            raise ValueError('invalid predicted normalized action chunk')
        if not isinstance(output[1], str):
            raise TypeError('inference did not return generated Python text')
        return prediction, output[1], obs, prompt
    finally:
        for scheduler, state in schedulers:
            scheduler.__dict__.clear()
            scheduler.__dict__.update(state)
        if previous is None:
            os.environ.pop('AGENTICWAM_INFER_CODE_TOKENS', None)
        else:
            os.environ['AGENTICWAM_INFER_CODE_TOKENS'] = previous


class IndexRecorder:
    def __init__(self, base):
        self.base, self.last = base, None

    def __len__(self):
        return len(self.base)

    def __getattr__(self, key):
        return getattr(self.base, key)

    def __getitem__(self, index):
        value = self.base[index]
        self.last = int(value['idx'])
        return value


def checked_sample(dataset, recorder, index):
    recorder.last = None
    value = dataset[index]
    if recorder.last != index:
        raise RuntimeError(f'loader replaced requested frame {index} with {recorder.last}; refusing to score it')
    return value


def absolute_prediction(normalized, state, transform, normalizer, key):
    relative = normalizer.backward(normalized)
    batch = {'state': {key: transform._state64(state)[None, None]}, 'action': {key: relative[None]}}
    absolute = transform.backward_action20(batch)['action'][key][0]
    return relative, absolute.numpy()


def action_reference(actions, mask, fps, source):
    from agenticwam.utils.custom34_review import validate_reference
    return validate_reference(dict(action20=actions, valid_mask=mask, fps=fps, source=source), len(actions))


def run(args):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate
    from agenticwam.datasets.lerobot.transforms.custom_unified_eef import CustomUnifiedEEFTransform
    from agenticwam.utils.custom34_review import compare_actions, save_video_frames, write_comparison_plots

    if os.environ.get('AGENTICWAM_ORACLE_CODE_FILE'):
        raise ValueError('unset AGENTICWAM_ORACLE_CODE_FILE; this is a free-generation evaluation')
    cfg = OmegaConf.load(args.train_config)
    if cfg.model.get('code_memory', False) or cfg.data.train.get('code_slices', False):
        raise ValueError('this tool targets the ordinary custom34 recipe without code history')
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f'choose a new output directory: {out}')
    stats = args.stats or args.train_config.with_name('dataset_stats.json')
    if not stats.is_file():
        stats = Path(str(cfg.data.train.pretrained_norm_stats)) if args.stats is None else stats
    if not stats.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError('checkpoint/statistics not found')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select an available GPU for real checkpoint inference')
    cfg.data.train.pretrained_norm_stats = str(stats.resolve())
    cfg.data.train.save_norm_stats = False
    print('[checkpoint-eval] loading original training dataset; no statistics are written', flush=True)
    ds = instantiate(cfg.data.train)
    base = ds.lerobot_dataset
    sources = base.multi_dataset._datasets
    if len(sources) != 1:
        raise ValueError('expected one custom v3 dataset root')
    single = sources[0]
    reader = single.meta.reader
    processor = base.processor
    custom = [t for t in processor.action_state_transforms if isinstance(t, CustomUnifiedEEFTransform)]
    if len(custom) != 1:
        raise ValueError('expected exactly one CustomUnifiedEEFTransform')
    transform = custom[0]
    key = str(processor.shape_meta['action'][0]['key'])
    recorder = IndexRecorder(base)
    ds.lerobot_dataset = recorder
    cameras = [str(m['key']) for m in processor.shape_meta['images']]
    horizon = int(cfg.data.train.num_frames)-1
    stride = int(cfg.data.train.global_sample_stride)
    plans = []
    for episode in args.episodes:
        slot = single.episodes.index(episode)
        for start in args.start_frames:
            length = reader.episode_length(episode)
            if start < 0 or start >= length:
                raise ValueError(f'episode {episode}: start frame {start} is out of range')
            if start+(horizon-1)*stride >= length and args.padding != 'repeat':
                raise ValueError(f'episode {episode}, frame {start}: choose an earlier start or --padding repeat')
            plans.append((episode, start, int(single._starts[slot])+start))
    precision = str(cfg.get('mixed_precision', 'bf16')).lower()
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32, 'no': torch.float32}[precision]
    cfg.model.include_text_encoder = False
    cfg.model.load_pretrain = False
    print('[checkpoint-eval] building checkpoint model; one GPU, no optimizer/DeepSpeed', flush=True)
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    print('[checkpoint-eval] loading saved weights', flush=True)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    sd = payload.get('model', payload)
    step = payload.get('step')
    required = set(model._trainable_state_dict())
    absent = sorted(required-set(sd))
    if absent:
        raise ValueError(f'checkpoint lacks {len(absent)} required tensors; examples: {absent[:8]}')
    loaded = model.load_state_dict(sd, strict=False)
    if loaded.unexpected_keys:
        raise ValueError(f'checkpoint/config mismatch; unexpected keys: {loaded.unexpected_keys[:8]}')
    del sd, payload
    precision_profile = prepare_inference_model(model, args.device, dtype)
    print('[checkpoint-eval] model precision:', json.dumps(precision_profile['after']), flush=True)
    canvases = args.canvas_tokens or [model._resolve_infer_code_tokens()]
    out.mkdir(parents=True)
    provenance = dict(evaluation='offline checkpoint generation through training loader; not historical training logs',
                      tool_version=TOOL_VERSION, model_precision_profile=precision_profile,
                      checkpoint=str(args.checkpoint.resolve()), step=step, config_sha256=sha256(args.train_config),
                      stats_sha256=sha256(stats), precision=precision, device=args.device,
                      required_checkpoint_tensors=len(required), inference_steps=args.inference_steps,
                      canvas_tokens=canvases, seeds=args.seeds, video_pixel_decode=False)
    for field, path in [('code_cache_sha256', cfg.data.train.get('policy_code_idx_path')),
                        ('skill_prompt_sha256', cfg.data.train.get('skill_prompt_map'))]:
        if path:
            provenance[field] = sha256(str(path))
    (out/'run.json').write_text(json.dumps(provenance, indent=2)+'\n')
    results, failures = [], []
    for episode, start, index in plans:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        sample = checked_sample(ds, recorder, index)
        batch = default_collate([sample])
        raw = reader.read_episode(episode, columns=['observation.state', 'action', 'task_index'])
        state = torch.as_tensor(np.asarray(raw.iloc[start]['observation.state'], dtype=np.float32))
        rows = start+np.arange(horizon)*stride
        valid = rows < len(raw)
        gt20 = np.stack(raw.iloc[np.minimum(rows, len(raw)-1)]['action']).astype(np.float32)
        if not np.array_equal(batch['action_is_pad'][0].numpy(), ~valid):
            raise ValueError('training loader and reference disagree about padding')
        gt64 = transform._action64(torch.from_numpy(gt20)[None], state[None])[0]
        expected_state = processor.normalizer.normalizers['state'][key].forward(transform._state64(state))
        expected_action = processor.normalizer.normalizers['action'][key].forward(gt64)
        state_error = float((expected_state-batch['proprio'][0, 0]).abs().max())
        action_error = float((expected_action-batch['action'][0]).abs().max())
        if max(state_error, action_error) > 1e-5:
            raise ValueError(f'training loader/raw-window mismatch: state={state_error}, action={action_error}')
        target = batch['policy_code'][0]
        for canvas in canvases:
            for steps in args.inference_steps:
                for seed in args.seeds:
                    folder = f'episode{episode:06d}_frame{start:06d}_c{canvas}_n{steps}_s{seed}'
                    case = out/folder
                    try:
                        started = time.perf_counter()
                        normalized, code, obs, prompt = infer_once(model, batch, steps=steps, canvas=canvas, seed=seed)
                        relative, pred20 = absolute_prediction(normalized, state, transform,
                                                               processor.normalizer.normalizers['action'][key], key)
                        reference = action_reference(gt20, valid, reader.fps/stride,
                                                     dict(root=str(reader.root), episode=episode, start_frame=start))
                        cm = code_metrics(code, target, model.code_expert.tokenizer, canvas)
                        prefix = '' if os.environ.get('AGENTICWAM_CODE_NO_INSTRUCTION') == '1' else (prompt if prompt.startswith('# Instruction:') else f'# Instruction: {prompt}\n')
                        plen = len(model.code_expert.tokenizer.encode(prefix, add_special_tokens=False))
                        cm.update(prompt_tokens=plen, prompt_plus_target_eos_tokens=plen+cm['target_tokens']+1,
                                  training_token_cap=int(model.code_expert.max_code_tokens),
                                  prompt_target_exceeds_training_cap=plen+cm['target_tokens']+1>model.code_expert.max_code_tokens)
                        case.mkdir()
                        np.savez_compressed(case/'chunks.npz', state34=state.numpy(), pred_action20=pred20,
                                            pred_relative64=relative.numpy(), gt_action20=gt20,
                                            gt_relative64=gt64.numpy(), valid_mask=valid)
                        (case/'prediction.py').write_text(code+'\n')
                        (case/'target.py').write_text(target)
                        (case/'prompt.txt').write_text(prefix)
                        # Keep the actual predictions even if a later metric/image export fails.
                        metrics = compare_actions(pred20, reference, pred_relative64=relative.numpy(), gt_relative64=gt64.numpy())
                        (case/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
                        (case/'code_metrics.json').write_text(json.dumps(cm, indent=2, allow_nan=False)+'\n')
                        artifact_warnings = []
                        try:
                            images = save_video_frames(obs, case/'observation_model_input', cameras, step_stride=0)
                            if images['status'] != 'saved':
                                artifact_warnings.append(dict(component='observation_images', diagnostic=images))
                        except Exception as exc:
                            artifact_warnings.append(dict(component='observation_images', error=f'{type(exc).__name__}: {exc}'))
                            (case/'observation_export_error.log').write_text(traceback.format_exc())
                        if args.plots:
                            try:
                                write_comparison_plots(case, pred20, reference, metrics, pred_relative64=relative.numpy(), gt_relative64=gt64.numpy())
                            except Exception as exc:
                                artifact_warnings.append(dict(component='comparison_plots', error=f'{type(exc).__name__}: {exc}'))
                                (case/'plot_export_error.log').write_text(traceback.format_exc())
                        item = dict(episode=episode, frame=start, global_index=index, seed=seed, inference_steps=steps,
                                    folder=folder, valid_action_steps=int(valid.sum()), code_metrics=cm,
                                    physical=metrics['physical'], target=target, prediction=code,
                                    normalized_state_matches_raw_max_error=state_error,
                                    normalized_action_matches_raw_max_error=action_error,
                                    input_fingerprints=dict(observation=tensor_fingerprint(obs), proprio=tensor_fingerprint(batch['proprio'][:, :1]),
                                                            prompt_embs=tensor_fingerprint(batch['prompt_embs'])),
                                    artifact_warnings=artifact_warnings, seconds=time.perf_counter()-started)
                        results.append(item)
                        with (out/'code_eval.jsonl').open('a') as f:
                            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
                        print(f'[checkpoint-eval] {folder}: API={cm["api_valid"]} stage={cm["primitive_sequence_match"]} left_pos_rmse={metrics["physical"]["left"]["position_error_mm"]["rmse"]:.2f}mm target_tokens={cm["target_tokens"]}/{canvas}', flush=True)
                    except Exception as exc:
                        errors = out/'errors'
                        errors.mkdir(exist_ok=True)
                        trace_path = errors/f'{folder}.log'
                        trace_path.write_text(traceback.format_exc())
                        failures.append(dict(folder=folder, error=f'{type(exc).__name__}: {exc}',
                                             traceback_file=str(trace_path.relative_to(out))))
                        print(f'[checkpoint-eval] FAILED {folder}: {exc}', file=sys.stderr, flush=True)
    warning_cases = sum(bool(r['artifact_warnings']) for r in results)
    summary = dict(provenance=provenance, completed=len(results), failures=failures,
                   artifact_warning_cases=warning_cases, results=results)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print('[checkpoint-eval] summary:', out/'summary.json', flush=True)
    return 1 if failures else (2 if warning_cases else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--stats', type=Path)
    p.add_argument('--episodes', type=int, nargs='+', default=[3])
    p.add_argument('--start-frames', type=int, nargs='+', default=[5])
    p.add_argument('--inference-steps', type=int, nargs='+', default=[10])
    p.add_argument('--canvas-tokens', type=int, nargs='+')
    p.add_argument('--padding', choices=('error', 'repeat'), default='error')
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plots', action='store_true')
    args = p.parse_args()
    if any(v<1 for v in args.inference_steps+(args.canvas_tokens or [])) or any(s<0 for s in args.seeds):
        p.error('steps/canvas must be positive; seeds must be nonnegative')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
