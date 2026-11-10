#!/usr/bin/env python3
"""Prepare and audit an opt-in custom34 code-repair pilot; never start training."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

CONTROLLED_ENV = (
    'AGENTICWAM_CODE_ROPE_MODE', 'AGENTICWAM_CODE_CANVAS',
    'AGENTICWAM_CODE_CANVAS_TAIL', 'AGENTICWAM_CODE_MASK_TMIN',
    'AGENTICWAM_CODE_MASK_FULLP', 'AGENTICWAM_PAD_EOS_WINDOW',
    'AGENTICWAM_PAD_EOS_WEIGHT', 'AGENTICWAM_INFER_CODE_TOKENS',
    'AGENTICWAM_CODE_NO_INSTRUCTION', 'AGENTICWAM_CODE_PREAMBLE_FILE',
    'AGENTICWAM_CODE_NO_HISTORY', 'AGENTICWAM_ORACLE_CODE_FILE',
    'AGENTICWAM_REMASK_ROUNDS', 'AGENTICWAM_DECODE_TEMP',
    'AGENTICWAM_ORDER_TEMP', 'AGENTICWAM_REVEAL_CAP',
    'AGENTICWAM_REVEAL_TAU', 'AGENTICWAM_MIN_CODE_LEN',
)
OPTIONS = dict(canvas_tail='masked', mask_tmin=0.5, mask_fullp=0.25,
               pad_eos_window=8, pad_eos_weight=1.0)


@contextmanager
def controlled_environment():
    previous = {k: os.environ.get(k) for k in CONTROLLED_ENV}
    try:
        for k in CONTROLLED_ENV:
            os.environ.pop(k, None)
        yield
    finally:
        for k, value in previous.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(tokenizer, text):
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def load_evaluator():
    path = ROOT/'deploy-custom/evaluate_custom34_checkpoint.py'
    spec = importlib.util.spec_from_file_location('custom_repair_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_prompts(cfg):
    train = cfg.data.train
    if str(train.get('code_instruction_mode', 'shared')) != 'shared':
        raise ValueError('this custom34 recipe requires shared code instruction mode')
    for key in ('instruction_idx_path', 'episode_instruction_map', 'override_instruction'):
        if train.get(key) is not None:
            raise ValueError(f'custom instruction override requires a separate prompt audit: {key}')
    path = Path(str(train.skill_prompt_map))
    prompts = json.loads(path.read_text())
    if len(prompts) != 4:
        raise ValueError('expected the four-task skill prompt map')
    tree = ast.parse((ROOT/'src/agenticwam/datasets/lerobot/robot_video_dataset.py').read_text())
    default = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_PROMPT' for t in n.targets))
    result = {}
    for task, preamble in prompts.items():
        text = preamble if train.get('skill_prompt_only', False) else preamble + default.format(task=task)
        result[task] = text if text.startswith('# Instruction:') else f'# Instruction: {text}\n'
    return result, path


def inventory(cfg, tokenizer):
    path = Path(str(cfg.data.train.policy_code_idx_path))
    payload = torch.load(path, map_location='cpu', weights_only=True)
    codes, index = payload['codes'], payload['code_id_by_idx']
    if not codes or index.ndim != 1 or not len(index) or int(index.min()) < 0 or int(index.max()) >= len(codes):
        raise ValueError('invalid flat code cache')
    evaluator = load_evaluator()
    bad = []
    lengths = []
    for code_id, code in enumerate(codes):
        if not isinstance(code, str):
            raise ValueError(f'non-string code at index {code_id}')
        lengths.append(len(encode(tokenizer, code)))
        if code.strip():
            try:
                evaluator.robot_calls(code)
            except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
                bad.append(dict(code_id=code_id, error=str(exc)))
    if bad:
        raise ValueError(f'invalid target robot APIs: {bad[:10]}')
    if not any(c.strip() for c in codes):
        raise ValueError('code cache contains no labeled programs')
    prompts, prompt_path = training_prompts(cfg)
    prompt_lengths = {task: len(encode(tokenizer, text)) for task, text in prompts.items()}
    return codes, lengths, prompts, dict(
        cache=str(path.resolve()), cache_sha256=sha256(path), frames=len(index),
        programs=len(codes), nonempty_programs=sum(bool(c.strip()) for c in codes),
        unlabeled_frames=int(sum(int((index == i).sum()) for i, c in enumerate(codes) if not c.strip())),
        maximum_target_tokens=max(lengths),
        target_token_quantiles={str(q): float(np.quantile(lengths, q)) for q in (.5, .95, .99)},
        prompt_tokens=prompt_lengths, prompt_map_sha256=sha256(prompt_path),
        maximum_prompt_tokens=max(prompt_lengths.values()),
        tokenizer_name=str(cfg.model.code_model_name),
        mask_token_id=tokenizer.mask_token_id, eos_token_id=tokenizer.eos_token_id,
    )


def audit(cfg, tokenizer=None):
    if str(cfg.model.get('code_rope_mode', '')) != 'complex':
        raise ValueError('pilot config must set model.code_rope_mode=complex')
    if str(cfg.model.code_objective) != 'joint' or cfg.model.get('code_memory', False):
        raise ValueError('pilot targets the joint custom34 recipe without code memory')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(cfg.model.code_model_name), trust_remote_code=False, local_files_only=True)
    if tokenizer.mask_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('tokenizer must expose native MASK and EOS tokens')
    with controlled_environment():
        codes, lengths, prompts, result = inventory(cfg, tokenizer)
        canvas = int(cfg.model.code_canvas)
        if int(cfg.model.infer_code_tokens) != canvas:
            raise ValueError('training and inference canvases must match')
        if result['maximum_target_tokens'] + 1 > canvas:
            raise ValueError('target plus EOS exceeds the canvas')
        if result['maximum_prompt_tokens'] + canvas > int(cfg.model.max_code_tokens):
            raise ValueError('longest prompt plus canvas exceeds max_code_tokens')
        from agenticwam.models.experts.code_expert import CodeExpert
        from agenticwam.models.experts.wan_video_dit import rope_apply
        from inspect import signature
        if 'code_training_options' not in signature(CodeExpert).parameters:
            raise ValueError('replace src/agenticwam/models/experts/code_expert.py first')
        build_tree = ast.parse((ROOT/'src/agenticwam/models/cosmos_mot/build.py').read_text())
        factory = next(n for n in build_tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == str(cfg.model._target_).rsplit('.', 1)[-1])
        arguments = {a.arg for a in factory.args.args + factory.args.kwonlyargs}
        if not {'code_rope_mode', 'code_training_options'} <= arguments:
            raise ValueError('replace src/agenticwam/models/cosmos_mot/build.py first')
        options = OmegaConf.to_container(cfg.model.code_training_options, resolve=True)
        if options.get('canvas_tail') != 'masked':
            raise ValueError('pilot config requires a masked, unsupervised canvas tail')
        tmin, fullp = float(options['mask_tmin']), float(options['mask_fullp'])
        if not math.isfinite(tmin) or not 0 <= tmin < 1 or not math.isfinite(fullp) or not 0 <= fullp <= 1:
            raise ValueError('invalid configured mask probabilities')
        if int(options['pad_eos_window']) < 0 or not math.isfinite(float(options['pad_eos_weight'])) or float(options['pad_eos_weight']) <= 0:
            raise ValueError('invalid configured EOS supervision')
        vocab = max(len(tokenizer), int(tokenizer.mask_token_id)+1, int(tokenizer.eos_token_id)+1)
        expert = CodeExpert(
            blocks=nn.ModuleList([]), embed_tokens=nn.Embedding(vocab, 8),
            final_norm=nn.Identity(), lm_head=nn.Linear(8, vocab, bias=False),
            hidden_dim=8, num_heads=1, attn_head_dim=8, vocab_size=vocab,
            max_code_tokens=int(cfg.model.max_code_tokens), rope_theta=1000000,
            tokenizer=tokenizer, code_objective='joint', mask_token_id=tokenizer.mask_token_id,
            code_canvas=canvas, code_rope_mode='complex', code_training_options=options)
        cache = expert.freqs.clone()
        nn.Sequential(expert).to(dtype=torch.bfloat16)
        if not expert.freqs.is_complex() or not torch.equal(cache, expert.freqs):
            raise ValueError('parent dtype conversion did not preserve complex RoPE')
        vector = torch.ones(1, 8, 8)
        rotated = rope_apply(vector, expert.freqs[:8].view(8, 1, -1), 1)
        torch.testing.assert_close(vector.norm(dim=-1), rotated.norm(dim=-1))
        longest = max(range(len(lengths)), key=lengths.__getitem__)
        task = max(prompts, key=lambda k: len(encode(tokenizer, prompts[k])))
        os.environ['AGENTICWAM_CODE_MASK_FULLP'] = '1'
        try:
            ids, labels = expert.build_code_batch([prompts[task]], [codes[longest]], 'cpu')
        finally:
            os.environ.pop('AGENTICWAM_CODE_MASK_FULLP', None)
        plen = len(encode(tokenizer, prompts[task]))
        if not bool((ids[0, plen:] == tokenizer.mask_token_id).all()):
            raise ValueError('full-mask training input differs from the inference canvas')
        eos_labels = int((labels == tokenizer.eos_token_id).sum())
        if eos_labels > 1 + int(options['pad_eos_window']):
            raise ValueError('EOS labels exceed the bounded supervision window')
        result.update(passed=True, canvas=canvas, max_code_tokens=int(cfg.model.max_code_tokens),
                      code_rope_mode=expert.code_rope_mode, rope_dtype=str(expert.freqs.dtype),
                      full_mask_probe_passed=True, eos_labels_for_longest_program=eos_labels,
                      code_training_options=options, inference_accuracy_measured=False)
    stats = Path(str(cfg.data.train.pretrained_norm_stats))
    result.update(stats=str(stats.resolve()), stats_sha256=sha256(stats))
    return result


def launcher(cfg, out, checkpoint, python, nproc, episodes, frame):
    q = shlex.quote
    recipe = out/'recipe/config.yaml'
    baseline = checkpoint.resolve()
    stats = out/'normalization_reference.json'
    train = shlex.join([
        python, '-m', 'accelerate.commands.launch', '--config_file',
        str(ROOT/'scripts/accelerate_configs/accelerate_zero1_ds.yaml'),
        '--num_processes', str(nproc), str(ROOT/'scripts/train.py'),
        '--config-path', str(recipe.parent), '--config-name', 'config'])
    expected = out/f'checkpoints/weights/step_{int(cfg.max_steps):06d}.pt'
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {q(str(ROOT))}
PY={q(python)}
RUN_DIR={q(str(out))}
RECIPE={q(str(recipe))}
export PYTHONPATH={q(str(ROOT/'src'))}
unset {' '.join(CONTROLLED_ENV)}
if [ -f src/agenticwam/utils/deepspeed_graph_count_cache.py ]; then
  export AGENTICWAM_CACHE_DS_GRAPH_COUNTS=1
fi
export AGENTICWAM_PHASE_TIMING=0
export AGENTICWAM_WAIT_DIAG_STEPS=0
export AGENTICWAM_MOT_NO_CKPT_EVERY=0
export AGENTICWAM_DISABLE_MOT_ATTN_CKPT=0
MODE=audit
if [ "$#" -gt 0 ]; then MODE="$1"; fi
audit_run() {{
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/custom_code_repair.py audit --config "$RECIPE" --report "$RUN_DIR/preflight.latest.json"
}}
review() {{
  "$PY" deploy-custom/evaluate_custom34_checkpoint.py --train-config "$RECIPE" --checkpoint "$1" --stats {q(str(stats))} --episodes {' '.join(map(str,episodes))} --start-frames {frame} --inference-steps 10 --canvas-tokens {int(cfg.model.infer_code_tokens)} --seeds 42 43 44 --device cuda:0 --out "$2"
}}
case "$MODE" in
  audit) audit_run ;;
  baseline)
    audit_run
    review {q(str(baseline))} "$RUN_DIR/eval_baseline_$(date +%Y%m%d_%H%M%S)"
    ;;
  train)
    audit_run
    {train}
    ;;
  eval)
    audit_run
    REVIEW_CHECKPOINT={q(str(expected))}
    if [ "$#" -gt 1 ]; then REVIEW_CHECKPOINT="$2"; fi
    review "$REVIEW_CHECKPOINT" "$RUN_DIR/eval_repair_$(date +%Y%m%d_%H%M%S)"
    ;;
  *) echo "Usage: bash run_code_repair.sh audit|baseline|train|eval [checkpoint]" >&2; exit 2 ;;
esac
"""


def prepare(args, tokenizer=None):
    source = args.source_config.resolve()
    checkpoint = args.checkpoint.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f'choose a new pilot output directory: {out}')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = OmegaConf.load(source)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.code_rope_mode = 'complex'
    cfg.model.code_training_options = OPTIONS
    if str(cfg.model.code_objective) != 'joint':
        raise ValueError('source checkpoint recipe must use joint code training')
    cfg.resume = None
    cfg.foundation_ckpt = str(checkpoint)
    cfg.output_dir = str(out)
    cfg.learning_rate = float(args.lr)
    cfg.max_steps = int(args.steps)
    cfg.save_every = min(100, int(args.steps))
    cfg.code_eval_every = min(50, int(args.steps))
    cfg.code_eval_inference_steps = 10
    cfg.code_eval_max_samples = 0
    cfg.eval_every = 0
    cfg.save_optimizer_state = False
    cfg.keep_last_ckpts = 2
    cfg.task_balanced_batch = True
    cfg.wandb.name = out.name
    cfg.wandb.group = 'custom-code-repair-pilot'
    if int(cfg.batch_size) % 4:
        raise ValueError('rank-local batch_size must be a multiple of four')
    stats = args.stats or source.with_name('dataset_stats.json')
    if args.stats is not None and not stats.is_file():
        raise FileNotFoundError(stats)
    if not stats.is_file():
        stats = Path(str(cfg.data.train.pretrained_norm_stats))
    stats = stats.resolve()
    if not stats.is_file():
        raise FileNotFoundError(stats)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(cfg.model.code_model_name), trust_remote_code=False, local_files_only=True)
    with controlled_environment():
        _, lengths, _, _ = inventory(cfg, tokenizer)
    canvas = max(128, math.ceil((max(lengths)+1)/32)*32) if args.canvas == 'auto' else int(args.canvas)
    cfg.model.code_canvas = canvas
    cfg.model.infer_code_tokens = canvas
    cfg.data.train.pretrained_norm_stats = str(stats)
    cfg.data.train.save_norm_stats = False
    for split in ('val',):
        if cfg.data.get(split) is not None:
            cfg.data[split].pretrained_norm_stats = str(stats)
            cfg.data[split].save_norm_stats = False
    result = audit(cfg, tokenizer)
    out.mkdir(parents=True)
    (out/'recipe').mkdir()
    reference = out/'normalization_reference.json'
    shutil.copyfile(stats, reference)
    for split in ('train', 'val'):
        if cfg.data.get(split) is not None:
            cfg.data[split].pretrained_norm_stats = str(reference)
    config = out/'recipe/config.yaml'
    OmegaConf.save(cfg, config)
    result.update(stats=str(reference), source_config=str(source),
                  source_config_sha256=sha256(source), prepared_config_sha256=sha256(config),
                  initial_checkpoint=str(checkpoint), initial_checkpoint_bytes=checkpoint.stat().st_size,
                  steps=int(args.steps), learning_rate=float(args.lr),
                  optimizer_and_step='new optimizer and schedule starting at zero; foundation weights only',
                  world_size=int(args.nproc), rank_microbatch=int(cfg.batch_size),
                  gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
                  effective_batch=int(cfg.batch_size)*int(args.nproc)*int(cfg.gradient_accumulation_steps),
                  expected_samples_per_task_per_update=int(cfg.batch_size)*int(args.nproc)*int(cfg.gradient_accumulation_steps)//4,
                  thresholds='pilot starting values, not validated optimal hyperparameters',
                  no_gpu_or_training_started=True)
    (out/'code_repair_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (out/'run_code_repair.sh').write_text(launcher(
        cfg, out, checkpoint, args.python, args.nproc, args.review_episodes, args.review_frame))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source-config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--stats', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--canvas', default='auto')
    p.add_argument('--nproc', type=int, default=2)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--review-episodes', type=int, nargs='+', default=[3, 1, 2, 3258])
    p.add_argument('--review-frame', type=int, default=5)
    a = sub.add_parser('audit')
    a.add_argument('--config', type=Path, required=True)
    a.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.steps < 1 or args.nproc < 1 or not math.isfinite(args.lr) or args.lr <= 0:
            parser.error('steps/nproc/lr must be positive')
        result = prepare(args)
    else:
        result = audit(OmegaConf.load(args.config))
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
