"""Counts for generated policy bodies; no generated code is ever executed."""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import os

import torch


COUNT_KEYS = (
    "samples", "unlabeled", "exact_matches", "ast_matches", "syntax_valid",
    "invalid_targets", "token_matches", "token_slots", "targets_over_canvas",
)


def _tree(code):
    try:
        tree = ast.parse(code)
        if not tree.body:
            return None
        compile(tree, "<code-metric>", "exec")
        return ast.dump(tree, include_attributes=False)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


def count_code_pair(prediction, target, tokenizer, canvas_tokens):
    """No special tokens or truncation. Outer whitespace is ignored for both bodies."""
    counts = Counter({key: 0 for key in COUNT_KEYS})
    if target is None or not target.strip():
        counts["unlabeled"] = 1
        return counts
    prediction, target = prediction.strip(), target.strip()
    pred_ids = tokenizer.encode(prediction, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    pred_tree, target_tree = _tree(prediction), _tree(target)
    counts.update(
        samples=1,
        exact_matches=int(pred_ids == target_ids),
        ast_matches=int(target_tree is not None and pred_tree == target_tree),
        syntax_valid=int(pred_tree is not None),
        invalid_targets=int(target_tree is None),
        token_matches=sum(p == t for p, t in zip(pred_ids, target_ids)),
        token_slots=max(len(pred_ids), len(target_ids)),
        targets_over_canvas=int(len(target_ids) > canvas_tokens),
    )
    return counts


def summarize_code_counts(counts):
    """Finalize only AFTER summing counts across ranks; undefined ratios are omitted."""
    result = {key: int(counts[key]) for key in COUNT_KEYS}
    n = counts["samples"]
    if n:
        for name, numerator in (
            ("exact_match", "exact_matches"), ("ast_exact_match", "ast_matches"),
            ("syntax_valid_rate", "syntax_valid"),
            ("target_over_canvas_rate", "targets_over_canvas"),
        ):
            result[name] = counts[numerator] / n
    if counts["token_slots"]:
        result["token_accuracy"] = counts["token_matches"] / counts["token_slots"]
    return result


def probe_code_batch(model, sample, *, inference_steps=10, max_samples=0):
    """Generate from current observations, one sample at a time (inference API is B=1).

    Restores train/eval flags, scheduler state, history and torch RNG even on failure.
    max_samples=0 evaluates every item of this rank's current microbatch.
    """
    if inference_steps <= 0 or max_samples < 0:
        raise ValueError("inference_steps must be positive and max_samples nonnegative")
    if os.environ.get("AGENTICWAM_ORACLE_CODE_FILE"):
        raise ValueError("Code generation metrics forbid AGENTICWAM_ORACLE_CODE_FILE")
    ce = model.code_expert
    if ce is None:
        raise ValueError("Code generation metrics require a code expert")
    video = sample["video"]
    if video.ndim != 5:
        raise ValueError("Expected sample['video'] as [batch, channels, cameras*frames, height, width]")
    batch_size = video.shape[0]
    cameras = int(model.n_cameras)
    if cameras <= 0 or video.shape[2] % cameras:
        raise ValueError("Expected time-stacked camera views in sample['video']")
    frames_per_view = video.shape[2] // cameras
    if frames_per_view < 1:
        raise ValueError("Each camera must have at least one frame")
    targets = sample.get("policy_code")
    if targets is None:
        targets = [""] * batch_size
    if isinstance(targets, str) or len(targets) != batch_size:
        raise ValueError("Expected one policy_code string per batch item")
    instructions = (sample.get("prompt_nohist")
                    if getattr(model, "code_memory_enabled", False) else None)
    if instructions is None:
        instructions = sample.get("prompt_code")
    if instructions is None:
        instructions = sample.get("prompt", [""] * batch_size)
    if isinstance(instructions, str) or len(instructions) != batch_size:
        raise ValueError("Expected one instruction per batch item")
    histories = sample.get("code_history") or [""] * batch_size
    canvas = model._resolve_infer_code_tokens()
    limit = min(batch_size, max_samples) if max_samples else batch_size
    counts = Counter({key: 0 for key in COUNT_KEYS})
    examples = []
    modes = [(module, module.training) for module in model.modules()]
    missing = object()
    transient = {key: getattr(model, key, missing) for key in (
        "_mem_code_history", "_cfg_mode", "_last_code_canvas",
    )}
    schedulers = [(scheduler, deepcopy(scheduler.__dict__)) for scheduler in (
        model.video_scheduler, model.action_scheduler,
    )]
    device = torch.device(model.device)
    rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            model.eval()
            for i in range(limit):
                target = targets[i]
                if target is None or not target.strip():
                    counts["unlabeled"] += 1
                    continue
                obs = torch.cat([
                    video[i:i+1, :, v*frames_per_view:v*frames_per_view+1]
                    for v in range(cameras)
                ], dim=2).to(device)
                proprio = sample.get("proprio")
                if proprio is not None:
                    proprio = proprio[i:i+1, :1].to(device)
                prompt_embs = sample.get("prompt_embs")
                if prompt_embs is None:
                    if model.text_encoder is None:
                        raise ValueError("Missing prompt_embs and text_encoder")
                    prompt_embs = model.text_encoder(sample["text"][i:i+1].to(device))
                else:
                    prompt_embs = prompt_embs[i:i+1].to(device)
                model._mem_code_history = histories[i]
                output = model.infer_action_trimodal(
                    obs_video=obs, proprio=proprio, prompt_embs=prompt_embs,
                    action_horizon=sample["action"].shape[1],
                    num_video_frames=frames_per_view, num_inference_steps=inference_steps,
                    instruction=instructions[i], code_cache=None, return_video=False,
                )
                prediction = output[1]
                if not isinstance(prediction, str):
                    raise TypeError("Inference must return a generated code string")
                counts.update(count_code_pair(prediction, target, ce.tokenizer, canvas))
                examples.append({"batch_index": i, "instruction": instructions[i],
                                 "target": target, "prediction": prediction})
    finally:
        for module, training in modes:
            module.training = training
        for scheduler, state in schedulers:
            scheduler.__dict__.clear()
            scheduler.__dict__.update(state)
        for key, value in transient.items():
            if value is missing:
                if hasattr(model, key):
                    delattr(model, key)
            else:
                setattr(model, key, value)
    return counts, examples
