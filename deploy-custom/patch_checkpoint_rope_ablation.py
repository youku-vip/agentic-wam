#!/usr/bin/env python3
"""Add an opt-in, process-local code-RoPE ablation to an installed evaluator.

The original mode does not change computation. The complex mode is guarded by
an exact match with the quantized cosine cache at the supplied theta.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time


BLOCK = '''    # Opt-in code RoPE experiment; no checkpoint or training files are written.
    from agenticwam.models.experts.wan_video_dit import precompute_freqs_cis
    rope_mode = os.environ.get("AGENTICWAM_EVAL_CODE_ROPE", "original")
    if rope_mode not in ("original", "complex"):
        raise ValueError("AGENTICWAM_EVAL_CODE_ROPE must be original or complex")
    old_rope = model.code_expert.freqs
    rope_theta = float(os.environ.get("AGENTICWAM_EVAL_CODE_ROPE_THETA", "1000000"))
    fresh_rope = precompute_freqs_cis(old_rope.shape[1]*2,
                                    end=old_rope.shape[0], theta=rope_theta).to(old_rope.device)
    cosine_matches = (True if old_rope.is_complex() else
                      torch.equal(old_rope, fresh_rope.real.to(old_rope.dtype)))
    code_rope_info = dict(mode=rope_mode, theta=rope_theta,
                          before_dtype=str(old_rope.dtype), before_complex=old_rope.is_complex(),
                          quantized_cosine_matches=cosine_matches, rebuilt=False)
    if rope_mode == "complex" and not old_rope.is_complex():
        if not cosine_matches:
            raise ValueError("RoPE cosine mismatch; verify theta before rebuilding the cache")
        model.code_expert.freqs = fresh_rope
        code_rope_info["rebuilt"] = True
    code_rope_info.update(after_dtype=str(model.code_expert.freqs.dtype),
                          after_complex=model.code_expert.freqs.is_complex())
    print("[checkpoint-eval] code RoPE:", json.dumps(code_rope_info), flush=True)
'''


def patch_source(source: str) -> str:
    if 'code_rope_info = dict(mode=rope_mode' in source:
        return source
    anchor = '    canvases = args.canvas_tokens or [model._resolve_infer_code_tokens()]\n'
    provenance = 'provenance = dict('
    if source.count(anchor) != 1 or source.count(provenance) != 1:
        raise ValueError('unrecognized evaluator layout; file not changed')
    result = source.replace(anchor, BLOCK+anchor, 1)
    result = result.replace(provenance, 'provenance = dict(code_rope_audit=code_rope_info, ', 1)
    compile(result, '<patched checkpoint evaluator>', 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluator', type=Path,
                        nargs='?', default=Path('deploy-custom/evaluate_custom34_checkpoint.py'))
    args = parser.parse_args()
    source = args.evaluator.read_text()
    updated = patch_source(source)
    if updated == source:
        print('RoPE ablation already installed; original mode remains the default')
        return
    backup = args.evaluator.with_name(args.evaluator.name+f'.bak_{time.time_ns()}')
    shutil.copy2(args.evaluator, backup)
    args.evaluator.write_text(updated)
    print('Installed; backup:', backup)
    print('Default: original. Opt in with AGENTICWAM_EVAL_CODE_ROPE=complex.')


if __name__ == '__main__':
    main()
