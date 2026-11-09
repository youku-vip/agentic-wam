#!/usr/bin/env python3
"""Offline expert-code conditioning probe; never a free-generation accuracy test.

Reuse an installed evaluator, including its local precision/reference/RoPE fixes,
without editing it. Expert Python is tokenized and AST-compared, never executed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import types


def build_probe(evaluator: Path, expert_code_file: Path):
    evaluator, expert_code_file = evaluator.resolve(), expert_code_file.resolve()
    expert = expert_code_file.read_text().strip()
    if not expert:
        raise ValueError('expert code file is empty')
    if os.environ.get('AGENTICWAM_ORACLE_CODE_FILE'):
        raise ValueError('unset AGENTICWAM_ORACLE_CODE_FILE; pass --expert-code-file instead')
    if int(os.environ.get('AGENTICWAM_REMASK_ROUNDS', '0')) != 0:
        raise ValueError('expert conditioning requires AGENTICWAM_REMASK_ROUNDS=0')
    source = evaluator.read_text()
    if (os.environ.get('AGENTICWAM_EVAL_CODE_ROPE') == 'complex'
            and 'code_rope_info = dict(mode=rope_mode' not in source):
        raise ValueError('install the RoPE ablation in the evaluator before selecting complex mode')
    anchor = 'provenance = dict('
    if source.count(anchor) != 1:
        raise ValueError('unrecognized evaluator provenance; original file is not changed')
    audit = dict(code_conditioning='provided expert code', code_generation_evaluated=False,
                 expert_code_file=str(expert_code_file),
                 expert_code_sha256=hashlib.sha256(expert_code_file.read_bytes()).hexdigest(),
                 probe='expert-conditioning-v1; compare physical action metrics only')
    source = source.replace(anchor, 'provenance = dict(**_expert_conditioning_audit, ', 1)
    source = source.replace(
        'offline checkpoint generation through training loader; not historical training logs',
        'offline expert-code conditioning probe through training loader; no free-generation accuracy')
    module = types.ModuleType('custom34_expert_conditioning_evaluator')
    module.__file__ = str(evaluator)
    module._expert_conditioning_audit = audit
    exec(compile(source, str(evaluator), 'exec'), module.__dict__)
    if module.ast_tree(expert) is None:
        raise ValueError('expert code has invalid syntax')
    module.robot_calls(expert)
    original_infer, original_metrics = module.infer_once, module.code_metrics

    def conditioned_infer(model, batch, *, steps, canvas, seed):
        if getattr(model.code_expert, 'code_objective', None) != 'joint':
            raise ValueError('this probe requires the joint code objective')
        if batch['policy_code'][0].strip() != expert:
            raise ValueError('supplied expert code does not match this observation target')
        tokens = model.code_expert.tokenizer.encode(expert, add_special_tokens=False)
        if len(tokens) + 1 > canvas:
            raise ValueError('expert code plus EOS does not fit the code canvas')
        previous = os.environ.get('AGENTICWAM_ORACLE_CODE_FILE')
        os.environ['AGENTICWAM_ORACLE_CODE_FILE'] = str(expert_code_file)
        try:
            result = original_infer(model, batch, steps=steps, canvas=canvas, seed=seed)
        finally:
            if previous is None:
                os.environ.pop('AGENTICWAM_ORACLE_CODE_FILE', None)
            else:
                os.environ['AGENTICWAM_ORACLE_CODE_FILE'] = previous
        if module.ast_tree(result[1]) != module.ast_tree(expert):
            raise ValueError('inference did not preserve the supplied expert program')
        return result

    def provided_metrics(prediction, target, tokenizer, canvas):
        result = original_metrics(prediction, target, tokenizer, canvas)
        result['provided_code_api_valid'] = result['api_valid']
        result['prediction_origin'] = 'expert code supplied; not freely generated'
        for key in ('exact_match', 'ast_exact_match', 'syntax_valid', 'api_valid',
                    'api_error', 'primitive_sequence_match', 'predicted_calls',
                    'token_matches', 'token_slots', 'token_accuracy'):
            result[key] = None
        return result

    module.infer_once, module.code_metrics = conditioned_infer, provided_metrics
    return module


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evaluator', type=Path,
                        default=Path(__file__).with_name('evaluate_custom34_checkpoint.py'))
    parser.add_argument('--expert-code-file', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    module = build_probe(args.evaluator, args.expert_code_file)
    sys.argv = [str(args.evaluator), *remaining]
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())
