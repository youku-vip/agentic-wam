#!/usr/bin/env bash
set -euo pipefail

SERVER_MODE="${1:-start}"
if [[ "$SERVER_MODE" != start && "$SERVER_MODE" != check ]]; then
  echo 'Usage: bash deploy-custom/start_custom34_real_test.sh [check|start]' >&2
  exit 2
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY:-/usr/local/bin/python3.11}"
DEPLOY_RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/custom_code_repair_20261110_024220}"
DEPLOY_CONFIG="${TRAIN_CONFIG:-$DEPLOY_RUN_DIR/recipe/config.yaml}"
DEPLOY_CHECKPOINT="${CKPT:-$DEPLOY_RUN_DIR/checkpoints/weights/step_000300.pt}"
DEPLOY_STATS="${STATS:-$DEPLOY_RUN_DIR/normalization_reference.json}"
DEPLOY_LOG_DIR="${LOG_DIR:-$DEPLOY_RUN_DIR/real_test_$(date +%Y%m%d_%H%M%S)}"
DEPLOY_GPU="${GPU_ID:-1}"
DEPLOY_HOST="${SERVER_HOST:-0.0.0.0}"
DEPLOY_PORT="${SERVER_PORT:-8765}"

cd "$REPO_DIR"
export AGENTICWAM_ROOT="$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export AGENTICWAM_DEPLOY_DTYPE=bf16
export AGENTICWAM_EVAL_PREPROC=auto
export AGENTICWAM_REMASK_ROUNDS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use the saved repair recipe and its training prompt map, without old ablations.
unset AGENTICWAM_CODE_ROPE_MODE AGENTICWAM_CODE_CANVAS AGENTICWAM_CODE_CANVAS_TAIL
unset AGENTICWAM_INFER_CODE_TOKENS AGENTICWAM_CODE_PREAMBLE_FILE AGENTICWAM_ORACLE_CODE_FILE
unset AGENTICWAM_CODE_NO_INSTRUCTION AGENTICWAM_CODE_NO_HISTORY
unset AGENTICWAM_CODE_MASK_TMIN AGENTICWAM_CODE_MASK_FULLP
unset AGENTICWAM_PAD_EOS_WINDOW AGENTICWAM_PAD_EOS_WEIGHT
unset AGENTICWAM_DECODE_TEMP AGENTICWAM_ORDER_TEMP
unset AGENTICWAM_REVEAL_CAP AGENTICWAM_REVEAL_TAU AGENTICWAM_MIN_CODE_LEN

"$PY_BIN" - "$DEPLOY_CONFIG" "$DEPLOY_CHECKPOINT" "$DEPLOY_STATS" "$DEPLOY_PORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import torch
from torch import nn
from agenticwam.models.experts.code_expert import CodeExpert

config, checkpoint, stats = map(Path, sys.argv[1:4])
for path in (config, checkpoint, stats, Path('deploy-custom/custom34_server.py'),
             Path('robotwin/deploy/deploy_policy.py')):
    if not path.is_file():
        raise FileNotFoundError(path)
if not 1 <= int(sys.argv[4]) <= 65535:
    raise ValueError('SERVER_PORT must be in 1..65535')
cfg = OmegaConf.load(config)
if str(cfg.mixed_precision) != 'bf16' or str(cfg.model.get('code_rope_mode')) != 'complex':
    raise ValueError('this launcher expects the BF16 / complex-RoPE repair configuration')
if cfg.model.get('code_memory', False) or str(cfg.model.code_objective) != 'joint':
    raise ValueError('this launcher expects joint code generation without code history')
if int(cfg.model.get('infer_code_tokens', 0)) != 128 or int(cfg.model.get('code_canvas', 0)) != 128:
    raise ValueError('this repair checkpoint must use the saved 128-token code canvas')
if int(cfg.data.train.num_frames) - 1 != 32:
    raise ValueError('this launcher expects a 32-step action horizon')
for split in (cfg.data.train, cfg.data.val):
    if list(split.video_size) != [256, 448] or str(split.processor.image_resize_mode) != 'letterbox':
        raise ValueError('expected training image processing: 256x448 letterbox')
prompt_map = Path(str(cfg.data.train.skill_prompt_map))
text_cache = Path(str(cfg.data.train.cosmos_text_cache_dir))
if not prompt_map.is_file() or not text_cache.is_dir():
    raise FileNotFoundError('training skill-prompt map/text cache is missing; relocate paths in the saved config')
prompts = json.loads(prompt_map.read_text())
stats_sha = hashlib.sha256(stats.read_bytes()).hexdigest()
recorded = Path(str(cfg.data.train.pretrained_norm_stats))
if recorded.is_file() and hashlib.sha256(recorded.read_bytes()).hexdigest() != stats_sha:
    raise ValueError('STATS differs from the normalization snapshot recorded in the training config')
probe = CodeExpert(blocks=nn.ModuleList([]), embed_tokens=nn.Embedding(32, 8),
    final_norm=nn.Identity(), lm_head=nn.Linear(8, 32, bias=False), hidden_dim=8,
    num_heads=1, attn_head_dim=8, vocab_size=32, code_rope_mode='complex')
before = probe.freqs.clone()
nn.Sequential(probe).to(dtype=torch.bfloat16)
if not probe.freqs.is_complex() or not torch.equal(before, probe.freqs):
    raise ValueError('replace the training RoPE repair: parent BF16 conversion changed the complex cache')
print(json.dumps(dict(preflight_passed=True, checkpoint=str(checkpoint.resolve()),
    train_config=str(config.resolve()), stats=str(stats.resolve()), stats_sha256=stats_sha,
    dtype='bf16', code_rope='complex128 preserved in CPU parent-cast probe',
    code_canvas=128, action_horizon=32, video_size=[256, 448], resize='letterbox',
    sample_stride=int(cfg.data.train.global_sample_stride), allowed_instructions=list(prompts),
    model_weights_loaded=False), ensure_ascii=False, indent=2), flush=True)
PY

if [[ "$SERVER_MODE" == check ]]; then
  exit 0
fi

mkdir -p "$DEPLOY_LOG_DIR"
echo "[custom34] GPU=$DEPLOY_GPU host=$DEPLOY_HOST port=$DEPLOY_PORT logs=$DEPLOY_LOG_DIR"
exec env CUDA_VISIBLE_DEVICES="$DEPLOY_GPU" "$PY_BIN" deploy-custom/custom34_server.py \
  --train-config "$DEPLOY_CONFIG" \
  --ckpt "$DEPLOY_CHECKPOINT" \
  --dataset-stats "$DEPLOY_STATS" \
  --device cuda:0 \
  --action-horizon 32 \
  --num-inference-steps 10 \
  --review-dir "$DEPLOY_LOG_DIR/reviews" \
  --vae-decode-dtype fp32 \
  --code-log-dir "$DEPLOY_LOG_DIR/code" \
  --host "$DEPLOY_HOST" --port "$DEPLOY_PORT"
