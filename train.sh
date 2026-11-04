cd /data/algorithm/agentic-wam

# ===== GPU / CUDA =====
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ===== Agentic-WAM 加速配置 =====
export AGENTICWAM_PHASE_TIMING=0
export AGENTICWAM_MOT_NO_CKPT_EVERY=1

# ===== W&B =====
export WANDB_BASE_URL="http://10.102.108.234:8080"
export WANDB_MODE=online

# WANDB_API_KEY 请通过 Kubernetes Secret 或安全环境变量注入
export WANDB_API_KEY="local-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"

# ===== 模型权重 =====
BASE=/data/modelRepository/w001/w001_agentic-wam-pretrain-yamego3k-soup-e1

PRETRAIN="$BASE/agenticwam_pretrain.pt"
SDAR="$BASE/weights/SDAR-1.7B-Chat"
QWEN="$BASE/weights/Qwen3-VL-2B-Instruct"
WAN_VAE="$BASE/weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"

# ===== 新训练输出目录 =====
OUT=./runs/air_fryer_zero2_bs96_3500_p012_v2

mkdir -p "$OUT"

# ===== GPU 检查 =====
/usr/local/bin/python3.11 - <<'PY'
import torch

n = torch.cuda.device_count()
assert n == 4, f"Expected 4 GPUs, got {n}"

print("GPU check: PASS", n)
print("Global batch: 4 GPUs x 6 Micro Batch x 4 Accum = 96")
PY

# ===== 启动正式训练（不使用 Nsight） =====
/usr/local/bin/python3.11 scripts/launch_train.py \
  --task posttrain_yam \
  --nproc 4 \
  --zero 2 \
  data=custom \
  paths.pretrain_ckpt="$PRETRAIN" \
  paths.code_model="$SDAR" \
  paths.code_vision="$QWEN" \
  paths.wan_vae="$WAN_VAE" \
  model.use_gradient_checkpointing=true \
  wandb.enabled=true \
  wandb.mode=online \
  wandb.project=agenticwam-air-fryer \
  wandb.name=aw0-bs96-3500-p012-v2 \
  wandb.group=air-fryer-v3 \
  batch_size=6 \
  gradient_accumulation_steps=4 \
  num_workers=8 \
  mixed_precision=bf16 \
  learning_rate=1e-4 \
  weight_decay=0.01 \
  max_grad_norm=1.0 \
  lr_scheduler_type=cosine \
  max_steps=3500 \
  log_every=10 \
  save_every=500 \
  eval_every=0 \
  +save_optimizer_state=true \
  +keep_last_ckpts=10 \
  output_dir="$OUT"
