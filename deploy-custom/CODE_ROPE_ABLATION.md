# CodeExpert 复数 RoPE 离线对照

用户的 CPU 检查确认：当前 SDAR builder 有 `expert.to(device=device, dtype=torch_dtype)`，CodeExpert.freqs 是非持久复数 buffer，经过 BF16 类型转换后成为实数，虚部丢失。共享 rope_apply 会直接使用这个缓存，数值上不再是原来的旋转。

这确认了实现问题，不足以证明它是所有代码/动作误差的唯一原因。现有权重可能已经在实数缓存路径下训练。训练时的非持久 buffer 不包含在 checkpoint 中，单靠权重文件不能确定当时的缓存状态。不能把推理切换到复数缓存就当作旧权重已自动修复。

本补丁仅给 `evaluate_custom34_checkpoint.py` 增加进程内、显式开启的对照模式，不改 CodeExpert、训练代码、服务端、数据或权重文件。默认 `AGENTICWAM_EVAL_CODE_ROPE=original` 完全保留构造器给出的缓存。设为 `complex` 时，只有重建缓存的实部转成旧 dtype 后与原缓存逐值一致，才允许补回复数缓存；否则报告 theta 不匹配。当前 builder 的 theta 是 1000000，可显式通过 `AGENTICWAM_EVAL_CODE_ROPE_THETA` 选择。

## 安装到已有、已修复的离线评测脚本

```bash
cd /data/algorithm/agentic-wam
/usr/local/bin/python3.11 deploy-custom/patch_checkpoint_rope_ablation.py
```

安装器会先编译检查并备份文件，保留之前的 BF16 和 reference 数组修复。重复安装不重复插入。

## 同窗口对照

```bash
RUN=/data/algorithm/agentic-wam/runs/air_fryer_zero2_10000_p012_v4_1
OUT="./checkpoint_eval_complex_rope_$(date +%Y%m%d_%H%M%S)"

AGENTICWAM_EVAL_CODE_ROPE=complex CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/src" \
  /usr/local/bin/python3.11 deploy-custom/evaluate_custom34_checkpoint.py \
  --train-config "$RUN/config.yaml" \
  --checkpoint "$RUN/checkpoints/weights/step_008000.pt" \
  --stats "$RUN/dataset_stats.json" \
  --episodes 3 --start-frames 5 --inference-steps 10 \
  --canvas-tokens 64 128 --seeds 42 --device cuda:0 --out "$OUT"
```

选择一张空闲且显存足够的 GPU。这个命令只重新评测一个窗口，和现有 original 结果的 seed、帧、步数、画布一致。`summary.json` 的 `provenance.code_rope_audit` 应显示 before_complex=false、quantized_cosine_matches=true、rebuilt=true、after_complex=true；这才是实际已加载模型的缓存信息。

不设置该变量或显式设置为 `original` 即保留原行为。环境变量只影响新启动的离线评测进程。是否将复数缓存用于后续训练、旧权重微调或服务端，需要根据对照结果和训练时实现决定。

## 验证范围

使用当前真实 CodeExpert 的小型 CPU 实例验证原模式缓存对象和数值不变、复数模式只改非持久缓存不改 state_dict、位置旋转保持模长、错误 theta 被拒绝、安装幂等与语法有效。未在本环境加载用户的真实大模型或运行 GPU 推理。

另外在用户已安装的历史评测脚本（v1 加 BF16 与 reference 数组两处修复）上执行了完整 CPU 报告导出验证：模型与数据源采用模拟对象，转换、归一化、指标和文件导出采用真实实现。补丁兼容该脚本，报告记录了从 BF16 实数缓存重建为 complex128 复数缓存。此检查不代表真实 checkpoint 的生成精度。
