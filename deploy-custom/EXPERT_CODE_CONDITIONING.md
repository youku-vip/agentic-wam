# 专家代码条件离线对照

适用于已安装 `evaluate_custom34_checkpoint.py` 的 custom34 离线评测。包装器复用该文件，保留本地 BF16、reference 数组和可选 RoPE 对照补丁；原评测脚本不被写入。

它利用项目已有的 `AGENTICWAM_ORACLE_CODE_FILE` 分支，将当前观测对应的专家程序 token 作为代码条件。该 Python 只被读取、分词和 AST 比较，不会执行；没有专家动作坐标被注入预测轨迹。普通离线评测仍拒绝这个环境变量，包装器只在一次推理调用内临时设置并恢复。

这是带额外专家信息的诊断实验，不能作为自由生成代码的评测或实机表现。`run.json`、`summary.json` 会标明 `code_conditioning="provided expert code"`、`code_generation_evaluated=false`；代码准确率字段置为 null。返回文本仍保存在 `prediction.py`，但其来源在指标中明确标为提供的专家代码。

要求代码目标为 `joint`，关闭 remask，专家文件与所选观测的目标代码完全匹配，且代码加 EOS 能放入画布。检查不满足时拒绝该样本，不会替换为其他目标。

例如，在与已有自由生成结果相同的观测、种子和 complex RoPE 模式下运行：

```bash
cd /data/algorithm/agentic-wam
RUN=/data/algorithm/agentic-wam/runs/air_fryer_zero2_10000_p012_v4_1
REFERENCE=/data/algorithm/agentic-wam/checkpoint_eval_complex_20261109_141550/episode000002_frame000005_c128_n10_s42
OUT="./checkpoint_eval_expert_code_$(date +%Y%m%d_%H%M%S)"

AGENTICWAM_EVAL_CODE_ROPE=complex AGENTICWAM_REMASK_ROUNDS=0 \
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/src" \
  /usr/local/bin/python3.11 deploy-custom/evaluate_custom34_expert_conditioning.py \
  --expert-code-file "$REFERENCE/target.py" \
  --train-config "$RUN/config.yaml" \
  --checkpoint "$RUN/checkpoints/weights/step_008000.pt" \
  --stats "$RUN/dataset_stats.json" \
  --episodes 2 --start-frames 5 --inference-steps 10 \
  --canvas-tokens 128 --seeds 42 43 44 --device cuda:0 --out "$OUT"
```

仅比较相同种子的物理动作指标。明显改善支持“代码条件会影响该样本的动作”这一判断，但不等于证明代码分支是唯一原因；没有改善也不能单独否定所有代码条件影响。此实验会把当前阶段专家程序固定在整个预测过程中，这是它与自由代码联合去噪的差别。

CPU 验证使用模拟模型和数据源，实际执行转换、归一化、物理指标和报告导出，检查环境变量恢复、专家目标不匹配拒绝以及提供代码不计为生成准确率。另验证历史 v1 评测脚本加本地三处修复的兼容性。本环境未加载用户的大模型权重或运行 GPU 对照。
