# 无历史 code_eval.jsonl 时重新评测 checkpoint

`code_eval.jsonl` 不是训练必需文件。当前 trainer 只有在代码生成 probe 成功后才写入该文件；所有 probe 失败、使用旧 trainer、日志目录不同或文件未保留，都可能导致文件不存在。`eval_every=0` 与独立的 `code_eval_every` 是两个开关，前者不会关闭后者。

本工具重新加载已有权重，通过 **config.yaml 中原训练数据读取和预处理流程** 生成代码与动作。它不会恢复训练，不创建优化器，不使用 DeepSpeed，不访问机器人或 HTTP 服务，也不解码生成视频的像素。因此不能把新结果称为第 8000 步训练当时的历史日志。

## v1.1 精度修复

v1 的构造器会留下部分 FP32 模块，独立评测没有经过训练时 DeepSpeed 的类型统一，因此可能报 `mat1 and mat2 must have the same dtype, but got Float and BFloat16`。这不是一次成功的模型效果评测，`completed=0` 不能解读为准确率为零。

v1.1 在加载权重后统一浮点参数和浮点 buffer 的推理精度，并使用 `Module.bfloat16()/half()/float()` 保留复数 RoPE 和整数 buffer。`run.json` 记录版本和转换前后的 dtype 数量；逐 case 失败时另存完整堆栈到 `errors/`。这一修复只涉及离线评测工具。

## 安装

将 `deploy-custom/evaluate_custom34_checkpoint.py` 放到同名目录。其他训练/服务端文件不需要替换；依赖当前项目已有的模型、数据读取、CustomUnifiedEEFTransform 和 custom34_review 模块。

## 单样本

使用一张有足够剩余显存的 GPU；如果现有服务占用 GPU 0，可以选择空闲的 GPU 1。下面的 `CUDA_VISIBLE_DEVICES=1` 会把物理 GPU 1 映射为本进程的 `cuda:0`。

```bash
cd /data/algorithm/agentic-wam
RUN=/data/algorithm/agentic-wam/runs/air_fryer_zero2_10000_p012_v4_1
OUT="./checkpoint_eval_step8000_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD/src" /usr/local/bin/python3.11 \
  deploy-custom/evaluate_custom34_checkpoint.py \
  --train-config "$RUN/config.yaml" \
  --checkpoint "$RUN/checkpoints/weights/step_008000.pt" \
  --stats "$RUN/dataset_stats.json" \
  --episodes 3 --start-frames 5 \
  --inference-steps 10 --canvas-tokens 64 --seeds 42 \
  --device cuda:0 --plots --out "$OUT"
```

模型会被单独加载一次，显存需求与推理有关；本工具不提供针对该 checkpoint 的显存容量承诺。训练配置中的数据、代码缓存、技能提示词、Cosmos 文本缓存和冻结模型权重路径必须在这台机器可访问。它默认优先使用 run 内的统计快照，且显式禁用数据集构造时写统计文件。

## 对照参数

需要比较预算或去噪步数时，使用一个新输出目录，把参数改为：

```bash
--inference-steps 10 20 --canvas-tokens 64 128 --seeds 42 43 44
```

对于同一窗口和 seed，不同参数组合从相同随机种子开始；每次推理后恢复 scheduler 和 RNG 状态。CLI 的 canvas 在该进程内临时覆盖推理环境变量，不改变 checkpoint 或其他运行进程。增加 token 或步数只是对照测试，不保证效果提升。

如需覆盖四个任务，可使用这份清洗数据的 `--episodes 3 1 2 3258 --start-frames 5`。默认拒绝不足 32 个真实动作的窗口。检查后期阶段时，可显式使用 `--start-frames 5 30 --padding repeat`；此时按训练流程重复尾帧，并将补齐动作排除于物理误差指标。默认只检查 episode 3、frame 5。

## 输出和如何判断

- `summary.json`：每个组合的代码/动作结果和失败原因。
- `code_eval.jsonl`：本次离线重新生成的逐样本记录。
- `run.json`：checkpoint step、配置/统计/缓存/提示词指纹、运行精度和参数。
- 每个 case 的 `target.py`、`prediction.py`、`prompt.txt`：标签、自由生成结果和实际 code prompt，不执行生成的 Python。
- `code_metrics.json`：真实 tokenizer 的提示词/目标长度，是否超过预算，语法正确率、API 参数完整性、原子 API 顺序/左右臂匹配。
- `metrics.json`、`chunks.npz`：32 步动作与专家的物理误差和实际数组。加 `--plots` 会生成位置/姿态/夹爪对比图。
- `observation_model_input/contact_sheet.png`：训练数据流程实际送入模型的三路观测。

API 完整性与语法不同：例如 `target_rpy=[179.56,-82.64]` 的 Python 语法有效，但 API 检查会失败。原子 API 序列匹配只检查 API 名称和左右臂，不能证明目标坐标、角度或物理任务成功。

脚本会验证数据读取没有把请求样本悄悄换成其他样本，并独立核对原始 state/action 经过转换、归一化后与训练 loader 一致。缺少 checkpoint 必需参数或索引不一致时停止，不将其计为低准确率。

如果通过训练数据流程重新生成也选错阶段，问题已在当前 checkpoint 的生成推理中出现；还需要结合输入/标签语义检查模型与训练数据。若离线生成较好而服务端较差，再比较双方输入、精度、推理参数和加载的权重。少量随机种子或单个样本不代表全数据集成功率。

## 已完成与未完成验证

在 CPU 上验证了混合 FP32/BF16 模块可重现原错误、统一精度后可运行且复数/整数 buffer 保留，以及三路首帧选择、代码参数检查、左右臂匹配、禁止执行生成代码、静默换样本保护、随机种子/scheduler 恢复（包括异常路径）、相对动作到绝对动作只逆归一化一次的还原。未在本环境加载用户的大模型权重或运行真实 GPU 推理；实际 tokenizer 长度和 checkpoint 效果由上述命令在用户环境测量。
