# Batch 代码生成精度

训练现在可以定期对当前 batch 真正生成代码，再与 `policy_code` 标签比较。主指标为 `code_eval/exact_match`。指标范围均为 0–1，例如 0.75 表示 75%。

实际训练入口导入的是 `src/agenticwam/trainer.py`；本次修改位于该文件和新增的 `src/agenticwam/utils/code_metrics.py`，不是仓库根目录的同名 `trainer.py`。

项目原有 `train/code_acc` 保留：它来自训练前向的 logits。在 joint/blockdiff 目标下，它只统计被 mask 的监督位置；已显露的目标代码和加噪的真实未来轨迹可能作为上下文，因此不能用它代替自由生成精度。

## 使用

`configs/train.yaml` 新增以下配置，适用于原来的 Hydra 训练入口：

```yaml
code_eval_every: 100
code_eval_max_samples: 0
code_eval_inference_steps: 10
```

- `code_eval_every`：每隔多少个 optimizer step 进行一次；设为 `0` 关闭。独立于 `eval_every` 和 `log_every`，原来 `eval_every=0` 也能记录代码指标。
- `code_eval_max_samples`：每张卡最多评测多少个样本；`0` 表示该卡当前 microbatch 全部样本。设为 `1` 可以减少开销，但只反映抽样结果。
- `code_eval_inference_steps`：真实推理的去噪步数；比较不同训练阶段时应保持相同，并尽量与部署一致。

在原有 `scripts/train.py` 或 `scripts/launch_train.py` 命令后追加：

```bash
code_eval_every=100 code_eval_max_samples=0 code_eval_inference_steps=10
```

如果启动 shell 脚本没有传递 `"$@"`，应将以上三项加入脚本原有训练命令，或直接修改 `configs/train.yaml`。无需修改数据集或重建 code cache。

沿用自定义任务配置的直接启动示例（模型路径、数据覆盖参数沿用已有命令）：

```bash
python scripts/launch_train.py --task posttrain_custom --nproc 4 --zero 2 \
  code_eval_every=100 code_eval_max_samples=0 code_eval_inference_steps=10
```

生成本身需要额外推理时间。当前模型的推理接口一次只支持一个样本，因此每张卡依次生成，卡之间并行。为避免每步明显变慢，默认每 100 步评测一次；只想短时间检查每个 step 时可设 `code_eval_every=1`。

## 指标定义

| 名称（W&B 前缀 `code_eval/`） | 口径 |
| --- | --- |
| `exact_match` | 预测代码与标签 token 序列完全相同的样本数 / 非空标签样本数。只忽略代码首尾空白，不截断标签。 |
| `token_accuracy` | 相同位置的相同 token 数 / `sum(max(预测长度, 标签长度))`。缺失和多生成都会扣分；是按位置匹配，不是编辑距离或语义相似度。 |
| `ast_exact_match` | Python AST 完全匹配的样本数 / 非空标签样本数。忽略格式和注释，保留函数、调用顺序、参数、数值与左右手差异；不等价于行为正确。 |
| `syntax_valid_rate` | 能解析并编译为非空 Python 程序的生成结果比例。空代码、只有注释、语法错误均不通过。不会执行生成代码；不校验机器人 API 是否存在。 |
| `samples` | 实际评测的非空标签样本数。 |
| `unlabeled` | 评测范围内空标签样本数；不参与精度分母，也不为其调用生成。 |
| `invalid_targets` | 非空但语法无效的标签数；保留在精度分母中，便于发现数据问题。 |
| `target_over_canvas_rate` | 标签正文 token 数超出当前配置的推理 canvas 长度的比例。仅作长度预警；实际可用长度还可能受 prompt 和模型上下文上限影响。 |
| `seconds` | 本次评测耗时，含生成和跨卡汇总。 |
| `failed_ranks` | 推理失败的 rank 数；任意 rank 失败均跳过本次精度，不将失败伪装成 0 分。 |

token 比较只包含代码正文，不加 tokenizer 的特殊 token，因此 prompt、padding 和 EOS 不会提高分数。所有 rank 先汇总整数计数，再计算比例，避免不同长度或空标签造成平均偏差。没有有效标签时不报告准确率。

如果代码包含连续坐标，数值略有不同也会导致完全匹配失败；可同时观察 token 准确率和生成样例。本指标尚不包含坐标误差容忍、API 参数校验或实际任务成功率。

## 评测输入和 batch 范围

评测使用 optimizer 更新后模型和本次更新最后一个 microbatch，输入每个相机的第一张图、当前 proprio、指令文本及文本 embedding。未来视频与真实 action 数值不传给推理；action 只提供所需预测长度；不传入目标 `policy_code` 或已有代码缓存。

例如 4 卡、每卡 batch size 为 6、梯度累积为 4：一次完整 probe 最多覆盖当前 microbatch 的 **24 个样本**，不是 optimizer effective batch 的 96 个样本。`max_samples=1` 时最多覆盖 4 个。空标签会进一步减少 `samples`。

指令选择与训练一致：开启 code memory 时优先 `prompt_nohist`，否则优先 `prompt_code`，最后使用 `prompt`。code memory 每个样本使用其已有的 `code_history`，这是带真实历史的单步代码评测，不代表完整 episode 自回归 rollout。评测结束恢复原来的历史、模块 train/eval 状态、scheduler 和 Torch RNG。

支持单卡、DDP 和 ZeRO 1/2 的非分片模型推理；ZeRO 3 暂不支持，会记录失败并跳过。本实现拒绝设置了 `AGENTICWAM_ORACLE_CODE_FILE` 的环境，以防 oracle 答案被误计为生成精度。关闭 code expert 的模型不触发 probe。

## 查看结果

终端输出 `[code-eval] step=...`，W&B 使用 `code_eval/` 前缀。输出目录新增 `code_eval.jsonl`：每次一行，包含所有 rank 的汇总指标，以及 **rank 0** 的标签和预测代码对照；样例不是全卡汇总。

示意值（不是实测训练结果）：batch 内 24 个有效标签中 18 个整段匹配，则 `samples=24`、`exact_match=0.75`。语法有效率很高、完全匹配率很低时，应检查函数和参数是否正确，不能只凭语法有效就判断模型学会任务。

## 验证和论文口径

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/check_code_metrics_distributed.py
```

CPU 测试覆盖输入无未来信息泄漏、独立历史、异常后的状态恢复、多余/缺失 token、空标签、AST 比较、禁止执行代码，以及真实双进程 Accelerate 计数汇总和单 rank 失败处理。当前环境没有 GPU 和训练权重，尚未验证真实 ZeRO 2 生成耗时或训练数值。

参考用户提供的[项目论文网站](https://agentic-wam.github.io/)，并核对其[公开页面源码](https://github.com/agentic-wam/agentic-wam.github.io/blob/6c1ac9b6d2d178a0649b142738bb6d7b829db606/src/app/page.tsx)。网站展示的 RoboTwin 成功率和实机任务评分衡量执行结果；这里新增的是代码生成诊断指标，不宣称复现论文评测。
