# 四任务均衡 batch

新增可选开关 `AGENTICWAM_TASK_BALANCED_BATCH=1`。默认仍使用原来的随机打乱；只有开启后才改变训练样本的分布。

当前配置每卡 batch=4、2 卡、梯度累积 8，四任务均衡后的组成如下：

| 范围 | 总样本 | 每种任务 |
| --- | ---: | ---: |
| 每卡每个 microbatch | 4 | 1 |
| 双卡同一次 microstep | 8 | 2 |
| 一次 optimizer update，累积 8 次 | 64 | 16 |

当前 loader 默认随机打乱全部帧，只能在长期维持原始帧比例，不能保证一个 batch 包含全部任务。用户提供的四任务帧数依次为 50,526、88,583、74,288、71,865，共 285,262 帧，原始比例约为 17.7%、31.1%、26.0%、25.2%。四任务的 episode 数相同，不代表帧数相同。

## 安装和启用

使用 `deliveries/training-throughput-fixes-v2-task-balanced.zip`，其中包含已有训练优化及本次均衡采样模块。解压后的安装器先校验源码 SHA256，再备份、安装 16 个源码/工具文件；保留本地 `train.sh`、全部 YAML、模型路径和数据文件。旧 v1 压缩包不会自动获得本次功能。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11

git fetch origin codex/custom-data-review-files
git show FETCH_HEAD:deliveries/training-throughput-fixes-v2-task-balanced.zip \
  > /tmp/training-throughput-fixes-v2-task-balanced.zip
STAGE_DIR=$(mktemp -d /tmp/agenticwam-task-balance.XXXXXX)
"$PY" -m zipfile -e /tmp/training-throughput-fixes-v2-task-balanced.zip "$STAGE_DIR"
"$PY" "$STAGE_DIR/scripts/install_training_optimizations.py" --project-root "$PWD"
```

在原来已跑通的正式训练环境变量基础上增加：

```bash
export AGENTICWAM_TASK_BALANCED_BATCH=1
bash train.sh
```

新源码在下次启动生效。每卡 batch_size 必须能被实际任务数整除，例如四任务使用 4、8、12；不满足时会直接报错。也可在训练命令中使用 `++task_balanced_batch=true`，显式配置优先于环境变量；无需替换现场 YAML。

此功能与 ZeRO-1 参数扫描缓存、视频稀疏读取相互独立，沿用之前已验证的环境即可。若更换为已包含官方扫描修复的 DeepSpeed（官方 v0.18.7 已包含 [PR #7886](https://github.com/deepspeedai/DeepSpeed/pull/7886)），单独验证兼容性并关闭 `AGENTICWAM_CACHE_DS_GRAPH_COUNTS`；本次均衡采样不会替换 DeepSpeed。

## 在现场核验

先运行只读采样计划检查，不读取视频、不加载模型，也不重算统计：

```bash
DATA_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src "$PY" scripts/verify_task_balanced_batches.py \
  --root "$DATA_ROOT" --batch-size 4 --world-size 2 --grad-accum 8 --steps 3 \
  > /tmp/task_balanced_batch_check.json
cat /tmp/task_balanced_batch_check.json
```

预期：`passed=true`，`expected_per_rank_microbatch` 每任务为 1，`observed_optimizer_batch_counts` 每次都是四任务各 16。默认检查全部 episode；若训练配置只选了部分 episode，验证时用 `--episodes` 传入同样的选择顺序。

训练启动时打印 `[task-balance] enabled`，包括任务编号到任务名的对应、原始帧数、每卡配额和采样 epoch 长度。正常日志间隔还会统计实际返回的数据：

```text
[task-balance] step=10 last_update_samples={0: 16, 1: 16, 2: 16, 3: 16}
```

W&B 对应 `sampling/task_0_samples` 等，记录的是当前一次完整 optimizer update 的样本数。它们覆盖所有梯度累积 microbatch；现有 loss 曲线仍沿用原来的记录口径，不会因为此开关自动变成整次累积的平均。

编号由采样器的任务名映射确定；以启动日志为准。多数据根目录中相同的主任务文本归为同组。`code_eval` 仍只评估更新最后一个 microbatch；当前双卡设置下共有 8 个样本，正常为每任务 2 个，空代码标签会被评测器排除。

## 采样与失败重试规则

任务索引由训练 loader 实际选择的 v3 episode 顺序及帧级 `task_index` 构建，支持 shared Parquet、episode 子集、非连续 episode 编号，以及同一 episode 内的任务变化。每个 packed Parquet 文件只读取所需的 episode/frame/task 三列，避免为每个视频窗口重新查任务。不会把四个任务假定为四段等长索引。

每个任务的样本池独立随机打乱。按任务配额抽取后，再打乱每卡 batch 内的顺序。较小任务池用完后会重新打乱并循环；较大任务池本 epoch 可能只访问部分帧。正常采样计划中，同一全局 microbatch 的两个 rank 使用不同样本；为保持这个性质，每个任务池在不足一次全局配额时重新洗牌，最多暂时留下小于该配额的尾部样本。

epoch 预算取数据集长度向上补齐到有效 batch 的整数倍，避免尾部 batch 被 Accelerate 补齐后失衡，也避免尾部只有部分梯度累积。当前数据和 batch=4/world=2/GAS=8 时，每个采样 epoch 为 285,312 个样本，每任务 71,328 个，接近原数据总长度。这会增加短任务的采样权重，训练 loss 的混合比例也会改变；不能将新旧混合 loss 的绝对值直接当成同口径对照。

读取/预处理异常或规避 padding 的重试始终限制在同一任务中。Base loader 在这个模式下会把读取失败交给外层按任务重试，禁止原来的跨任务随机替换。任务内连续失败达到上限时明确报错；重试可能重复同任务样本。任务类别来自实际数据，返回字段新增 `sampling_task_id`，用于跨卡计数，模型的图像、action、proprio 和代码内容沿用原处理逻辑。

## 断点续训与验证范围

采样使用独立的 CPU Torch Generator，按 seed 和 epoch 决定次序。恢复时重放跳过部分的随机过程，再从保存的 rank-local microbatch 偏移继续，避免因预取而把尚未训练的 batch 计入进度。

这里可复现的是采样器的索引计划；随机数据增强或故障重试产生的替代样本，还会受各自随机状态和实际读取结果影响。

新的 `trainer_state.json` 保存采样方案，包括任务索引签名、seed、batch/world/GAS。均衡模式恢复要求这些条件一致；不一致时会明确报错。旧 checkpoint 或切换随机/均衡模式时会打印样本顺序变化提示：可以继续加载原有训练状态，但这属于改变采样分布，不能称为复现原来的样本次序。开启均衡采样不需要重新生成 code cache 或 state/action 统计；本次实现保持现有归一化定义。

本地 56 项测试通过，包含各卡/全局/累积配额、跨 rank 索引不重复、任务池循环、确定性恢复、真实 v3 索引映射、同任务失败/padding 重试以及图像/动作/代码输入不变。另用两个真实 CPU/DDP 进程执行当前 trainer，GAS=8、跨 epoch 更新三次，每次都统计到四任务各 16 个样本，且权重同步。当前环境没有 GPU 或现场完整数据，尚未运行真实 ZeRO-1 GPU 训练。
