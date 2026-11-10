# 两卡单卡 batch=2 的 Custom34 修复试验

当前错误位于 DeepSpeed 反向梯度 allreduce 前的 flatten 临时缓冲分配：需申请 1.86 GiB，而 GPU 剩余 1.08 GiB。日志中 allocated=52.51 GiB、reserved but unallocated=24.36 GiB，分配器缓存和碎片也可能影响该次分配。它与先前整段代码 logits 转 FP32 的失败发生在不同位置。

本包在前一版代码 CE 显存修复上，增加小微批次的均衡采样和当前 run 专用配置准备工具。它不会启动训练、读取大 checkpoint 张量、改写数据集或重算统计值。

| 参数 | 原设置 | 新设置 |
|---|---:|---:|
| 每卡 batch_size | 4 | 2 |
| GPU 数 | 2 | 2 |
| gradient_accumulation_steps | 8 | 16 |
| 每次更新总样本数 | 64 | 64 |
| 每次更新每任务样本数 | 16 | 16 |

四任务无法同时放进一个只有 2 个样本的单卡微批次。新模式在两卡合计的每个微批次中严格保证每任务 1 个样本，任务组在卡间轮换；本设置下每张卡累积 16 次后各任务 8 个，两卡合计各 16 个。每卡 batch=4 的旧采样序列、随机数消耗和 sampler 描述符保持原样；改变 batch 的优化器状态续训仍会被原有严格检查拒绝。

## 手动安装

保留上次 v1.1 的代码 CE 显存修复，备份后按相同路径替换以下文件：

~~~text
src/agenticwam/utils/samplers.py
src/agenticwam/trainer.py
scripts/configure_custom_code_small_batch.py
scripts/custom_code_repair.py
scripts/verify_task_balanced_batches.py
~~~

前三个文件用于本次试验。后两个分别更新今后的 prepare 批次约束与只读批次验证工具。

## 准备当前 run 的配置

~~~bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
NEW_RUN="$PWD/runs/custom_code_repair_20261110_024220"

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" \
  scripts/configure_custom_code_small_batch.py \
  --run "$NEW_RUN" --batch-size 2
~~~

该命令读取现有生成的 launcher，确定真实进程数 2；将微批次改为 2，自动将梯度累积改为 16。它用真实 Accelerate BatchSamplerShard 与合成任务索引在 CPU 验证比例，在修改前检查错误，不读取真实视频。已保存 step 权重或设置 resume 的试验会被拒绝；本次连续 OOM 尚未完成首个优化器更新，符合此工具的适用范围。

它会在当前 run 下：

- 用 small_batch_backup_* 保存原 recipe/config.yaml 和 run_code_repair.sh。
- 生成 recipe/ds_zero1_bs2.json 和 recipe/accelerate_zero1_bs2.yaml；reduce_bucket_size 和 allgather_bucket_size 均不大于 50000000 个元素，保留原 overlap_comm、contiguous_gradients 等其他设置。
- 将 run_code_repair.sh 指向新 Accelerate 配置，并启用混合注意力 checkpoint、expandable_segments 分配器选项。
- 保存 small_batch_config_report.json，记录配置和 CPU 任务计数检查结果。

共享的项目 DeepSpeed/Accelerate 配置保持原样。统计文件、代码缓存、图像尺寸、128 token 画布、学习率、300 步上限及 foundation 权重保持原样。新通信桶的数值单位是元素而非字节；50M BF16 元素约 95 MiB。单个超大参数及其他缓冲仍可能大于桶阈值，实际 GPU 峰值需要重跑确认。减小微批次主要节省激活显存，不缩小模型和优化器状态。

工具成功时应显示：

~~~text
passed: true
rank_microbatch: 2
world_size: 2
gradient_accumulation_steps: 16
effective_batch: 64
observed_update_counts: {0:16, 1:16, 2:16, 3:16}
~~~

该配置命令执行一次即可；重复执行会拒绝覆盖已有小批次配置。

## 训练和复测

~~~bash
CUDA_VISIBLE_DEVICES=0,1 bash "$NEW_RUN/run_code_repair.sh" train
~~~

实际训练日志应有 balance_scope=global、global_quota=1，以及 last_update_samples={0:16,1:16,2:16,3:16}。模型和新 run 仍从指定的旧 8000 步权重开始本次 300 步试验。

完成后复测，现有基线沿用：

~~~bash
CUDA_VISIBLE_DEVICES=1 bash "$NEW_RUN/run_code_repair.sh" eval
~~~

相同有效 batch 并不保证逐位相同的训练数值；微批次分组和随机噪声/掩码采样发生变化。CPU 检查包括旧采样序列回归、真实 Accelerate 分片、任务计数、恢复顺序、配置备份与范围，以及两个 CPU/DDP 进程运行实际 trainer 三次更新、跨 epoch 和权重同步。GPU/DeepSpeed 的真实显存与吞吐未在本环境测量。
