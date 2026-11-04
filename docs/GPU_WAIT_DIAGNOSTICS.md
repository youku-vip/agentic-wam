# 定位训练中的 GPU 等待

目标是用当前两卡训练的时间线，区分数据供应、CPU/GPU 同步和卡间通信等待。当前环境没有真实 GPU、数据集和模型权重，因此这里提供的是诊断能力，尚未测出用户机器上的主瓶颈或提速比例。

## 启用

更新本分支源码后，沿用已能正常运行的两卡 `train.sh`，在启动前执行：

```bash
export AGENTICWAM_WAIT_DIAG_STEPS=1
export AGENTICWAM_WAIT_DIAG_WARMUP=1
bash train.sh
```

`AGENTICWAM_WAIT_DIAG_STEPS` 默认为 `0`，即关闭。上例从本次启动或恢复的位置开始，先完成 1 个 optimizer step，再抓取完整的 1 个 optimizer step。梯度累积 8 时，抓取范围覆盖 8 个 microstep，并包含优化器更新；不是只测一个 microstep。需要更长窗口时可设 `STEPS=2`，同时会增加文件大小与诊断开销。

这是新开关，独立于 `AGENTICWAM_PHASE_TIMING`。原脚本将 `PHASE_TIMING` 设为 `0` 也能采集。抓取期间会临时跳过旧阶段计时插入的 barrier 和 CUDA synchronize，避免把这些诊断同步当成训练本身的等待。它不主动新增分布式 barrier 或 CUDA synchronize；PyTorch profiler 自身启动、停止和导出仍有开销，不能拿这段吞吐当正常训练速度。

采集完成会自动停止 profiler、导出文件，然后继续训练。无需改 batch、学习率、loss 或数据采样。当前已经运行的旧进程不会因更新磁盘文件自动获得新诊断能力，应在下一次正常启动/恢复时开启。诊断完成后，未来启动可关闭：

```bash
unset AGENTICWAM_WAIT_DIAG_STEPS
unset AGENTICWAM_WAIT_DIAG_WARMUP
```

不要用 `CUDA_LAUNCH_BLOCKING=1` 跑性能对比，它会改变执行的同步方式。短时诊断期间也应保留原有 checkpoint 和 batch 设置，便于对照；如要排除生成评测，需在训练命令内部显式设 `code_eval_every=0`，并记录这个变化。

## 输出

两张卡分别在当前 `output_dir` 下写文件：

```text
wait_diagnostics/
  rank0-step000001-<唯一时间戳>.summary.json
  rank0-step000001-<唯一时间戳>.trace.json
  rank1-step000001-<唯一时间戳>.summary.json
  rank1-step000001-<唯一时间戳>.trace.json
```

`step` 是抓取开始时已经完成的 optimizer step 数，恢复训练时会使用恢复后的步数。先检查两张卡的 `summary.json`；需要进一步分析时，在本地用 Perfetto 或支持 Chrome trace 的工具打开 `trace.json`。优先提供两份 summary；它们包含实际 world size、ZeRO stage、每卡 batch、梯度累积、每卡 worker 数，以及耗时最大的 CPU 算子。

摘要中的 `host_phases` 是主机侧耗时，包含 Python 操作、CUDA 提交和阻塞；不是纯 GPU 计算时间。嵌套阶段会重叠，不能直接将所有行相加：

| 阶段 | 覆盖范围 |
| --- | --- |
| `data_next_including_h2d` | 取下一个 prepared batch；包含等待 worker、DataLoader 拼接/传输和 Accelerate 的 H2D 搬运。只观察主进程，不展开 worker 内部解码。 |
| `input_device_and_dtype` | 模型入口的 `.to(device/dtype)`。prepared batch 已在 GPU 时可能只是同设备操作，不应一律算成额外复制。 |
| `vae_encode` | 冻结 VAE 的视频编码前向。 |
| `text_embeddings` | 取 batch 中的文本 embedding、可选文本编码和 dtype/device 转换；逐文件缓存读取发生在 worker 时归于数据阶段。 |
| `sigma_density_cpu_roundtrip` | 当前 sigma 重加权实现中的 `y.cpu()`、CPU 概率计算和返回 GPU。只在对应模型分支启用时出现。 |
| `code_tokenize_and_mask` | Python tokenizer、构建代码张量、随机 mask；内部 GPU 标量转 CPU 引发的同步也会落在此区间。 |
| `vision_prefix` | 图像 ViT 前向和 merger。 |
| `mot_forward` | 三专家混合 attention 的前向提交。 |
| `code_loss_and_accuracy` | 代码输出头、交叉熵和训练 token 准确率。 |
| `metric_scalars_to_host` | 将 loss/accuracy/sigma 等 CUDA 标量转成 Python 数字时发生的等待。 |
| `backward_including_comm` | 反向传播，可能包含梯度通信和重计算。 |
| `clip_and_optimizer_including_comm` | 梯度裁剪与优化器更新，可能包含 ZeRO 通信。 |
| `wandb_log` | 实际发生 W&B 日志提交时的主机耗时。 |

外层 `forward` 覆盖模型整个前向，包括其中的等待。其他 CUDA runtime、NCCL、Python 调度和保存开销也会出现在 trace 中；窗口没有覆盖到日志或 checkpoint 的触发步时，不会测到它们。

## 如何判断

| 时间线现象 | 下一步 |
| --- | --- |
| GPU 无 kernel/复制事件，主进程长时间处于 `data_next_including_h2d` | 继续区分 worker 读盘、视频解码和 H2D；对比本地存储、worker 数和 CPU 配额。 |
| GPU 空白与代码 tokenization、CPU roundtrip 或 Python 调度重合 | 优先处理对应的主机工作，例如缓存 tokenization、消除 CPU 往返或合并指标同步。 |
| 一张卡尚在准备数据，另一张卡已进入长时间 NCCL kernel | 先解决 rank 进度不一致；不能仅凭 NCCL 时间长就认定互联带宽差。 |
| 两卡均已进入通信，NCCL kernel 持续很长 | 检查 GPU 拓扑、通信带宽与 bucket 设置；保留可比较的模型和 batch。 |
| GPU kernel 连续运行，CPU 卡在 `cudaStreamSynchronize` | CPU 在等待 GPU 完成计算，这本身不能证明 GPU 空闲；查看 kernel 内容和覆盖区间。 |

`timeline` 的 coverage 来自 trace 中 GPU kernel、复制和 memset 区间的并集，不会将并发事件重复相加。NCCL kernel 单独统计，因为它也可能在等待。`uncovered_trace_window_fraction` 是该捕获窗口没有记录 GPU 活动的比例，不是硬件 SM 利用率，也不是完整训练的空闲比例。

如果 `gpu_activity_available=false`，说明没获得 GPU 活动事件（CPU 测试或现场 CUDA/CUPTI 采集不可用），不是 GPU 利用率为 0。此时先分析 CPU 阶段，不能用缺失的事件下 GPU 等待结论。`partial=true` 表示抓取没覆盖完整配置窗口，例如异常或训练提前结束。

## 静态检查已发现的等待点

当前模型在前向中将多个 CUDA 指标转成 Python `float`；joint 随机 mask 也会读取 CUDA 标量，sigma 密度计算有 GPU→CPU→GPU 往返。这些操作可能触发同步，但尚未测量它们是否占主要耗时。prepared DataLoader 本身会搬运 batch，不能把随后模型里所有 `.to(device)` 都算作重复 H2D。

此前日志中的 `data=1820, fwd=0, bwd=0` 与旧版关闭计时的口径一致，不能直接用来认定 CPU 解码是主瓶颈。`1-allocated/reserved` 也不是显存碎片率，不建议据此每步清空缓存。当前 trace 诊断不依赖这些旧字段。

## 源码和验证

诊断接入实际入口 `src/agenticwam/trainer.py`，并在 `agenticwam_model.py` 中加入范围标记，默认不开启 profiler。另修正训练循环的一处 DDP 分支：始终调用模型包装器的 `forward`，避免因为 DDP 不暴露 `training_loss` 而绕过梯度同步。ZeRO 仍通过 DeepSpeed engine 的 forward。

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/check_wait_diagnostics_distributed.py
```

验证覆盖默认关闭、梯度累积下的窗口边界、异常时部分导出、CPU profiler 输出、缺失 CUDA 事件处理、GPU 事件区间合并、CPU toy 训练的 loss/RNG/参数更新一致性，以及实际训练循环的双进程 DDP 权重同步。CPU 测试不能替代真实两卡 ZeRO-1 的 CUDA/CUPTI 和吞吐验证。
