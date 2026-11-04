# 定位训练中的 GPU 等待（v2）

目标是用当前两卡训练的时间线，区分数据供应、CPU/GPU 同步和卡间通信等待。当前环境没有真实 GPU、数据集和模型权重，因此这里提供的是诊断能力，尚未测出用户机器上的主瓶颈或提速比例。

v2 默认使用 **host 模式**，不启动 `torch.profiler`；先获取主机阶段、实时进度与 Python 线程栈。现场 `py-spy` 已确认 v1 的两个训练主线程都停在 `WaitDiagnostics.close → profile.key_averages → _parse_kineto_results`：GPU 采集已经完成，CPU 上的事件解析拖住了训练。这次持续 0% 由诊断后处理造成，不能据此认定视频解码或 NCCL 是原始训练的主瓶颈。

v2 在所有模式下移除了训练主线程上的 `key_averages()` 和原始 trace JSON 解析，显式 profiler 模式只停止采集、导出原始 trace 和轻量阶段摘要。分析移到独立离线进程；不再在线重建 Kineto FunctionEvents。GPU/CUPTI 模式仍未在用户硬件实测，默认使用 host。

如果当前训练已经卡住，先保留末尾日志和进程状态。在两个训练进程退出后，下一次沿用原训练/恢复配置设置 `AGENTICWAM_WAIT_DIAG_STEPS=0` 可关闭本工具。只在终端修改环境变量不能改变已经运行的进程。若关闭后仍卡住，应继续查看首 batch、worker 和通信，而不是反复开启 CUDA profiler。

## 启用

更新本分支源码后，沿用已能正常运行的两卡 `train.sh`，在启动前执行：

```bash
export AGENTICWAM_WAIT_DIAG_STEPS=1
export AGENTICWAM_WAIT_DIAG_WARMUP=0
export AGENTICWAM_WAIT_DIAG_MODE=host
export AGENTICWAM_WAIT_DIAG_TIMEOUT=90
bash train.sh
```

`AGENTICWAM_WAIT_DIAG_STEPS` 默认为 `0`，即关闭。上例从本次启动或恢复的第一个 optimizer step 开始记录，梯度累积 8 时覆盖 8 个 microstep，并包含优化器更新。首 batch 阻塞时应使用上例 `WARMUP=0`；显式设为 `1` 可只汇总预热后的阶段耗时。warmup 期间也会更新进度文件，但不混入采样窗口的耗时统计。需要更长窗口时可设 `STEPS=2`。

`TIMEOUT=90` 表示从第一次准备取 batch 起，每 90 秒记录一次 Python 线程栈；正常但缓慢的阶段也可能被记录，不能单凭 `Timeout` 字样认定死锁。采样完成或异常退出时会取消定时输出。watchdog 在初次 DataLoader iterator 创建后才启动，因此卡在 iterator 创建时可能只有进度文件，没有线程栈。

此开关独立于 `AGENTICWAM_PHASE_TIMING`。原脚本将 `PHASE_TIMING` 设为 `0` 也能记录。采样期间会临时跳过旧阶段计时插入的 barrier 和 CUDA synchronize。本工具不主动新增分布式 barrier 或 CUDA synchronize；host 模式增加本地进度文件与计时开销，显式 profiler 模式还增加 profiler 起停和导出开销，不应拿诊断窗口吞吐当正常训练速度。

采样完成会自动结束诊断、导出文件，然后继续训练。无需改 batch、学习率、loss 或数据采样。当前已经运行的旧进程不会因更新磁盘文件自动获得新诊断能力，应在下一次正常启动/恢复时开启。诊断完成后，未来启动可关闭：

```bash
unset AGENTICWAM_WAIT_DIAG_STEPS
unset AGENTICWAM_WAIT_DIAG_WARMUP
unset AGENTICWAM_WAIT_DIAG_MODE
unset AGENTICWAM_WAIT_DIAG_TIMEOUT
```

不要用 `CUDA_LAUNCH_BLOCKING=1` 跑性能对比，它会改变执行的同步方式。短时诊断期间也应保留原有 checkpoint 和 batch 设置，便于对照；如要排除生成评测，需在训练命令内部显式设 `code_eval_every=0`，并记录这个变化。

## 输出

host 模式在当前 `output_dir` 下为两张卡分别写摘要：

```text
wait_diagnostics/
  rank0-step000001-<唯一时间戳>.summary.json
  rank1-step000001-<唯一时间戳>.summary.json
```

`step` 是采样开始时已经完成的 optimizer step 数，上例 `WARMUP=0` 从 step 0 开始；恢复训练时使用恢复后的步数。摘要包含实际 world size、ZeRO stage、每卡 batch、梯度累积、每卡 worker 数。host 模式没有算子级 profiler 表和 GPU 时间线，`trace_file=null`、`gpu_activity_available=false` 是正常结果，不表示 GPU 没有工作。

即使尚未完成一步，也能在本地 `/tmp` 查看各 rank 的状态：

```bash
find /tmp -maxdepth 2 -type f \
  -path '/tmp/agenticwam-waitdiag-rank*/progress.json' -print -exec cat {} \;
find /tmp -maxdepth 2 -type f \
  -path '/tmp/agenticwam-waitdiag-rank*/stacks.log' -print
```

每次运行使用独立目录，按最新控制台路径及进程 PID 选取当前两份文件。`progress.json` 包含 `pid`、`optimizer_step`、`phase_stack` 和更新时间：`dataloader_iterator_start` 表示创建 iterator/worker，`data_next_including_h2d` 表示取 batch/搬运，`forward` 的子阶段表示模型内部工作。末尾 `stacks.log` 在采样持续超过指定秒数时记录 Python 线程栈。不要对未注册处理器的训练进程发送 `SIGUSR1`，它可能直接终止进程。

只有在 host 模式稳定运行、需要更细时间线时，才考虑显式 `AGENTICWAM_WAIT_DIAG_MODE=cpu` 或 `cuda`。这两种模式会启动 torch.profiler，另外生成 `.trace.json`，但不在线生成算子聚合表，摘要标记 `analysis_deferred=true`。CUDA 模式尚未在用户硬件实测；若出现采集启动/停止阻塞，进度事件会保留 `profiler_start_begin` 或 `profiler_stop_begin`。可在本地用 Perfetto 或支持 Chrome trace 的工具打开 trace。

v1 在调用 `key_averages()` 之前已经导出了 trace，原文件可以保留。离线分析 v1/v2 的原始文件：

```bash
PYTHONPATH=src /usr/local/bin/python3.11 scripts/summarize_wait_trace.py \
  ./runs/<当前运行目录>/wait_diagnostics/rank*.trace.json \
  > /tmp/wait_trace_summary.json
```

离线脚本只读取 Chrome JSON，计算阶段总耗时、CUDA runtime 调用耗时与 GPU 活动覆盖；不创建 profiler、不调用 `key_averages()`，不构建模型或初始化 CUDA。它仍需要 CPU 和内存来读 trace，建议在当前被后处理拖住的任务正常结束后运行。结果是包含两卡的一个 JSON；host 阶段为 inclusive 时间，并非算子 self time。

摘要中的 `host_phases` 是主机侧耗时，包含 Python 操作、CUDA 提交和阻塞；不是纯 GPU 计算时间。嵌套阶段会重叠，不能直接将所有行相加：

| 阶段 | 覆盖范围 |
| --- | --- |
| `dataloader_iterator_start` | 初次创建 DataLoader iterator/worker，仅做进度标记；在完整 optimizer step 耗时窗口之前。 |
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

外层 `forward` 覆盖模型整个前向，包括其中的等待。显式 profiler 模式的 trace 还会记录 CUDA runtime、NCCL、Python 调度等事件；窗口没有覆盖到日志或 checkpoint 的触发步时，不会测到它们。host 模式只能定位主机停留阶段，不能分离该阶段里的纯 GPU 计算时间。

## 如何判断

| 时间线现象 | 下一步 |
| --- | --- |
| GPU 无 kernel/复制事件，主进程长时间处于 `data_next_including_h2d` | 继续区分 worker 读盘、视频解码和 H2D；对比本地存储、worker 数和 CPU 配额。 |
| GPU 空白与代码 tokenization、CPU roundtrip 或 Python 调度重合 | 优先处理对应的主机工作，例如缓存 tokenization、消除 CPU 往返或合并指标同步。 |
| 一张卡尚在准备数据，另一张卡已进入长时间 NCCL kernel | 先解决 rank 进度不一致；不能仅凭 NCCL 时间长就认定互联带宽差。 |
| 两卡均已进入通信，NCCL kernel 持续很长 | 检查 GPU 拓扑、通信带宽与 bucket 设置；保留可比较的模型和 batch。 |
| GPU kernel 连续运行，CPU 卡在 `cudaStreamSynchronize` | CPU 在等待 GPU 完成计算，这本身不能证明 GPU 空闲；查看 kernel 内容和覆盖区间。 |

显式 CUDA profiler 模式中，`timeline` 的 coverage 来自 trace 中 GPU kernel、复制和 memset 区间的并集，不会将并发事件重复相加。NCCL kernel 单独统计，因为它也可能在等待。`uncovered_trace_window_fraction` 是该捕获窗口没有记录 GPU 活动的比例，不是硬件 SM 利用率，也不是完整训练的空闲比例。

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

验证覆盖默认关闭、梯度累积下的窗口边界、异常时部分导出、host 模式不调用 profiler/CUDA synchronize、预热期间进度、定时 Python 堆栈、显式 CPU profiler 输出、禁止在线调用 native key_averages/读取原始 trace、离线 JSON 分析、缺失 CUDA 事件处理、GPU 事件区间合并、CPU toy 训练的 loss/RNG/参数更新一致性，以及实际训练循环的双进程 DDP 权重同步。CPU 测试不能替代真实两卡 ZeRO-1 的 CUDA/CUPTI 和吞吐验证。
