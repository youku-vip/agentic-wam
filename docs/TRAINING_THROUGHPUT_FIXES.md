# 双卡训练等待：DeepSpeed 参数扫描与视频读取优化

本次交付包含两个独立的可选优化，默认都关闭。现场计时和堆栈支持优先验证 DeepSpeed 参数扫描缓存，再单独测试视频稀疏读取。安装器先校验 SHA256、备份旧源码，再安装 13 个源码/工具文件；保留现场的 `train.sh`、所有 YAML、`coding_v4`、模型路径和训练数据。

## 现场证据与适用范围

用户上传的两个 host 摘要均覆盖预热后 1 个 optimizer step、8 个 microstep，ZeRO-1、每卡 batch=4、梯度累积=8、world_size=2，有效 batch=64。

| 记录的主阶段总耗时 | rank 0 | rank 1 |
| --- | ---: | ---: |
| 获取 batch，含 H2D | 0.0849 s | 0.0877 s |
| forward | 5.4760 s | 5.9431 s |
| backward，含 DeepSpeed 内部操作 | 129.1843 s | 128.6554 s |
| 后续 clip/optimizer 调用 | 0.4294 s | 0.4254 s |
| 上述阶段合计 | 135.1746 s | 135.1116 s |

反向/DeepSpeed 占所列主阶段约 95%；所有 8 个 microstep 都慢，不仅是更新边界。该窗口的数据等待已经很小，但不能据此推断所有训练时段都没有数据瓶颈。host 模式没有采集 GPU 活动，不能把 `gpu_activity_available=false` 解读为 GPU 利用率为零。嵌套子阶段不能再加到这些主阶段上。

Accelerate 的 DeepSpeed backward 包装会在同步边界内部调用 engine.step，因此这里不能把整个 backward 都称为纯反向 CUDA 计算或梯度通信。

用户进一步提供的堆栈里，rank 0 的 autograd 线程多次位于：

```text
DeepSpeedZeroOptimizer.grad_handling_hook
  count_used_parameters_in_backward
    torch.autograd.graph._get_grad_fn_or_grad_acc
```

现场行号与本地 DeepSpeed 0.18.5 的实现相符：每个参数的梯度 hook 都扫描同一个完整参数列表。若有 P 个参与 hook 的参数张量，可能反复扫描约 P×P 次。这是明确的 CPU 开销线索；尚不能把 129 秒全部归因于它。

两卡拓扑为同 NUMA 下的 PIX，未显示 NVLink。仅凭此信息不能判断 PCIe/NCCL 占了多少耗时。本轮不同时修改通信 bucket、batch、checkpoint 或 NCCL 环境变量，避免失去对照。

## 安装

在已有项目中取出交付包，不需要切换分支或覆盖本地训练配置：

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11

git fetch origin codex/custom-data-review-files
git show FETCH_HEAD:deliveries/training-throughput-fixes-v1.zip > /tmp/training-throughput-fixes-v1.zip

STAGE_DIR=$(mktemp -d /tmp/agenticwam-throughput.XXXXXX)
"$PY" -m zipfile -e /tmp/training-throughput-fixes-v1.zip "$STAGE_DIR"
"$PY" "$STAGE_DIR/scripts/install_training_optimizations.py" \
  --bundle-dir "$STAGE_DIR" --project-root "$PWD"
```

安装器会打印备份目录 `.training-throughput-backups/<编号>`，其中 `installation.json` 记录替换和新增的文件。安装前已校验全部待安装文件，校验失败不会写入项目源码。这个包针对本仓库现有 trainer/loader 版本；如果现场自行修改过这些源码，安装前需查看差异，备份中会保留原文件。正在运行的训练进程不会热加载新源码，补丁在下一次启动生效。

## 先验证 DeepSpeed 参数扫描缓存

新开关为 `AGENTICWAM_CACHE_DS_GRAPH_COUNTS=1`，只针对匹配的 ZeRO-1/2 hook 实现生效，其他实现会打印 `skipped`。它在当前进程中替换该函数引用，不修改 site-packages。

计数依旧来自 DeepSpeed 原函数：按“当前 autograd graph task + hook 捕获的稳定参数列表”缓存，而不是用模型参数总数替代。不同反向图、条件分支、未使用参数和 checkpoint 内部反向图分别计数；缓存最多保存 64 个整数键/计数，不保留参数或计算图。并发首次查询允许重复扫描，避免持有 Python 锁进入 autograd。

保持现有 batch=4、GAS=8、workers=4、checkpoint 和 ZeRO JSON 不变。先在视频优化关闭的相同条件下对照开关 0/1；下面是启用后的启动命令：

```bash
export AGENTICWAM_CACHE_DS_GRAPH_COUNTS=1
export AGENTICWAM_SPARSE_VIDEO_DECODE=0
export AGENTICWAM_VIDEO_DECODER_CACHE_SIZE=0

export AGENTICWAM_WAIT_DIAG_MODE=host
export AGENTICWAM_WAIT_DIAG_STEPS=1
export AGENTICWAM_WAIT_DIAG_WARMUP=1
bash train.sh
```

启动应看到 `[ds-graph-cache] enabled`。每个日志间隔会记录累计 `original_scans` 和 `cache_hits`，W&B 对应 `perf/ds_graph_count_original_scans`、`perf/ds_graph_count_cache_hits`。host 摘要的 `metadata.ds_graph_count_cache_enabled` 应为 true。

对比两次预热后的 backward/DeepSpeed 总耗时、step/s、loss 是否有限，以及两卡是否持续推进。不要用单次 nvidia-smi 的瞬时读数评价提速。若仍慢，保留新摘要和采样栈，再定位剩余开销。回退只需设 `AGENTICWAM_CACHE_DS_GRAPH_COUNTS=0` 后重新启动。

正常长训时关闭诊断采集：

```bash
export AGENTICWAM_WAIT_DIAG_STEPS=0
bash train.sh
```

包中保留 v2 诊断实现，默认 host 模式，不会在训练线程调用曾导致长时间停顿的 `profiler.key_averages()`。不要为了计时重新启用旧版 profiler 实现。

## 再验证视频优化

现场视频只读基准的 16 个窗口中，dense 新建 decoder 共 1.7042 秒，sparse 新建共 0.6190 秒，sparse 复用共 0.4431 秒；抽选后的 9 帧全部 SHA256 一致。约 3.85 倍是单文件、热缓存、复用命中的解码基准结果，不是训练提速倍数。

实现现在只在图像路径提前选择原来的 `[0,4,8,...,32]`，随后处理 9 帧；state 原始窗口仍为 33、action 仍为 32，最终模型 proprio 仍按原逻辑取 32 帧。相机顺序、episode 的 shared-MP4 起点、编码 FPS 定位、padding、代码索引与归一化策略保持原逻辑。模型本来已经使用这 9 帧，因此模型视频 token 数没有减少，无需为此重新生成代码或重算 state/action 统计。

TorchCodec decoder 缓存仅在 DataLoader worker 的主线程、CPU 解码时保留，每个 worker 独立有界 LRU；父进程、辅助线程和 CUDA 解码不保留句柄。文件替换/大小或 mtime 改变会使旧缓存失效；解码异常时清除该文件句柄后沿用原来的 PyAV 回退。容量 8、2 卡各 4 worker 时最多保留 64 个 decoder 句柄；这不是预解码全部视频或预取队列大小。

先用现场 YAML 和现有统计做只读正确性检查：

```bash
PY=/usr/local/bin/python3.11
DATA_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30

CUDA_VISIBLE_DEVICES="" PYTHONPATH=src "$PY" scripts/verify_custom_video_decode.py \
  --root "$DATA_ROOT" \
  --code-cache "$DATA_ROOT/coding_v4/policy_code_idx_custom.pt" \
  --episodes 0 1087 2174 3261 --workers 2 \
  > /tmp/video_decode_equivalence.json
cat /tmp/video_decode_equivalence.json
```

工具对每个指定 episode 的首、中、尾采样，比较最终全部字段的字节哈希、实际样本索引和采样后 RNG 状态。默认使用现场 `configs/real_post_train/data/custom.yaml`；其他配置可指定 `--config`。要求 `passed=true`、`failures=[]`；同时查看 worker 快照的缓存命中数。`--workers 0` 故意不在父进程缓存 decoder，需验证缓存命中时使用 workers>0。工具不加载模型，不重算或写入统计，不输出图像和代码内容。

验证通过后单独开启视频优化，再与前一个已启用 DeepSpeed 补丁的运行对比：

```bash
export AGENTICWAM_SPARSE_VIDEO_DECODE=1
export AGENTICWAM_VIDEO_BACKEND=torchcodec
export AGENTICWAM_VIDEO_DECODER_CACHE_SIZE=8
export AGENTICWAM_VIDEO_DECODER_NUM_THREADS=1
export AGENTICWAM_WAIT_DIAG_STEPS=0
bash train.sh
```

这些环境变量可直接用于原有 YAML，无需替换现场配置。YAML 中四个新增选项为 null 时使用环境变量；显式非 null 配置优先。启动日志应出现 `sparse=True image_frames=9 state_frames=33 action_frames=32`。若反复出现 TorchCodec 回退警告，应先处理 backend 失败，再评价缓存效果。

当前 custom 图像变换为 ToTensor/Resize。以后增加依赖整个时间序列的随机增强时，需要重新执行正确性比较。视频帧间压缩依然可能需要解码中间依赖帧，33/9 不是承诺的解码倍数；随机跨文件训练的缓存命中率也可能低于单文件基准。

## 验证记录与限制

本地 CPU 环境通过 46 项测试。新增测试覆盖实际 DeepSpeed 计数函数的动态未使用参数、不同列表、重入 checkpoint、梯度一致性、缓存边界和参数释放；视频覆盖三相机不同起点、shared-MP4、非规则原始 timestamp、短 episode、首尾填充、非单位 stride、最终图像/action/proprio/code/padding 一致性，以及两个真实 DataLoader worker 的句柄隔离与复用。安装测试覆盖哈希失败不写入和保留本地配置。

额外 CLI 验证使用合成 v3 数据集、2 worker、3 episode 的 9 个点。计数 CPU 小图基准见 `docs/benchmarks/deepspeed_graph_count_cpu.json`，只证明重复扫描成本可复现，不能当作 GPU/ZeRO 训练提速倍数。本环境没有用户原始视频、真实模型权重或 GPU，尚未做端到端双卡训练验证。
