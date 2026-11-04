# Custom 数据视频解码检查

结论：有明确优化空间，优先测试图像路径提前抽帧；解码器复用的收益取决于真实文件命中率。当前只新增只读基准和审查结果，没有修改训练 loader、预处理、数据集或采样。现场此前两卡持续 0% 已由 py-spy 定位到诊断工具的 `key_averages()` CPU 后处理；已在提交 `9fc49e6` 修复。这次阻塞不能用来证明视频解码是训练主瓶颈。

## 当前路径

1. `BaseLerobotDataset` 为每路相机请求 33 个时间点；state 33 帧、action 32 帧。
2. v3 loader 按 episode 的视频文件、起始时间和编码帧率定位窗口，调用视频解码函数。
3. 解码函数返回 `[33,3,H,W]`、float32、范围 `[0,1]`。
4. `_get_image` 将其乘 255 转回 uint8；processor 的 `ToTensor` 又转换成 float32，再对全部 33 帧做 Resize。
5. `RobotVideoDataset._get` 最后才按 `[0,4,8,...,32]` 选出 9 帧。

因此，每路有 24 帧参与了输出张量构建和图像处理，随后被丢弃。按三路、424×240、float32 的原始返回张量计算，33 帧约 **115.29 MiB/样本**，9 帧约 **31.44 MiB/样本**。这只是返回张量大小，不是进程峰值内存，也不是 GPU 显存；模型当前已经只接收每路 9 帧，提前抽帧不会减少模型视频 token 数。

`torchcodec` 分支已经调用 `get_frames_at(indices=...)` 批量取帧，不能再将“从逐帧改成批量”作为新增收益。当前测试版本 TorchCodec 0.5 的 FFmpeg 线程默认是 1；`torchvision.io.VideoReader`/PyAV 路径的默认线程策略不同，需要分别测量。

## 最值得改的部分

### 提前采样图像帧

只为图像选择原窗口中的 9 个时间点，在转换和 Resize 前完成；继续保留原来的 state/action 窗口和代码索引。

这需要联动修改，不能只把 `num_frames: 33` 改为 `9`：

- 图像 delta timestamps 只选择原先的 `[0,4,...,32]`；v3 shared-MP4 的 episode 起始偏移、实际编码帧率规则继续保留。
- processor 当前用 `num_obs_steps=33` 检查图像 shape，需要单独定义图像步数，保留 state/action 的 33/32。
- `RobotVideoDataset` 当前还会再次下采样，必须同步处理，防止重复抽样、越界和 `image_is_pad` 错位。
- 对比 episode 开头、中间、结尾，以及短 episode 的边界填充；三相机顺序、最终图像、action、proprio、padding、全局帧索引、policy_code 都应保持一致。

仅改变图像读取/预处理且保持相同输入帧时，无需因此重新计算 state/action 的归一化统计。当前 custom 配置只有 ToTensor/Resize；若未来添加依赖整个时间序列的随机增强，还需另外检查抽样前后的增强一致性。

视频是帧间压缩，要求返回 9 帧仍可能必须解码关键帧及中间依赖帧，所以 33/9 不是可承诺的解码或训练提速倍数。PyAV 路径还会遍历 seek 后的中间帧；稀疏请求对不同 backend 的收益不相同。

### 在 worker 内复用解码器

当前两个 backend 每次调用都新建 reader/decoder。v3 多个 episode 可以共享一个 MP4；同一 worker 重复命中文件时，复用可能减少打开、读取头信息和初始化开销。

实际实现应使用有上限的 worker 内 LRU，避免在父进程创建 native decoder 后被 fork 继承；同一个 decoder 也不应被多个线程同时 seek。容量应根据文件数和随机采样命中率测试，不能无限保留所有 decoder。

本次基准只模拟一个保留 decoder，相当于单文件 100% 命中上限，未将这个缓存方案接入训练。合成样例中，缓存 dense 仅约 1.04 倍，低于提前稀疏请求的收益，不能据此保证真实随机文件访问有同样收益。

### backend、线程和存储

先确认是否真的使用 TorchCodec，还是反复失败后回退 PyAV。只有 backend 工作正常，速度对比才有意义。当前两卡每卡 4 worker 共 8 worker；TorchCodec 默认每 decoder 一线程，直接增加 worker 或线程不一定有效。现有 DataLoader 已开启常驻 worker、预取和 pinned memory。

网络存储下还要区分 seek/read 与 CPU 解码。优先用相同窗口做独立 loader 对照，不把缩短 GOP、重编码或导出大量图片当成默认步骤。DataLoader worker 中继续使用 CPU 解码；本项目现有函数也明确不支持在 worker 中直接初始化 CUDA 解码。

## 合成 AV1 测量

测试素材为本次生成的 60 帧、424×240、10fps、GOP=30 的 AV1 视频，**不是用户数据**。单 CPU 进程、TorchCodec、FFmpeg 一线程，16 个随机窗口，每场景预热 2 次。计时只覆盖现有解码调用，像素 SHA256 对照在计时之外。

| 场景 | 返回帧数 | 每窗口 p50 | 相对 fresh dense 的总耗时提速 |
| --- | ---: | ---: | ---: |
| 每窗口新建 decoder，dense | 33 | 47.89 ms | 1.00× |
| 每窗口新建 decoder，sparse | 9 | 24.09 ms | 1.96× |
| 单 decoder 复用，dense | 33 | 46.11 ms | 1.04× |
| 单 decoder 复用，sparse | 9 | 22.62 ms | 2.13× |

四个场景各 16 个窗口，抽选后的 9 帧 SHA256 均与 dense 基线相同。另有真实 CPU 解码单测覆盖随机窗口和 decoder 复用。完整数据见 `docs/benchmarks/video_decode_synthetic_av1.json`。

这些数值包含热 OS page cache、单个短文件及重叠窗口，不能外推成真实长 MP4、多 worker、跨文件随机训练的收益；也不包含 Resize、文本缓存、H2D、模型和通信。本环境没有用户原始视频，尚未测量真实训练提速。

## 在真实数据上复测

以下工具只读视频，不修改视频或训练配置。独立运行基准时应避免与正常训练争抢 CPU/存储：

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
DATA_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
VIDEO=$(find "$DATA_ROOT/videos" -type f -name '*.mp4' -print -quit)

PYTHONPATH=src "$PY" scripts/benchmark_video_decode.py \
  --video "$VIDEO" --windows 16 --warmup 2 --frames 33 --stride 4 --threads 1 \
  > /tmp/video_decode_benchmark.json
```

检查 `all_selected_frames_identical=true`，再比较四场景的 `window_seconds_p50/p95`。本工具明确使用 TorchCodec，不静默切换 backend；若现场 TorchCodec 加载失败，保留错误并使用已有 loader 的 PyAV 对照，避免将 backend 回退误认为缓存效果。

已有 raw loader 工具可比较随机跨文件读取（仍不包含完整训练预处理）：

```bash
PYTHONPATH=src "$PY" scripts/diagnose_custom_data.py loader \
  --root "$DATA_ROOT" --workers 4 --batch-size 4 --batches 32 --warmup 4 \
  --backend torchcodec > /tmp/loader_dense.json

PYTHONPATH=src "$PY" scripts/diagnose_custom_data.py loader \
  --root "$DATA_ROOT" --workers 4 --batch-size 4 --batches 32 --warmup 4 \
  --backend torchcodec --sparse-video > /tmp/loader_sparse.json
```

`--sparse-video` 是已有诊断工具的基准选项，不是训练开关。真实 loader 与模型端仍需前述联动修改后才可启用提前采样。
