# v4.1：生成视频非有限值与动作对比导出修复

报错中的 `(1,3,27,256,448)` 符合三路相机、每路 9 帧的布局。v4 把形状检查和 `np.isfinite()` 放在同一条件里，信息只显示形状，容易误导。根据该版本的判断条件，这个形状触发报错说明生成视频中包含 NaN 或 Inf。

已有日志可以证明导出前的动作20通过了有限值检查，不能仅凭视频异常认定动作也无效；也不能由此断言动作预测准确。另一个 `episode 0 frame 16: only 31 real steps` 是独立的窗口长度问题。

## 手动更新五个文件

已安装 v4 时，备份后按相对路径更新：

| 文件 | 操作 |
|---|---|
| deploy-custom/custom34_server.py | 替换 |
| robotwin/deploy/deploy_policy.py | 替换 |
| src/agenticwam/utils/custom34_review.py | 替换 |
| deploy-custom/review_custom34_chunks.py | 替换 |
| src/agenticwam/utils/review_vae_decode.py | 新增 |

客户端不变。模型权重、统计文件及训练配置继续使用原来的配套版本。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
PKG=/tmp/custom34-deployment-v4.1
curl -fL 'https://github.com/youku-vip/agentic-wam/raw/refs/heads/codex/custom-data-review-files/deliveries/custom34-deployment-v4.1.zip' -o /tmp/custom34-deployment-v4.1.zip
"$PY" -m zipfile -e /tmp/custom34-deployment-v4.1.zip "$PKG"
```

替换的是你实际启动的 custom34_server.py 路径。复制上述五个文件后，重启服务端。

## 服务端参数

在原启动命令中使用：

```bash
--vae-decode-dtype fp32 --review-dir ./custom34_reviews_fp32 --port 9654
```

fp32 是 v4.1 的默认解码模式。它在导出图像时暂时把 VAE 浮点参数和注册 buffer 转成 float32，并在解码内部禁用外层 bf16 autocast；逐相机解码后恢复精确的原始参数存储与 dtype。动作专家、视频去噪、观测编码方式、归一化及 checkpoint 不作更改。

此前推理方法有外层 autocast，仅把传入 latent 转成 float32 不足以保证卷积以 FP32 运行，因此需要同时控制解码权重和局部 autocast。FP32 可以排查降低精度造成的解码异常，但尚未在用户 GPU/checkpoint 上确认根因；如果 latent 本身已有 NaN，升高解码精度不能把它修复。

需要对照原解码行为时可用 `--vae-decode-dtype native`。不会把整个模型改成 FP32，也不需要重算数据统计值。

## 先跑一个完整窗口

使用之前同一个 ROOT，先仅评测起点 0，并选择新的输出目录：

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" deploy-custom/review_custom34_chunks.py \
  --root "$ROOT" \
  --server http://127.0.0.1:9654 \
  --episodes 0 \
  --start-frames 0 \
  --out ./custom34_expert_review_fp32_01
```

ROOT 是原先评测的数据集根目录；不要让它在新终端中为空。确认单样本后再增加 episodes 与 start-frames。

若继续评测起点 `0 8 16`，episode 0 总共 47 帧，从 16 开始只能取到 31 帧。可以去掉起点 16，或显式加 `--padding repeat`，此时重复行会被标记并排除于误差指标。

## 输出与诊断

即使生成图像异常，仍先保存 chunks.npz、metrics.json、CSV，再生成动作曲线。能够输出的正常帧按原值保存；含 NaN/Inf 的帧不伪装成有效 PNG，也不通过 nan_to_num 变成黑图。总览相应位置标为 INVALID。

- metadata.json 的 video_decode：latent 和解码像素的形状、dtype、NaN/±Inf 数量、有限值范围和解码精度。
- generated/video_diagnostics.json：像素统计与无效帧数量。
- generated/frames.json：每个帧槽位的保存状态，无效帧的 file=null。
- generated/nonfinite_video.npz：存在非有限值时保留实际原始视频数组。
- action20_all_dimensions.png、dimensions20/*.png、relative64_all_dimensions.png：动作对比图独立保存。

如果 latent 有限、像素非有限，问题位于 VAE 解码或其输入幅值/权重路径；若 latent 已非有限，则需检查上游视频去噪。新诊断用于区分这两种情况，不能在没有数据时确认是哪一种。

响应 review.status=partial 表示动作报告已保存，但图像有缺失或无效帧。评测脚本会下载这些报告，并在 summary.json 中分别记录 image_warning_chunks、failed_chunks。退出码 2 表示动作对比完成但有图像警告；退出码 1 表示有窗口/推理/报告失败；0 表示全部完成。

旧版失败可能发生在动作数组写入之前，因此旧失败目录不一定可以恢复完整对比，需要重新推理。
