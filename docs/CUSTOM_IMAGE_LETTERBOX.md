# Custom 图像等比例缩放与补边

需要 `256×448` 时使用新版 [256×448 适配包与配置命令](CUSTOM_IMAGE_256X448.md)。本页保留 v1 的 `224×320` 说明；v2 同时支持两种尺寸。

当前 custom 数据原图为 `[高=240, 宽=424]`。旧训练链路先通过 processor 的 `ToTensor → Resize([224,320])` 分别缩放宽高，再通过 RobotVideoDataset 的等比例 resize、中心裁剪和 `[-1,1]` 归一化。第一步的缩放比例分别为 `224/240≈0.9333`、`320/424≈0.7547`，物体相对变高约 23.7%；后续处理在相同尺寸下基本不改变像素，无法恢复原始比例。服务端默认 parity 路径也直接缩放到 `[224,320]`，与旧训练一致。

本补丁增加 `processor.image_resize_mode=letterbox`。它将 processor 中显式指定 `[高,宽]` 的 torchvision Resize 替换为统一的 LetterboxResize，保留原来的插值方式与 antialias 参数。对当前数据，处理结果为：

```text
240×424 → 等比例缩放至 181×320 → 上补 21 行、下补 22 行 → 224×320
```

缩放尺寸会按像素取整，几何比例误差仅来自这一取整。完整视野保留，补边使用黑色，在 `[0,1]` 中为 0、归一化后为 -1。同一 clip 的全部帧使用相同变换。三路相机分别处理，图像时间采样、相机顺序、代码标签和动作/state 归一化保持原有逻辑。模型输入尺寸和视觉 token 数保持现有配置。

## 安装

交付包为 `deliveries/custom-image-letterbox-v1.zip`，适配当前已安装四任务均衡采样 v2 补丁的项目。安装器校验 SHA256、备份 5 个待安装源码/工具文件，然后安装；不会覆盖 `train.sh` 和现场 YAML。包中的 YAML 仅供测试参考，不由安装器复制。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11

git fetch origin codex/custom-data-review-files
git show FETCH_HEAD:deliveries/custom-image-letterbox-v1.zip > /tmp/custom-image-letterbox-v1.zip

PATCH_DIR=$(mktemp -d /tmp/agenticwam-letterbox.XXXXXX)
"$PY" -m zipfile -e /tmp/custom-image-letterbox-v1.zip "$PATCH_DIR"
"$PY" "$PATCH_DIR/scripts/install_image_preproc_patch.py" \
  --bundle-dir "$PATCH_DIR" --project-root "$PWD"
```

安装器打印源码备份目录 `.image-preproc-backups/<编号>`。安装源码本身默认保持 stretch，以便旧模型继续使用旧预处理。正在运行的进程不会热加载修改。

## 启用训练和验证

下面的命令只把现场 YAML 中训练、验证两个 processor 的 `image_resize_mode` 设为 letterbox；其余文本、注释、换行、`coding_v4`、提示词 JSON 路径等配置全部保留。修改前生成同目录备份，重复执行不重复改写。

```bash
"$PY" scripts/configure_custom_image_preproc.py \
  --config configs/real_post_train/data/custom.yaml --mode letterbox
```

对应配置为：

```yaml
data:
  train:
    processor:
      image_resize_mode: letterbox
  val:
    processor:
      image_resize_mode: letterbox
```

保留现有 `raw_shape: [3,240,424]`、输出 `shape: [3,224,320]`、`video_size: [224,320]` 和 Resize 的 size，补丁会在 processor 初始化时替换 Resize 操作。该配置会进入训练输出目录的 `config.yaml`。

为采用新预处理的训练设置新的 `output_dir`，然后使用原有训练启动命令：

```bash
bash train.sh
```

启动日志应出现 `[image-preproc] mode=letterbox`。不需要重新生成 code 缓存、Cosmos 文本缓存或 state/action 统计。已经按 stretch 训练的模型需要在新预处理下微调并评测；更换图像几何并不等于已经验证模型精度提升。

## 服务端推理

训练与部署使用相同的 `custom.yaml`，设置：

```bash
export AGENTICWAM_EVAL_PREPROC=auto
```

随后使用原来的 `custom34_server.py` 启动命令，加载在 letterbox 配置下训练/微调的 checkpoint。服务端会根据 processor 的配置自动选择 letterbox，调用与数据集共享的 `transform_image`，再沿用视频归一化。启动日志应出现 `[preproc] mode=letterbox`。

如果环境中保留了旧的 `AGENTICWAM_EVAL_PREPROC=parity`、legacy 或 parity_crop，letterbox 配置会报错，避免训练与推理几何不一致。设为 auto 或清除此环境变量即可。训练与验证配置不同也会在 custom 服务启动时报错。

旧权重继续推理时应使用原来 stretch 配置。可以在服务端原启动命令后追加两条覆盖，而不改当前新训练配置：

```bash
--override '++data.train.processor.image_resize_mode=stretch' \
--override '++data.val.processor.image_resize_mode=stretch'
```

图像预处理配置不是从权重文件自动推断的，部署时应以对应训练目录的 `config.yaml` 为准。

## 检查与回退

纯 CPU 回归检查，不占训练 GPU、不加载模型权重。需保留上面的 PATCH_DIR 变量：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src:$PATCH_DIR/tests" \
  "$PY" -m unittest discover -s "$PATCH_DIR/tests" -p test_letterbox_image_preproc.py
```

测试使用合成图像和一个打包 Parquet/MP4 的 LeRobot v3 fixture，覆盖横竖图、补边位置、圆形比例、旧 stretch 兼容、训练/验证/实际部署方法逐像素一致，以及完整 loader 中动作/代码标签等非图像字段不变、dense/sparse 结果一致。另检查配置更新保留原文、重复执行和回退、安装校验失败不写入项目。

本次相关 CPU 回归共 50 项通过（另有 13 项参数化子检查），包括均衡采样、代码指标、视频解码和原安装器的回归检查。

在真实数据集上，沿用已有只读检查命令确认 sparse/dense 两条路径都在新预处理下保持一致：

```bash
DATA_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src "$PY" scripts/verify_custom_video_decode.py \
  --root "$DATA_ROOT" --code-cache "$DATA_ROOT/coding_v4/policy_code_idx_custom.pt" \
  --episodes 0 1087 2174 3261 --workers 2 \
  > /tmp/letterbox_video_equivalence.json
cat /tmp/letterbox_video_equivalence.json
```

本环境没有真实数据集和 GPU，未执行真实视频或双卡训练验证，也未验证机器人任务成功率。

回退图像配置即可恢复旧处理，新增源码可以保留：

```bash
"$PY" scripts/configure_custom_image_preproc.py \
  --config configs/real_post_train/data/custom.yaml --mode stretch
export AGENTICWAM_EVAL_PREPROC=auto
```

重启训练/服务使配置生效。回退时选择与 stretch 预处理对应的权重。
