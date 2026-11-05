# Custom 256×448：原图补边、训练与部署同步

本版本将最终模型输入适配为 `[高=256, 宽=448]`。原始数据为 `240×424`，采用 `letterbox + image_letterbox_scale_up=false`，原图不放大：上、下各补 8 行，左、右各补 12 列黑色像素。原始画面不裁剪、不拉伸，补边区域在视频 `[-1,1]` 归一化后为 -1。若实际输入大于目标尺寸，则先等比例缩小再补边。

训练、验证、服务端使用同一套 processor 图像变换。部署的最终 resize/crop 和 parity 路径从训练配置读取尺寸，已移除固定 `224×320` 限制。旧 `224×320` 配置继续可用。训练/验证尺寸、相机 shape、补边模式不一致时提前报错；processor 的 Resize size 与 shape_meta 不一致也会在初始化时报错。

## 下载并安装

适配当前已安装四任务均衡采样 v2 的项目，不要求先安装图像补边 v1。安装器先校验、备份，再安装 6 个源码/工具文件；服务端同时安装到 `deploy-custom/custom34_server.py` 和常用的 `deploy/custom34_server.py`。不覆盖现场 `train.sh` 和 YAML。压缩包中的 YAML 是参考配置，仅供测试和查看。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11

git fetch origin codex/custom-data-review-files
git show FETCH_HEAD:deliveries/custom-image-letterbox-v2-256x448.zip > /tmp/custom-image-letterbox-v2-256x448.zip

PATCH_DIR=$(mktemp -d /tmp/agenticwam-image256.XXXXXX)
"$PY" -m zipfile -e /tmp/custom-image-letterbox-v2-256x448.zip "$PATCH_DIR"
"$PY" "$PATCH_DIR/scripts/install_image_preproc_patch.py" \
  --bundle-dir "$PATCH_DIR" --project-root "$PWD"
```

安装器打印 `.image-preproc-backups/<编号>` 源码备份位置。不要直接将整个解压目录覆盖到项目中。

## 将现场配置切换为 256×448

```bash
"$PY" scripts/configure_custom_image_preproc.py \
  --config configs/real_post_train/data/custom.yaml \
  --mode letterbox --size 256 448 --no-upscale
```

配置工具先备份原 YAML，然后同步更新：

- train/val 的 `video_size`。
- 三路相机的输出 `shape: [3,256,448]`，包括 processor 使用独立 metadata 的情况。
- 两个 processor 的 train/val Resize size。
- 两个 processor 的 `image_resize_mode: letterbox`、`image_letterbox_scale_up: false`。

原始 `raw_shape: [3,240,424]`、`coding_v4`、提示词 JSON 路径、归一化文件、其余字段、注释与换行均保留。重复执行不重写文件。只改 mode、不指定 size 时仍兼容 v1 的用法。

原有 `train.sh` 继续使用 `task=posttrain_custom data=custom` 即可；配置工具修改的是其实际读取的 custom.yaml。为新分辨率设置新的 `output_dir` 后启动：

```bash
bash train.sh
```

启动日志应出现 `[image-preproc] mode=letterbox scale_up=False`，输出 image_shapes 应为三路 `(3,256,448)`。训练运行目录的 `config.yaml` 会记录这套配置。现有 checkpoint 可以作为微调起点，但应在新图像几何下继续训练和评测后部署。

仓库另有完整参考配置 `configs/real_post_train/data/custom_256x448.yaml`，支持 `task=posttrain_custom data=custom_256x448`。参考文件的数据/code 路径需要自行匹配现场；已有项目优先使用上述配置工具以保留本地设置。安装器不复制此参考 YAML。

视频与 episode 数据无需重新编码；code 缓存、Cosmos 文本缓存、state/action 统计无需重算。动作 horizon=32、三路相机、每路 9 张训练图像和任务均衡采样逻辑保持原有配置。

## 部署

沿用服务端原启动命令、相同 `posttrain_custom` 配置和在 256×448 下训练/微调的权重，并设置：

```bash
export AGENTICWAM_EVAL_PREPROC=auto
```

服务端启动日志应包含：

```text
[preproc] mode=letterbox video_size=[256, 448] scale_up=False
```

客户端继续上传原始 `240×424` 图像（`--env.im_h=240 --env.im_w=424`），补边在服务端完成。部署也要能读取对应训练的图像配置；尺寸不能从权重文件自动推断。不要仅对旧权重切换服务端尺寸就当成模型已适应新图像几何。

## 验证与资源开销

纯 CPU 检查，不加载真实模型权重、不占训练 GPU。保留上述 PATCH_DIR 变量：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src:$PATCH_DIR/tests" \
  "$PY" -m unittest discover -s "$PATCH_DIR/tests" -p test_letterbox_image_preproc.py
```

测试覆盖原始像素逐字节保留、上下/左右补边、训练/验证/实际部署方法逐像素一致、完整 v3 loader 的 `video=[3,27,256,448]`、非图像字段不变、sparse/dense 一致、224×320 兼容、配置备份/回退与安装校验。还使用真实 Qwen3-VL 实现的小型随机视觉网络验证 336 个前缀 token 和 merger 反向梯度，使用实际 Cosmos tokenizer 与 multiview RoPE 验证新 latent 空间尺寸；这些不是完整预训练模型的 GPU 训练测试。

本次相关 CPU 回归 61 项通过，另有 22 项参数化子检查。Hydra 的 `task=posttrain_custom data=custom_256x448` 配置组合也通过了只读检查。

实际数据检查沿用已有工具：

```bash
DATA_ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src "$PY" scripts/verify_custom_video_decode.py \
  --root "$DATA_ROOT" --code-cache "$DATA_ROOT/coding_v4/policy_code_idx_custom.pt" \
  --episodes 0 1087 2174 3261 --workers 2 \
  > /tmp/video_equivalence_256x448.json
cat /tmp/video_equivalence_256x448.json
```

预期 `passed=true`，示例 video shape 为 `[3,27,256,448]`。

每路代码视觉 token 从 70 增至 112，三路前缀从 210 增至 336。每个样本的视频 token 从 630 增至 1008。视觉 token 数量增加 60%，整体耗时与峰值显存需要实测，不能按这一比例直接推算。

本环境没有真实数据、完整模型权重和 GPU，未执行真实双卡训练或机器人成功率评测。若新尺寸显存不足，可先保留当前任务均衡 batch=4/GAS=8，恢复 mixed-attention checkpoint：

```bash
export AGENTICWAM_MOT_NO_CKPT_EVERY=0
```

同时保持 `model.use_gradient_checkpointing=true` 和 `model.mot_checkpoint_mixed_attn=true`。任务均衡采样要求每卡 batch 是任务数 4 的倍数，不能在保持当前严格均衡规则时直接改为 batch=2。

恢复旧 `224×320` stretch 配置时：

```bash
"$PY" scripts/configure_custom_image_preproc.py \
  --config configs/real_post_train/data/custom.yaml \
  --mode stretch --size 224 320 --upscale
export AGENTICWAM_EVAL_PREPROC=auto
```

选择与旧预处理对应的权重，重启训练/服务。也可从配置工具打印的 YAML 备份恢复完整原配置。
