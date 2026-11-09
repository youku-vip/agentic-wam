# 保存生成图像并对比专家 32 步动作

此功能使用数据集中的专家动作 `action[t:t+32]` 作对照。模型只接收第 t 帧的观测图像、state34 和任务文本。专家未来动作仅用于推理后的指标与绘图，不作为模型输入。

支持清洗前、清洗后的单个 LeRobot v3 数据集根目录。动作比较使用反归一化后的绝对 EEF20；另输出以同一个 state[t] 为基准的相对 EEF64 对比。不会把归一化动作和原始坐标混在一起比较，也不会用后续 observation.state 冒充专家 action。

## 已安装 v3 时，手动更新四个文件

| 包内文件 | 操作 |
|---|---|
| deploy-custom/custom34_server.py | 替换服务端入口 |
| robotwin/deploy/deploy_policy.py | 替换推理封装，获取每次生成的视频帧 |
| src/agenticwam/utils/custom34_review.py | 新增导出、误差统计与绘图工具 |
| deploy-custom/review_custom34_chunks.py | 新增数据集离线评测脚本 |

客户端文件与 v3 相同。本次更新继续使用 v3 的 deployment_text_cache.py、图像 processor、CustomUnifiedEEFTransform 和 v3/episode_reader.py；包内也携带这些依赖，尚未安装时按相对路径复制。已经完成 v3 安装时无需再次更改训练配置。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
PKG=/tmp/custom34-deployment-v4
curl -fL 'https://github.com/youku-vip/agentic-wam/raw/refs/heads/codex/custom-data-review-files/deliveries/custom34-deployment-v4.zip' -o /tmp/custom34-deployment-v4.zip
"$PY" -m zipfile -e /tmp/custom34-deployment-v4.zip "$PKG"
```

手动备份并复制表中四个文件，目标根目录为 `/data/algorithm/agentic-wam`。完整备份安装器也保留在包内，适用于首次安装。

绘图使用 Matplotlib，数据读取使用 PyArrow 和 PyAV，图片保存使用 Pillow；保留原训练环境的 PyTorch/CUDA 版本。这几个依赖缺失时安装：

```bash
"$PY" -m pip install matplotlib pyarrow av Pillow
```

## 启动服务端，保存每次生成的图片

将 RUN_DIR 和 CKPT 改成实际评测的训练 run 与 checkpoint。下面仍用此前讨论的 run、step_004000 示例。运行离线评测时，暂停真实机器人客户端请求，使用此独立评测服务。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
RUN_DIR="$PWD/runs/air_fryer_zero2_10000_p012_v4_1"
CKPT="$RUN_DIR/checkpoints/weights/step_004000.pt"
export AGENTICWAM_DEPLOY_DTYPE=bf16
export AGENTICWAM_EVAL_PREPROC=auto
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/src" "$PY" deploy-custom/custom34_server.py \
  --train-config "$RUN_DIR/config.yaml" \
  --ckpt "$CKPT" \
  --device cuda:0 \
  --num-inference-steps 10 \
  --review-dir ./custom34_reviews \
  --code-log-dir ./custom34_code_logs \
  --host 0.0.0.0 --port 8765
```

默认每次 `/act` 都保存图像；无需再设置 AGENTICWAM_SAVE_ALL_REPLANS，也不只保存 episode 第一次推理。`--no-review` 可以关闭此功能。

32 是动作 chunk 的步数。若训练配置为 num_frames=33、action_video_freq_ratio=4，模型解码视频通常每路 9 帧：相对观测步 0、4、8、…、32；三路共 27 张。第 0 帧是条件观测的重建，后面才是未来预测。实际数量、相机顺序和帧步偏移写入 generated/frames.json 和 metadata.json。

每次保存路径独立，服务重启也不会覆盖前一次：

```text
custom34_reviews/<server_id>/episode000001_call00000001/
  observation_raw/                  原始三路输入 PNG
  observation_model_input/          实际预处理后的三路输入 PNG
  generated/head/frame_000.png ...  头部生成帧
  generated/left/frame_000.png ...  左腕生成帧
  generated/right/frame_000.png ... 右腕生成帧
  generated/contact_sheet.png       三路生成图像总览
  generated/frames.json             每张图的相机和时间步
  chunks.npz                       state34、预测 action20、预测 relative64
  metadata.json                    指令、代码、checkpoint、统计哈希、请求 ID
  index.html                       浏览入口
```

视频解码、PNG 写盘和对比绘图会增加请求耗时与磁盘占用。保存失败会在日志和响应的 review.status 中明确标为 error；不会把已经计算出的动作变成空动作。

## 从数据集取观测，评测 32 步动作

另开终端执行。ROOT 可以换成清洗前的原始根目录；无需修改脚本。任务文本从该 episode 的 task_index 和 meta/tasks 中读取。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30_translation_v2
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" "$PY" deploy-custom/review_custom34_chunks.py \
  --root "$ROOT" \
  --server http://127.0.0.1:8765 \
  --episodes 0 1 2 3261 \
  --start-frames 0 \
  --out ./custom34_expert_review_01
```

脚本本身不加载模型、不使用 GPU、不发机器人控制命令。它在每个 chunk 前 reset 模型，发送专家第 t 帧的观测，取得预测，再下载服务端保存的整套图片和对比报告；最终可以在脚本所在机器打开 `custom34_expert_review_01/index.html`。

多起点可写 `--start-frames 0 8 16`，每个所选 episode 都评测这些起点。默认必须有完整真实 32 步；尾部不足时报告失败，不跨 episode 取值。若需要与训练的尾部重复策略一致，可显式添加 `--padding repeat`；重复补齐的行会标记为无效，并排除在误差统计之外。

若训练使用 global_sample_stride>1，则脚本按照服务端声明的 stride 取 action[t+k*stride]，时间轴使用对应的有效动作频率。默认 stride=1 时就是 action[t:t+32]。

脚本通过 PyAV 按 PTS 读取 RGB 帧。视频查询使用 `episode videos/from_timestamp + frame_index / encoded_fps`，可处理打包 MP4 和清洗后视频软链接。每路观测的目标/实际解码时间保存到 video_alignment.json，避免把 Parquet 中的原始不等间隔采集时间误当成 MP4 帧时间。

输出中的每个 episode/frame 子目录包含：

| 文件 | 内容 |
|---|---|
| action20_all_dimensions.png | 20 个维度各一个子图，专家蓝线、预测橙线 |
| dimensions20/*.png | 20 维单独的对比曲线 |
| physical_errors.png | 左右臂 XYZ 欧氏距离误差（mm）、旋转测地角误差（度）、夹爪曲线 |
| relative64_all_dimensions.png | 同一个起始状态下的 64 维相对动作比较；灰色面板为不用的手部占位维度 |
| per_dimension_metrics.csv | 20 维分别计算 MAE、RMSE、最大绝对误差、偏差及单位 |
| action20_comparison.csv | 每步每维的预测值、专家值和有符号误差 |
| metrics.json | 全部指标、20 维与 64 维误差、有效行数量、左右夹爪准确率 |
| chunks.npz | 预测/专家 action20、relative64、当前 state34、有效 mask |
| index.html | 生成图像和动作曲线的浏览页面 |

左臂 action20 的 0:3 是 XYZ、3:9 是 rotation6D；右臂 9:12 是 XYZ、12:18 是 rotation6D；18、19 是左右夹爪。夹爪按客户端相同规则以 >0.5 判为 open。旋转误差通过旋转矩阵计算，不直接相减可能跨越 ±180° 的 Euler 角。

这是从专家观测出发的离线 chunk 预测误差，不能单独作为闭环抓取成功率。对比同一 checkpoint 时，保持提示词、统计文件、推理步数及评测起点一致；如果所选 episode 参与过训练，结果反映的是这些样本上的拟合表现。

## HTTP 扩展（客户端不必修改）

普通 /act 请求仍然只有 instruction/state/images 等字段，自动保存预测图像和动作。离线脚本额外发送 reference={action20:[32,20], valid_mask:[32], fps:10, source:{...}}，此字段只用于推理后的对比。

响应增加 review={status,id,path,download_url,has_reference}。GET /health 包含 review_enabled 和保存位置。GET /reviews/<id>/bundle.zip 可下载该次保存的图片、数组与报告。离线脚本会核对请求 ID、服务实例 ID、下载的预测动作及专家动作，防止混入另一请求的结果。
