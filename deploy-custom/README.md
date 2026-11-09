# Custom34 服务端与客户端部署 v4.1

v4.1 修复非有限生成视频阻断动作对比，增加局部 FP32 VAE 解码与诊断，升级方法见 [REVIEW_NONFINITE_FIX.md](REVIEW_NONFINITE_FIX.md)。

适用于本项目的双臂夹爪自定义数据：raw state34 → unified EEF64 → 模型相对动作64 → absolute EEF action20。原始、清洗后的数据都通过对应训练 run 的配置和统计值部署。代码基于本次上传的旧服务端和 RealMan/ZMQ 客户端修改。

新增每次推理生成图像保存与专家 32 步动作逐维对比，完整命令见 [CHUNK_REVIEW.md](CHUNK_REVIEW.md)。

默认同步推理，RTC 关闭；夹爪按模型动作执行（>0.5 为 open，否则 close）。默认无 15 步锁爪、无闭合度低于 50 后自动开爪回位。客户端传原始 RGB 图像，服务端按训练 processor 的尺寸和模式处理；256×448 letterbox 的 checkpoint 会继续使用该设置。部署不修改训练配置。

## 1. GPU 服务端安装

保留现有训练环境及模型依赖。此包没有模型权重、训练数据或技能 JSON。

```bash
cd /data/algorithm/agentic-wam
PY=/usr/local/bin/python3.11
PKG="$PWD/custom34-deployment-v4.1"
curl -fL 'https://github.com/youku-vip/agentic-wam/raw/refs/heads/codex/custom-data-review-files/deliveries/custom34-deployment-v4.1.zip' -o /tmp/custom34-deployment-v4.1.zip
"$PY" -m zipfile -e /tmp/custom34-deployment-v4.1.zip "$PKG"
"$PY" "$PKG/deploy-custom/install_custom34_deployment.py" --server-root "$PWD"
```

安装器检查压缩包文件哈希，先备份全部将替换的文件，再安装十个服务端文件。备份清单路径会打印为 `BACKUP .../manifest.json`。它不覆盖 YAML、train.sh、数据或模型文件。

## 2. GPU 服务端启动

`RUN_DIR` 指向此次要部署的训练 run；`CKPT` 必须选该 run 实际保存的 weights 文件。下面用已讨论的 run 和 step_004000 举例，换成你要评测的 checkpoint。

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
  --vae-decode-dtype fp32 \
  --code-log-dir ./custom34_code_logs \
  --host 0.0.0.0 --port 8765
```

默认不加 `--fast-joint`、`--rtc`，先使用同步模式。一个服务端实例配一个客户端，不要让多个机器人或负载均衡副本共用 episode 状态。

配置与归一化读取顺序：

- `--train-config` 直接读取训练 run 保存的已解析配置，不要求当前仓库还有 posttrain_custom1/custom1 分组文件。
- 统计值优先使用显式 `--dataset-stats`，其次是 config.yaml 同目录的 dataset_stats.json，再次是配置中的 train.pretrained_norm_stats。必须与 checkpoint 配套。
- 技能提示词读取 train.skill_prompt_map 指定的 JSON，按当前 instruction 查找；完全复用训练的 skill_prompt_only/code_instruction_mode。不会临时加入训练没见过的新 API 说明。
- 文本嵌入默认复用 train.cosmos_text_cache_dir，按训练时包裹后的完整提示词计算 SHA256 查找。四种任务的英文指令须与训练一致。找不到缓存时报错，不替换为其他任务。
- 如果迁移过文件，可追加 `--dataset-stats /新的/统计.json`、`--skill-prompt-map /新的/skill_prompt_by_instruction.json`、`--text-cache-dir /新的/cosmos_text_embeds_cache`。这些应是训练文件的原样副本。
- 如果需要新任务文本，可预计算对应嵌入，或显式使用 `--text-mode encoder --override paths.text_encoder=/完整/Cosmos-Reason1-7B`，后者另加载文本模型，显存开销明显增加。
- 其他权重迁移用重复的点路径参数，例如 `--override paths.code_model=/新路径/SDAR-1.7B-Chat`。已解析配置内部的 model.xxx 路径也可能需要同步改；以保存的配置实际字段为准。

旧的 `--task posttrain_custom --dataset /训练数据根目录 --override data=custom` 方式仍保留，但必须有对应 Hydra 配置文件。使用 --train-config 时不混用 --task/--dataset。

查看服务状态：

```bash
curl -f http://127.0.0.1:8765/health
```

检查 state_dim=34、action_dim=20、chunk_size=32、action_type=absolute_eef20；对于相应训练配置还应看到 video_size=[256,448]、image_resize_mode=letterbox，以及统计文件 stats_sha256。

## 3. 机器人客户端安装

在能执行原来 `python -m environment.rtc_client_agenticwam` 的机器人项目目录操作。仍使用原有的 config_tools、RealManGym、trans_tools、IKSolver、Draccus 环境；不要在 GPU 项目目录创建另一套机器人接口。

```bash
PY=/usr/local/bin/python3.11
PKG=/tmp/custom34-deployment-v4.1
curl -fL 'https://github.com/youku-vip/agentic-wam/raw/refs/heads/codex/custom-data-review-files/deliveries/custom34-deployment-v4.1.zip' -o /tmp/custom34-deployment-v4.1.zip
"$PY" -m zipfile -e /tmp/custom34-deployment-v4.1.zip "$PKG"
"$PY" "$PKG/deploy-custom/install_custom34_deployment.py" --client-auto
```

安装器只查找模块路径，不导入机器人模块。找不到时改用 `--client-root /包含environment目录的机器人项目`，或者 `--client-file /实际路径/environment/rtc_client_agenticwam.py`。客户端也先备份再替换。

只读检查 SDK 的旋转约定，不连接或控制机械臂：

```bash
"$PY" "$PKG/deploy-custom/check_client_rotation.py"
```

该检查比较多个非零姿态，报告角度单位、Euler 顺序、6D 行/列编码及 SDK 正反转换误差。项目的训练转换假设 6D 为旋转矩阵的前两列。客户端保留你原来的 SDK 正反转换；若检查未确认 first_two_columns_verified=true，请提供输出与 trans_utils 两个转换函数核对，不要直接转置已训练模型的数据。

## 4. 机器人客户端启动（同步，RTC 关闭）

保留原来的启动目录，将 SERVER_URL 改成 GPU 服务的实际可达地址。

```bash
PY=/usr/local/bin/python3.11
SERVER_URL='http://GPU服务器IP:8765'
export AGENTICWAM_AUTO_GRIPPER_RECOVERY=0
export AGENTICWAM_GRIPPER_LOCK_STEPS=0
export AGENTICWAM_SAVE_OBS=0
export AGENTICWAM_HTTP_TIMEOUT=120
"$PY" -m environment.rtc_client_agenticwam \
  --config_path=/data/code/common-tools/config_tools/config_dir/main_config.yaml \
  --robots.e16.urdf_path=/data/code/wheelrobot3_ros1/src/dual_arm_robot/wheelbot_description/urdf/wheelbot.urdf \
  --choice=zmq --which_robot=e16 \
  --env.post_url="$SERVER_URL" \
  --env.num_steps=20000 --rtc.s_min=20 \
  --env.im_w=424 --env.im_h=240 \
  --env.obs=robot --env.action_which=both --env.debug=False \
  --env.infermode=rot6d \
  --rtc.prompt="The right gripper picks up the pink cake and puts it into the air fryer." \
  --rtc.async_mode=false --rtc.use_rtc=false \
  --rtc.beta=10.0 --rtc.action_dim=20 --env.max_hz=10 --rtc.H=32
```

客户端 H 必须等于服务端 chunk_size；10 Hz 应与此数据的采样/执行约定一致。同步模式是采集新观测 → 推理等待 → 执行前 s_min 行 → 重新采集；推理等待期间不会额外发开爪指令。

如需要恢复旧行为，只在客户端显式设置：

```bash
export AGENTICWAM_AUTO_GRIPPER_RECOVERY=1
export AGENTICWAM_GRIPPER_LOCK_STEPS=15
```

两个开关独立：前者启用旧的闭合度低于 50 后开爪、返回本次任务初始姿态的恢复逻辑；后者启用夹爪切换后的锁定步数。默认都关闭。

## 5. 新任务、暂停与异常

客户端原有 8300 端口任务接口保留。每次 POST /prompt 都表示开始新任务，即使文本完全相同：停止旧 chunk、清空服务端 episode/code 状态、采集新观测重新规划。DELETE /prompt 暂停发动作。

```bash
curl -f -X POST http://127.0.0.1:8300/prompt \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The left gripper opens the air fryer."}'
curl -f -X DELETE http://127.0.0.1:8300/prompt
```

服务端每次 /act 都打印生成代码，代码只用于模型条件和审阅，客户端执行的是返回的数值 action20，不通过 exec 执行生成的 Python。网络失败、形状错误、无效 6D 或响应 request_id 不匹配时，该请求不返回可执行动作；同步循环等待后重新采集。服务端实例 ID 变化时丢弃动作，重新握手和 reset，不继续使用旧 episode。

## 6. 回退与验证范围

使用安装时打印的实际备份清单路径回退：

```bash
"$PY" "$PKG/deploy-custom/install_custom34_deployment.py" \
  --undo /实际备份路径/manifest.json
```

回退前会检查安装后文件是否又被编辑；有后续编辑时拒绝覆盖。测试结果和运行命令见压缩包内 deploy-custom/DEPLOYMENT_V4_1_TESTING.md。已测试 CPU 转换、模拟 HTTP 和同步控制循环、配置/提示词与图像处理一致性；未在此环境加载你的 checkpoint、运行 CUDA 或连接机器人，不能据此保证实机抓取成功率。
