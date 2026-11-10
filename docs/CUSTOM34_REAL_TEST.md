# 第 300 步 Custom34 权重的真机测试启动方式

本启动脚本适配本次 BF16 / complex-RoPE / 128-token 代码画布的修复权重，保持 256×448 letterbox 图片处理、32 步绝对 EEF20 输出及原训练统计值。默认同步使用，不开启 RTC 或 fast-joint。服务端提供推理接口，机器人动作由机器人端客户端执行。

当前离线样本已观察到闭爪指令提前约 0.2 秒，以及某条 32 步预测末帧出现额外张爪。本脚本不更改预测值。首轮采用客户端每次执行 8 步后重规划；较短执行段能缩短开环范围，不能保证消除这些错误。

## 文件与安装

若服务器已有此前部署版，只需新增 deploy-custom/start_custom34_real_test.sh。包内也附当前 custom34_server.py、RobotWin policy 包装器、文本缓存、图片审阅及数据转换依赖，缺失或旧版时按同路径备份替换。

保留训练修复的 src/agenticwam/models/experts/code_expert.py 和 src/agenticwam/models/cosmos_mot/build.py；本包不包含或覆盖它们。旧版部署程序不能代替这两处训练端修复。

## 服务端启动

~~~bash
cd /data/algorithm/agentic-wam

PY=/usr/local/bin/python3.11 GPU_ID=1 \
  bash deploy-custom/start_custom34_real_test.sh check

PY=/usr/local/bin/python3.11 GPU_ID=1 \
  bash deploy-custom/start_custom34_real_test.sh start
~~~

check 在 CPU 核对路径、配套统计快照、画布、图片配置及复数缓存的 BF16 类型转换，不读取大 checkpoint 张量，也不连接机器人。start 通过检查后加载模型并启动服务，终端出现 READY/serving 后可连接客户端。check 不验证大模型权重内容或 GPU 峰值显存。

默认路径：

~~~text
RUN_DIR=/data/algorithm/agentic-wam/runs/custom_code_repair_20261110_024220
TRAIN_CONFIG=$RUN_DIR/recipe/config.yaml
CKPT=$RUN_DIR/checkpoints/weights/step_000300.pt
STATS=$RUN_DIR/normalization_reference.json
GPU_ID=1
SERVER_HOST=0.0.0.0
SERVER_PORT=8765
~~~

可通过同名环境变量选择其他已保存快照或迁移后的目录，例如 CKPT=/完整路径/step_000200.pt。脚本保留 bf16 模型计算，VAE 图片导出单独使用 fp32。提示词来自训练 skill_prompt_map 与训练文本嵌入缓存，启动时打印可用英文任务，不加入新 API 说明。训练画布、RoPE、图片设置保持配套配置。

## 查看服务并重置任务状态

另开终端：

~~~bash
curl -f http://127.0.0.1:8765/health

curl -f -X POST http://127.0.0.1:8765/reset \
  -H 'Content-Type: application/json' -d '{}'
~~~

预期 health 字段：state_dim=34、model_dim=64、action_dim=20、chunk_size=32、rtc_enabled=false、code_objective=joint、video_size=[256,448]、image_resize_mode=letterbox、review_enabled=true。stats_sha256 应与启动预检查打印的一致。本次统计快照的已知值为 430b13a4b062871da7dd89a48ec05f03dd1b7944e373706157b075c3b51f40ee。

若通过平台域名/代理访问，SERVER_PORT 使用该域名实际映射的容器端口；机器人端 post_url 指向其可达地址。

## 机器人端同步模式

沿用此前已安装的 HTTP 版 environment.rtc_client_agenticwam。把 SERVER_URL 改为实际可达地址。以下保留用户原来的机器人配置路径，调整 s_min=8：

~~~bash
PY=/usr/local/bin/python3.11
SERVER_URL='http://GPU服务器IP:8765'
export AGENTICWAM_AUTO_GRIPPER_RECOVERY=0
export AGENTICWAM_GRIPPER_LOCK_STEPS=0
export AGENTICWAM_HTTP_TIMEOUT=120

"$PY" -m environment.rtc_client_agenticwam \
  --config_path=/data/code/common-tools/config_tools/config_dir/main_config.yaml \
  --robots.e16.urdf_path=/data/code/wheelrobot3_ros1/src/dual_arm_robot/wheelbot_description/urdf/wheelbot.urdf \
  --choice=zmq --which_robot=e16 \
  --env.post_url="$SERVER_URL" \
  --env.num_steps=20000 --rtc.s_min=8 \
  --env.im_w=424 --env.im_h=240 \
  --env.obs=robot --env.action_which=both --env.debug=False \
  --env.infermode=rot6d \
  --rtc.prompt="The left gripper turns the temperature control knob." \
  --rtc.async_mode=false --rtc.use_rtc=false \
  --rtc.beta=10.0 --rtc.action_dim=20 --env.max_hz=10 --rtc.H=32
~~~

--rtc.H 保持 32，与模型输出和 stepwise 归一化对应；--rtc.s_min=8 控制同步客户端执行前 8 行后重新规划。服务端 --replan-steps 不决定 HTTP 客户端实际执行多少行。此客户端采集的 240×424 RGB 图片由服务端按训练配置处理成 256×448 letterbox。

数值 action20 是两臂绝对 XYZ/rotation6D 与绝对夹爪值；客户端按已有 rot6d 接口转换执行。生成的 Python 是模型条件与审阅内容，不通过 exec 驱动机器人。默认没有旧的夹爪 15 步锁定与自动开爪回初始姿态覆盖。

四条训练任务指令：

~~~text
The left gripper opens the air fryer.
The right gripper picks up the pink cake and puts it into the air fryer.
The left gripper turns the temperature control knob.
The left gripper closes the air fryer.
~~~

机器人端已有任务 API 可用于开始同一或新任务、暂停发动作：

~~~bash
curl -f -X POST http://127.0.0.1:8300/prompt \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The left gripper turns the temperature control knob."}'

curl -f -X DELETE http://127.0.0.1:8300/prompt
~~~

新任务会清空客户端队列并重置服务端 episode/code 状态。DELETE /prompt 暂停客户端发动作，不是机器人硬件急停接口。

## 留存结果

每次 /act 保存输入图像、生成图像、chunks.npz 和 metadata.json；有专家 reference 请求时还输出对比指标/图。真机普通请求没有专家未来动作，所以不会自动产生真机专家对比曲线。生成图片失败会记录诊断，已完成的动作推理与图像导出状态分别保留。

默认日志目录为 RUN_DIR/real_test_时间戳，reviews 保存按请求分目录的图片和数组，code 保存每次推理的完整程序。固定保存位置可设置 LOG_DIR=/完整路径。服务端终端也打印每次生成的代码。

## 本环境验证范围

检查 shell 语法、CPU-only preflight、Tiny CodeExpert 父模块 BF16 转换后复数缓存完整性、服务端 CLI 参数及现有 EEF20↔64 桥接验证。没有在本环境加载用户的大权重、连接机器人或发送硬件动作；真实 GPU 服务与机器人执行需要用户服务器上的本次测试。
