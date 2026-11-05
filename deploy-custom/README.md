# RealMan RTC Client ↔ Agentic-WAM Custom34 无认证 HTTP 服务端

## 兼容范围

- 基于上传的原始 `rtc_client_agenticwam.py` 修改，保留 RealManGym、RealMan 机械臂执行、IKSolver、FastAPI `/prompt`/`/replay`、夹爪稳定器、异常重推理等控制逻辑。
- 兼容本包 `custom34_server.py`（即此前的无 Token 自定义双臂服务端），**不是**原 OpenWAM WebSocket 推理服务，也不是官方 YAM 14D 关节服务。
- 输入：`instruction` 字符串、**完整 34D** 原始状态、`head/left/right` RGB uint8，经 PNG/base64 封装后 HTTP `POST /act`。
- 输出：服务器 `status/action/latency/message`，客户端解析为 `np.ndarray [H,20] float32` 绝对 EEF 目标。服务端内部执行 34→64 状态、Flow Matching 推理、64D delta→20D absolute 变换；客户端只做实际机器人观测/指令映射。
- RTC：服务端推理期 rectified-flow inpainting，客户端按**观测时刻**截取旧动作，在 `/act` 发送 `committed=[K,20]` 和 `freeze=L`；带 K−L 软引导尾段。推理返回后跳过已经过时的动作索引；过期/冻结不足时丢弃新区块，不强制切换。无需更改训练。

## 安装（两台计算机）

GPU 服务器：

```bash
cp custom34_server.py /data/algorithm/agentic-wam/deploy/custom34_server.py
# 训练时的 CustomUnifiedEEFTransform 仍需位于
# src/agenticwam/datasets/lerobot/transforms/custom_unified_eef.py
```

机器人客户端：

```bash
cp /path/to/robot_project/rtc_client_agenticwam.py /path/to/robot_project/rtc_client_agenticwam.py.bak
cp rtc_client_agenticwam.py /path/to/robot_project/rtc_client_agenticwam.py
```

机器人端 Python 依赖保持旧项目的 `environment/`、`kinematics_tools/`、`trans_tools/`、`config_tools/`、FastAPI、Draccus 等不变。**推理通信不再依赖 `websockets`**，使用标准库 HTTP；PNG 编码仍依赖 Pillow。建议两端尽量保持环境和 checkpoint 一致。

## 服务端启动（真 RTC）

```bash
cd /data/algorithm/agentic-wam
export AGENTICWAM_DEPLOY_DTYPE=bf16
export AGENTICWAM_EVAL_PREPROC=parity
export AGENTICWAM_CODE_PIN=0
CUDA_VISIBLE_DEVICES=0 python deploy/custom34_server.py \
  --ckpt /真实路径/step_003500.pt \
  --task posttrain_custom \
  --dataset /data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30 \
  --num-inference-steps 10 \
  --fast-joint --rtc \
  --code-log-dir /tmp/agenticwam_custom_codes \
  --host 127.0.0.1 --port 8765
```

若 checkpoint 采用 `image_resize_mode: letterbox` 训练，改用 `export AGENTICWAM_EVAL_PREPROC=auto`，并保持训练/验证 processor 配置与对应训练一致。服务端会调用共享的等比例缩放补边变换。安装和回退见 [图像比例修复](../docs/CUSTOM_IMAGE_LETTERBOX.md)。

上例在 GPU 本机测试。若机器人在另外一台可信内网机器，服务器需通过受保护的端口/SSH 隧道可达；直接监听 `0.0.0.0` 时必须在防火墙限制客户端源地址，不能无认证暴露公网。**没有 Token 或 Authorization 要求**。

按你的实际训练任务名、checkpoint、stats 和 `model.infer_code_tokens` 设置启动参数。每次 `POST /act` 时服务端打印完整 POLICY CODE，并按 `--code-log-dir` 写文件。

## 客户端 Draccus 配置

在你现有配置中，至少修改：

```yaml
# cfg.env（原本 ws:// 的 WebSocket 地址改为 HTTP）
post_url: "http://GPU服务器IP:8765"
infermode: rot6d
# max_hz: 使用你实际控制/训练适配的频率（不要机械地改为30或10）

# cfg.rtc
H: 32                # 必须等于服务端 /health 的 chunk_size，通常训练 num_frames-1
s_min: 16            # 重新规划触发步；应按真实端到端延迟与控制频率选择
action_dim: 20
use_rtc: true
async_mode: true
```

启动时客户端会先 `GET /health` 校验 `state_dim=34, model_dim=64, action_dim=20, chunk_size=H`，同时确认 server `rtc_enabled=true`；再 `POST /reset`。若参数不一致会**报错退出**，而不是错误控制机器人。

推荐分阶段验证：

1. **通信/输出检查**：客户端 `use_rtc: false, async_mode: false`；服务端可暂不加 `--fast-joint --rtc`。先做非执行/dry-run，确认输入三相机、34D 状态和 20D 动作的所有坐标/夹爪约定。
2. **异步 RTC**：服务端 `--fast-joint --rtc`，客户端 `use_rtc: true, async_mode: true`。先不连接机械臂，用离线观测/模拟控制验证动作分块连续性。
3. **真机小范围验证**：控制器自己负责坐标系、限位、速度/加速度/碰撞约束和故障保持。仅在验证后开启实际运动。

## RTC 时序及预算

- 新 Chunk 的第 0 行对应**采集新观测时刻**。客户端记录该时刻旧 Chunk 的游标 `s`，发送旧动作 `A_cur[s:s+K]`，其中前 `L` 帧硬冻结、后面的 `K-L` 帧权重衰减。此时 `K<=24`，`L<=K`。
- `L` 来自最近测得的端到端推理延迟（单位动作步）及可配置余量；`K` 额外保留软引导过渡段。参数可用 `AGENTICWAM_RTC_FREEZE_MARGIN=2`、`AGENTICWAM_RTC_SOFT_ROWS=6` 修改。
- 返回时若控制执行已前进 `d` 帧，从新 Chunk 的 **第 d 行**继续执行，不重复播放新 Chunk 的前 d 行。如果 `d` 超出预测冻结区或已经超过整块长度，会拒绝切换并保持旧块末端目标，等待更可靠的新观测。
- **硬件限制**：当前按一次 `env.step` 对应一个 Action 行、以整数游标近似对齐。没有真实机器人动作执行时间戳/相机采集时间戳，无法保证亚帧对齐。上机前必须测量执行频率、采集延迟、推理 RTT 和控制器真实跟踪误差。
- 若 H=32、s_min=16、30Hz，则从重规划触发到旧动作块耗尽仅剩约 `16/30=0.53s`，推理 RTT 超过它就不能持续不间断执行。降低推理延迟、提早触发重规划或经验证后调整控制频率；**不能靠 RTC 开关解决算力延迟**。
- 丢弃的过期 Chunk 不会自动提高成功率；在网络/模型持续慢于控制需求时，机器人会保持最后一个目标点。请确认你的底层执行器具备真正的超时保持及安全停止能力。

## 本次同步修改

- 去掉 WebSocket `ping/reset/obs` 协议；HTTP `GET /health`、`POST /reset`、`POST /act`。
- 发送**完整** raw 34D（原代码只发 `state[14:34]` 20D）。
- 三路图像键改为 `head/left/right`；PNG/base64 与原来一致。
- 响应读取二维 `action`，**不**额外 `unsqueeze`。
- `H` 默认 32、`action_dim` 默认 20，服务端校验不匹配就停止。
- `asyncio.to_thread` 承担 PNG 压缩和 HTTP I/O，避免阻塞执行协程。超时不盲目重发 `/act`，避免模型 episode/code state 被重复推进。
- 客户端任务切换时停止旧动作，并 `POST /reset` 清空服务端每 Episode 的代码上下文。
- 重新采样以最新观测为基准，避免重复使用旧图像；异常重新推理依然保持并阻塞等待（加超时）。
- 原客户端每次请求都保存三张 JPG 的行为改为默认关闭，必要时设置 `AGENTICWAM_SAVE_OBS=1`。
- 客户端原有无鉴权 FastAPI 控制端默认绑定 `127.0.0.1:8300`（如确需外部调用，用 `PROMPT_SERVER_HOST` 显式配置并隔离网络）。

## 无设备测试

需要 `numpy` 和 `Pillow`：

```bash
python test_custom34_rtc_adapter.py
python -m py_compile rtc_client_agenticwam.py
```

测试采用模拟 HTTP 服务端，验证 `GET /health`、`POST /reset`、`POST /act`、状态34、RGB PNG、动作 `[32,20]`、RTC `committed/freeze` 和观测时刻对齐以及 horizon 不一致时拒绝连接。另可将 `validate_custom34_bridge.py` 放到 Agentic-WAM 的 `deploy/` 下测试 SE(3) 变换。

**目前仅完成语法和无设备 HTTP 模拟测试；尚未使用你的 GPU Checkpoint、真实相机、`trans_utils` Rotation6D 实现或 RealMan 硬件做集成测试。必须检查 Rotation6D 是旋转矩阵前两列的格式，以及绝对 EEF 单位、参考坐标系、夹爪语义与训练一致。**
