# 机器人 WebSocket 接口说明

本文档描述机器人与服务端之间的完整 WebSocket 协议，包括连接认证、连接管理、消息格式、实时语音会话、文本转语音、动作下发与错误处理。

- 服务实现：`util/audio_play.py`（`AudioPlayServer`）
- 实时语音实现：`core/ws_realtime_agent.py`（`RealtimeAudioHandler`）
- 服务地址：`ws://<server_host>:443/`（在 `run.py` 中配置 `AudioPlayServer(host="0.0.0.0", port=443)`）

## 1. 连接与认证

连接信息全部通过握手 Header 传递，不需要先发送注册事件。

```python
import websockets

headers = {
    "Authorization": "Bearer <robot_api_key>",
    "robot_id": "robot_001",
}

async with websockets.connect("ws://127.0.0.1:443/", additional_headers=headers) as ws:
    print(await ws.recv())
```

| Header          | 必填 | 说明                                                        |
| --------------- | -- | --------------------------------------------------------- |
| `Authorization` | 是  | 固定格式 `Bearer <robot_api_key>`，与 `config.robot_api_key` 一致 |
| `robot_id`      | 是  | 机器人唯一标识，服务端据此建立连接映射                                       |

握手成功后会收到一条合并消息：

```json
{"type": "connect", "state": 1, "messages": "success"}
```

握手失败（`Authorization` 缺失/不匹配，或缺少 `robot_id`）时服务端返回同一条错误并主动关闭连接：

```json
{"type": "error", "message": "Authorization error"}
```

## 2. 连接管理规则

1. **一个 robot\_id 只保留最后一次连接**：新连接注册时，服务端先挂载新连接，再摘除并关闭旧连接，旧连接的未就绪音频缓冲会被清空。
2. **断连关闭会话**：连接断开时立即关闭该 robot 的实时语音会话。
3. **断连清空缓冲**：连接断开时，该连接未就绪的音频缓冲会被清空；新连接始终以空缓冲开始。
4. **新连接必须重新 audio\_start**：每次建立连接后 `robot_session_ready` 都重置为未就绪。
5. **音色锁定**：同一会话复用期间固定使用首次 `audio_start` 传入的音色，后续传入的 `voice_id` 不会覆盖。

### 2.1 唤醒流程

客户端在唤醒词触发时按以下顺序处理：

1. 判断当前是否已有可用的 WebSocket 连接（已完成握手且未断开）。
2. **连接存在**：直接复用该连接，发送 `audio_start` 开始本轮对话。
3. **连接不存在或已断开**：先创建连接（Header 携带 `Authorization`、`robot_id`），收到 `connect` 后发送 `audio_start`。
4. 发送 `audio_start` 后无需等待 `session_started`，立即开始发送 `audio_data`；服务端会缓冲并在会话创建成功后按序补发。
5. 连接断开或 `audio_start` 失败时，服务端关闭语音会话；下次唤醒重新执行上述判断。

```
唤醒
  │
  ├─ 连接存在？──是──▶ 发送 audio_start ──▶ 发送 audio_data
  │
  └──否──▶ 建立连接（Authorization/robot_id）──▶ 收到 connect ──▶ 发送 audio_start ──▶ 发送 audio_data
```

> 同一 `robot_id` 只保留一个连接。连接断开后语音会话已关闭，下一次唤醒无论连接是否存在，都必须重新发送 `audio_start`。

## 3. 保活

保活分两层，互相独立：

1. **协议层 Ping/Pong**：服务端每 20 秒发送一次 Ping，20 秒内未收到 Pong 即判定连接断开。标准 WebSocket 客户端库会自动回复 Pong，业务代码无需处理。
2. **业务空闲断开**：连接超过 10 分钟没有收到任何客户端事件（`audio_start`、`audio_data`、`text` 等）时，服务端主动断开连接并关闭语音会话。协议层 Ping/Pong 不计入业务事件。

客户端不应再发送 `{"type":"ping"}`，服务端会将其视为未知消息忽略。

## 4. 消息格式

- 所有消息均为 JSON 文本帧。
- 音频统一使用 base64 编码。
- `type` 字段标识事件类型。

## 5. 客户端 → 服务端事件

| type          | 说明             | 是否必须      |
| ------------- | -------------- | --------- |
| `audio_start` | 启动一轮实时语音会话     | 实时语音必填    |
| `audio_data`  | 发送实时音频数据       | 实时语音必填    |
| `interrupt`   | 机器人回复过程中被打断时发送 | 必填        |
| `text`        | 文本转语音播放        | 必填        |
| `audio`       | 音频总结           | 必填        |
| `robot`       | 旧版注册事件         | 已废弃, 兼容旧版 |

### 5.1 audio\_start

启动一轮实时语音会话，参数放在 `message` 中。

```json
{
  "type": "audio_start",
  "message": {
    "app_id": "app_xxx",
    "request_id": "req_xxx",
    "voice_id": "voice_xxx"
  }
}
```

| 字段           | 必填 | 说明                              |
| ------------ | -- | ------------------------------- |
| `app_id`     | 否  | 应用标识，原样传给实时语音处理器                |
| `request_id` | 否  | 本轮请求标识，便于日志追踪                   |
| `voice_id`   | 否  | 音色标识；会话复用期间以首次值为准               |
| `robot_id`   | 否  | 省略时使用握手头 `robot_id`；若填写必须与握手头一致 |

行为说明：

- 每轮唤醒发送一次，同一连接可连续多轮调用。
- 会话复用期间 `voice_id` 锁定首次值，后续传入不会覆盖。
- 发起请求的连接不是该 `robot_id` 的当前活跃连接，或会话创建失败时，返回统一错误并断开连接。

成功响应：

```json
{"type": "session_started", "message": "ready"}
```

失败响应（所有 `audio_start` 失败统一返回该消息，随后服务端主动断开连接并关闭会话）：

```json
{"type": "error", "message": "会话创建失败，请重试"}
```

> 发送 `audio_start` 后即可继续发送 `audio_data`，无需等待 `session_started`：服务端会先缓冲，启动成功后按原顺序补发。每轮 `audio_start` 都会先清空上一轮残留的音频缓冲；启动失败时服务端返回错误并断开连接。

### 5.2 audio\_data

发送实时音频数据。

```json
{
  "type": "audio_data",
  "message": "<base64 编码的 PCM 数据>",
  "context": ""
}
```

| 字段        | 必填 | 说明                  |
| --------- | -- | ------------------- |
| `message` | 是  | base64 编码的 PCM 音频   |
| `context` | 否  | 业务透传上下文，原样交给实时语音处理器 |

音频格式要求：

| 项目   | 要求                   |
| ---- | -------------------- |
| 采样率  | 16000 Hz             |
| 位深   | 16 bit               |
| 声道   | 单声道                  |
| 建议分片 | 每帧 3200 字节（约 100 ms） |
| 编码   | base64               |

**未就绪时的缓冲行为**：

- 在 `session_started` 之前到达的 `audio_data` 会先进入该连接的有界缓冲，最多 50 帧（约 5 秒）。
- `audio_start` 成功后，缓冲音频会按原顺序补发，补发完成后才置为就绪。
- 缓冲溢出时丢弃最早的音频，并返回一次：

```json
{"type": "error", "message": "会话未就绪，音频缓冲已满，已丢弃最早的音频"}
```

- `audio_start` 失败时缓冲会被清空，需重新发起 `audio_start`。

### 5.3 text

文本转语音播放。服务端把 `message.text` 合成为音频，再通过 `audio` / `audio_end` 事件下发。

```json
{
  "type": "text",
  "message": {
    "robot_id": "robot_001",
    "text": "你好，我是机器人",
    "voice_type": "female",
    "speed": 0,
    "volume": 0,
    "request_id": "req_xxx"
  }
}
```

| 字段           | 类型     | 必填 | 说明                                  |
| ------------ | ------ | -- | ----------------------------------- |
| `robot_id`   | string | 是  | 播报目标机器人，必须是已连接在线的 robot             |
| `text`       | string | 是  | 需要播报的文本                             |
| `voice_type` | string | 是  | 音色类型，取值由 TTS 服务定义                   |
| `speed`      | number | 否  | 语速，默认 `0`                           |
| `volume`     | number | 否  | 音量，默认 `0`                           |
| `request_id` | string | 否  | 请求标识，缺省自动生成 UUID，同时作为下发音频的 `msg_id` |

响应：合成的音频按分片下发，最后以 `audio_end` 结束。

```json
{"type": "audio", "message": "<base64 音频>", "msg_id": "req_xxx"}
{"type": "audio_end", "message": "", "msg_id": "req_xxx"}
```

失败时：

```json
{"type": "error", "message": "文本转语音错误: ..."}
```

### 5.4 audio（音频总结）

音频总结后播报：服务端先用 `system` 提示词对 `audio_base64` 做总结，再把总结文本合成为音频播报。

```json
{
  "type": "audio",
  "message": {
    "robot_id": "robot_001",
    "audio_base64": "<base64 音频>",
    "system": "总结提示词",
    "voice_type": "female",
    "request_id": "req_xxx",
    "speed": 0,
    "volume": 0
  }
}
```

| 字段             | 类型     | 必填 | 说明                        |
| -------------- | ------ | -- | ------------------------- |
| `robot_id`     | string | 是  | 播报目标机器人                   |
| `audio_base64` | string | 是  | 需要总结的音频，base64 编码         |
| `system`       | string | 是  | 总结使用的系统提示词                |
| `voice_type`   | string | 是  | 播报音色                      |
| `request_id`   | string | 否  | 请求标识，缺省自动生成，同时作为 `msg_id` |
| `speed`        | number | 否  | 语速，默认 `0`                 |
| `volume`       | number | 否  | 音量，默认 `0`                 |

响应与失败格式同 `text`，失败信息为 `音频总结错误: ...`。

### 5.5 interrupt

机器人在回复过程中被打断时发送，例如用户再次唤醒、开口说话或按下打断键。

```json
{"type": "interrupt", "message": "robot_001"}
```

| 字段        | 类型     | 必填 | 说明              |
| --------- | ------ | -- | --------------- |
| `message` | string | 是  | 被打断的 `robot_id` |

行为说明：

- 服务端删除调度器中任务 ID 以该 `robot_id` 开头的所有待执行动作，已排期但未执行的动作不会再触发。
- 客户端应同时停止本地正在播放的音频与动作执行。
- 不会关闭 WebSocket 连接，也不影响实时语音会话；后续是否继续对话由新的 `audio_start` 或 VAD 事件决定。
- 成功与否没有响应事件，仅在服务端记录日志。

### 5.6 robot（已废弃）

旧版注册事件，`robot_id` 以握手头为准。新客户端无需发送。

```json
{"type": "robot", "message": "robot_001"}
```

| 字段        | 类型     | 必填 | 说明                        |
| --------- | ------ | -- | ------------------------- |
| `message` | string | 是  | 旧协议中的 `robot_id`，必须与握手头一致 |

服务端回复 `{"type": "ready", "messages": "success"}`；若 `message` 与握手头 `robot_id` 不一致，返回 `robot_id 与握手头 robot_id 不一致`，连接不关闭。

## 6. 服务端 → 客户端事件

| type               | 说明                  | message 内容             |
| ------------------ | ------------------- | ---------------------- |
| `connect`          | 握手成功                | 空                      |
| `ready`            | 旧版 `robot` 注册的兼容响应  | 空                      |
| `session_started`  | 实时语音会话已就绪           | `"ready"`              |
| `speech_started`   | VAD 检测到用户开始说话       | 空                      |
| `speech_end`       | VAD 检测到用户说话结束       | 空                      |
| `transcript_delta` | 语音识别结果（增量）          | 识别文本                   |
| `response`         | 模型回复文本              | 回复文本                   |
| `stop_audio`       | 要求机器人停止收音（如用户说“安静”） | 空                      |
| `tool`             | 模型调用工具              | `{"name": "工具名"}`      |
| `audio`            | 下发 TTS 音频分片         | base64 PCM，另含 `msg_id` |
| `audio_end`        | 一轮 TTS 音频结束         | 空，另含 `msg_id`          |
| `hermes`           | 下发动作执行指令            | 动作内容对象，另含 `msg_id`     |
| `error`            | 错误提示                | 错误描述                   |

示例：

```json
{"type": "session_started", "message": "ready"}
{"type": "speech_started", "message": ""}
{"type": "transcript_delta", "message": "今天天气"}
{"type": "response", "message": "今天晴，气温 25 度。"}
{"type": "audio", "message": "<base64>", "msg_id": "msg_001"}
{"type": "audio_end", "message": "", "msg_id": "msg_001"}
{"type": "hermes", "message": {"action": "wave"}, "msg_id": "msg_002"}
{"type": "error", "message": "会话创建失败，请重试"}
```

## 7. 典型时序

### 7.1 实时语音对话

```
客户端                                     服务端
  │──── 握手 Header: Authorization/robot_id ──▶│
  │◀─── {"type":"connect","state":1,...} ──────│
  │──── audio_start (app_id/request_id/voice) ▶│
  │──── audio_data ... ──────────────────────▶│  （无需等待，服务端缓冲）
  │                                            │ 创建/复用会话（锁定音色）
  │◀─── {"type":"session_started"} ────────────│
  │──── audio_data ... (继续发送) ─────────────▶│

  │                                            │ VAD
  │◀─── speech_started ────────────────────────│
  │◀─── transcript_delta ──────────────────────│
  │◀─── speech_end ────────────────────────────│

  │                                            │ 模型回复
  │◀─── response ──────────────────────────────│
  │◀─── audio / audio_end (TTS 播放) ──────────│
  │◀─── hermes (动作下发) ─────────────────────│
```

### 7.2 文本转语音

```
客户端 ──── {"type":"text","message":{...}} ───▶ 服务端
客户端 ◀─── {"type":"audio","msg_id":...} ────── 服务端（多次）
客户端 ◀─── {"type":"audio_end","msg_id":...} ── 服务端
```

### 7.3 重连

```
旧连接                       服务端                         新连接
  │                            │◀── 握手（同 robot_id）──────│
  │                            │ 新连接挂载，准备清空旧缓冲   │
  │◀──────── close ────────────│                            │
  │                            │──── connect ──────────────▶│
  │                            │       （旧连接缓冲丢弃）     │
  │                            │◀── audio_start ────────────│
  │                            │──── session_started ──────▶│
```

## 8. 错误信息

| 错误信息                         | 触发场景                                             | 连接是否关闭 |
| ---------------------------- | ------------------------------------------------ | ------ |
| `Authorization error`        | `Authorization` 缺失/不匹配，或握手头没有 `robot_id`         | 是      |
| `robot_id 与握手头 robot_id 不一致` | 旧版 `robot` 事件的 robot\_id 与握手头不同                  | 否      |
| `实时音频流处理器未配置`                | 服务端未调用 `set_realtime_handler`                    | 否      |
| `会话创建失败，请重试`                 | `audio_start` 缺少 robot\_id、连接归属校验失败，或实时会话未创建/未激活 | 是      |
| `未找到机器人会话`                   | 未注册的连接发送 `audio_data`                            | 否      |
| `会话未就绪，音频缓冲已满，已丢弃最早的音频`      | 未就绪缓冲超过 50 帧                                     | 否      |
| `文本转语音错误: ...`               | TTS 合成失败                                         | 否      |
| `音频总结错误: ...`                | 音频总结失败                                           | 否      |

## 9. 客户端接入注意事项

1. 每个 `robot_id` 同一时刻只使用一个连接；重连后旧连接会被服务端关闭。
2. 重连后必须重新发送 `audio_start`，不能沿用旧连接的就绪状态。
3. 发送 `audio_start` 后即可连续发送 `audio_data`，无需等待 `session_started`；提前发送的音频会被缓冲，最多保留约 5 秒。
4. `audio_start` 失败后服务端会断开连接并关闭会话，需重新连接后再发起 `audio_start`。
5. 音频参数必须为 16 kHz / 16 bit / 单声道，每帧建议 3200 字节。
6. 经过反向代理时，需确保 `Authorization` 和 `robot_id` 请求头被转发，且代理允许 WebSocket Upgrade。
7. 客户端无需实现业务心跳，保持对协议层 Ping 的自动 Pong 即可。
8. 语气/音色以每轮首次成功 `audio_start` 传入的 `voice_id` 为准。

