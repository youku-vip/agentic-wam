import asyncio
import base64
import io
import json
from collections import deque
from urllib.parse import urlparse
import glob
import time
import numpy as np
from loguru import logger
from config_tools.draccus_config import BaseConfig
from kinematics_tools.ik_solver import IKSolver
from environment.realman_env import RealManGym
from trans_tools import trans_utils
import draccus
import os
import matplotlib
import sys
import cv2
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import threading
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

matplotlib.use('Agg')  # 非交互式后端

from typing import Any, Dict
from collections import defaultdict

logger.remove()

# 添加一个sink，只输出WARNING及以上级别的日志
logger.add(sys.stderr, level="WARNING")


def normalize_http_base_url(uri: str) -> str:
    """Use the existing cfg.env.post_url, allowing old ws:// address prefixes."""
    raw = uri.strip()
    if not raw:
        raise ValueError("cfg.env.post_url is empty")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
#    if scheme not in ("http", "https") or not parsed.hostname or not parsed.port:
#        raise ValueError(f"expected http://host:port, got {uri!r}")
    if parsed.path not in ("", "/", "/act"):
        logger.warning("[HTTP] ignoring old WebSocket path %s; HTTP server uses /act", parsed.path)
    base = f"{scheme}://{parsed.netloc}"
    if scheme != parsed.scheme:
        logger.warning("[HTTP] converted %s:// URL into %s:// for custom34 server", parsed.scheme, scheme)
    return base


def encode_png_b64(img: np.ndarray) -> str:
    """HxW RGB uint8 -> PNG/base64 for the custom34 HTTP server."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_three_images_sync(head_img, left_img, right_img, base_dir="concat_img"):
    """同步保存三张图像到根目录下的 concat_img 文件夹"""
    # 根目录下的 concat_img 文件夹
    save_path = Path.cwd() / base_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 使用毫秒时间戳避免文件名冲突
    timestamp = int(time.time() * 1000)
    
    # 保存三张图像
    cv2.imwrite(
        str(save_path / f"head_{timestamp}.jpg"),
        cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        str(save_path / f"left_{timestamp}.jpg"),
        cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        str(save_path / f"right_{timestamp}.jpg"),
        cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)
    )
    
    logger.warning(f"图像已保存到 {save_path}/")

# 客户端发送完整 raw state[34]；由服务端的训练变换转换为 unified EEF64。

def convert_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    转换来自环境的观测数据格式，以匹配模型要求
    """
    dst_obs = defaultdict(dict)
    state_np = np.zeros((34,), dtype=np.float32)
    state_np[:14] = obs["joint_positions"][:14]
    state_np[14:17] = obs["ee_pos_quat"][:3]
    state_np[17:23] = trans_utils.euler_angle_to_rot6d(obs["ee_pos_quat"][3:6])
    state_np[23:26] = obs["ee_pos_quat"][6:9]
    state_np[26:32] = trans_utils.euler_angle_to_rot6d(obs["ee_pos_quat"][9:12])
    state_np[32] = 1 if 'open' == obs["gripper_position"][0] else 0
    state_np[33] = 1 if 'open' == obs["gripper_position"][1] else 0
    dst_obs["state"] = state_np
    logger.warning(f'obs keys: {obs.keys()}')
    for name in obs.keys():
        if name.endswith('head_rgb'):
            dst_obs['images']['cam_high'] = obs[name]
        elif name.endswith('left_rgb'):
            dst_obs['images']['cam_left_wrist'] = obs[name]
        elif name.endswith('right_rgb'):
            dst_obs['images']['cam_right_wrist'] = obs[name]
        elif name.endswith('_rgb'):
            dst_obs['images'][name[:-4]] = obs[name]
        else:
            dst_obs[name] = obs[name]

    return dst_obs

def make_zero_observation(dst_obs, prompt: str = "pick up the object") -> dict:
    """Build the custom34 HTTP JSON request, not OpenWAM WebSocket messages.

    State is ALL 34 numbers. Images use head/left/right PNG/base64; their RGB
    order is unchanged. The HTTP server performs training-identical preprocessing.
    """
    images = dst_obs["images"]
    state = np.asarray(dst_obs["state"], dtype=np.float32)
    if state.shape != (34,) or not np.isfinite(state).all():
        raise ValueError(f"raw state must be finite (34,), got {state.shape}")
    head_img = np.asarray(images["cam_high"])
    left_img = np.asarray(images["cam_left_wrist"])
    right_img = np.asarray(images["cam_right_wrist"])
    if os.environ.get("AGENTICWAM_SAVE_OBS", "0") == "1":
        save_three_images_sync(head_img, left_img, right_img)
    return {
        "instruction": prompt,
        "state": state.tolist(),
        "images": {
            "head": encode_png_b64(head_img),
            "left": encode_png_b64(left_img),
            "right": encode_png_b64(right_img),
        },
    }


class SharedPrompt:
    """跨线程共享的任务指令变量（FastAPI 服务线程写入，客户端事件循环线程读取）。"""

    def __init__(self, initial_prompt: str = ""):
        self._prompt = (initial_prompt or "").strip()
        self._lock = threading.Lock()

    def set(self, value: str) -> str:
        with self._lock:
            self._prompt = (value or "").strip()
            return self._prompt

    def get(self) -> str:
        with self._lock:
            return self._prompt

    def clear(self) -> None:
        self.set("")


class PromptRequest(BaseModel):
    prompt: str


# 子任务轨迹回放默认配置（来自 test-checkjoint-folding.py）
REPLAY_DATA_PATH = "/data/code/e2e-manipulation/environment/environment/traj/"
REPLAY_SKIP_RATE = 3
REPLAY_WAIT_PER_STEP = 0.0


def replay_joint_trajectory(
    env,
    ep_path: str = REPLAY_DATA_PATH,
    skip_rate: int = REPLAY_SKIP_RATE,
    wait_per_step: float = REPLAY_WAIT_PER_STEP,
):
    """子任务轨迹回放：复现录制的双臂关节轨迹。

    逻辑来自 test-checkjoint-folding.py，不新建 RoboticArm 连接，
    复用 env.rm_client 已连接的左右臂对象直接执行 rm_movej。
    注意：不能走 env.step()，其底层 exec_rm_movej 会把目标关节裁剪到
    当前值 ±5° 以内（模型输出安全限制），跳帧路标间距超过 5° 时回放轨迹会走样。

    轨迹数据格式（parquet, observation.state 列）:
    - state[:7]   左臂 7 关节
    - state[7:14] 右臂 7 关节
    - state[-2]   左夹爪 (1=open, 0=close)
    - state[-1]   右夹爪 (1=open, 0=close)
    """
    rm_client = env.rm_client
    left_arm = rm_client.left_arm
    right_arm = rm_client.right_arm

    episode_files = sorted(glob.glob(os.path.join(ep_path, "*.parquet")))
    all_episodes = []

    for ep_file in episode_files:
        df = pd.read_parquet(ep_file)
        if 'observation.state' in df.columns:
            state_traj = np.stack(df['observation.state'].values)
            all_episodes.append(state_traj)

    if len(all_episodes) == 0:
        logger.warning(f"[REPLAY] {ep_path} 下没有可用轨迹数据")
        return

    traj = all_episodes[0]
    logger.warning(f"[REPLAY] 轨迹长度: {len(traj)}, 跳帧率: {skip_rate}, 开始回放...")

    prev_left_grip = None
    prev_right_grip = None

    for step in range(0, len(traj), skip_rate):
        state = traj[step]
        left_joints = state[:7].tolist()
        right_joints = state[7:14].tolist()
        left_grip = state[-2]
        right_grip = state[-1]

        # 夹爪只在状态变化时下发（阻塞等待完成）
        if prev_left_grip is None or left_grip != prev_left_grip:
            rm_client.set_gripper_status('left', 'open' if left_grip == 1 else 'close')
            prev_left_grip = left_grip
        if prev_right_grip is None or right_grip != prev_right_grip:
            rm_client.set_gripper_status('right', 'open' if right_grip == 1 else 'close')
            prev_right_grip = right_grip

        # 关节运动：与 test-checkjoint-folding.py 完全一致（v=7, block=1 阻塞到位）
        left_arm.rm_movej(joint=left_joints, v=7, r=0, connect=0, block=1)
        right_arm.rm_movej(joint=right_joints, v=7, r=0, connect=0, block=1)

        if wait_per_step > 0:
            time.sleep(wait_per_step)

    logger.warning("[REPLAY] 轨迹回放完成")


class ReplayController:
    """子任务轨迹回放控制器：供 FastAPI 线程触发，后台线程执行，主控制循环查询状态后暂停。"""

    def __init__(self, env):
        self._env = env
        self._lock = threading.Lock()
        self._replaying = False
        self._thread = None

    def is_replaying(self) -> bool:
        with self._lock:
            return self._replaying

    def trigger(self) -> dict:
        with self._lock:
            if self._replaying:
                return {"status": "busy", "replaying": True}
            self._replaying = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="subtask-replay")
        self._thread.start()
        return {"status": "started", "replaying": True}

    def _run(self):
        try:
            replay_joint_trajectory(self._env)
        except Exception as e:
            logger.error(f"[REPLAY] 回放异常: {e}")
        finally:
            with self._lock:
                self._replaying = False
            logger.warning("[REPLAY] 子任务轨迹回放结束")


def build_prompt_app(shared_prompt: SharedPrompt, replay_controller: ReplayController = None) -> FastAPI:
    app = FastAPI(title="Task Prompt Server")

    @app.post("/prompt")
    async def set_prompt(req: PromptRequest):
        """规划模型调用：设置/更新任务指令"""
        shared_prompt.set(req.prompt)
        return {"status": "ok", "prompt": shared_prompt.get()}

    @app.get("/prompt")
    async def get_prompt():
        """读取当前任务指令"""
        return {"prompt": shared_prompt.get()}

    @app.delete("/prompt")
    async def clear_prompt():
        """清空任务指令（客户端将停止执行任何动作）"""
        shared_prompt.clear()
        return {"status": "cleared", "prompt": ""}

    @app.get("/health")
    async def health():
        return {"status": "ok", "prompt": shared_prompt.get()}

    @app.post("/replay")
    async def start_replay():
        """规划模型调用：判定下一步为需回放的子任务时触发轨迹回放（后台线程执行，主循环暂停）"""
        if replay_controller is None:
            return {"status": "error", "detail": "replay_controller 未配置"}
        result = replay_controller.trigger()
        logger.warning(f"[REPLAY] /replay 被调用: {result}")
        return result

    @app.get("/replay")
    async def replay_status():
        """查询轨迹回放是否进行中"""
        return {"replaying": replay_controller.is_replaying() if replay_controller else False}

    return app


def start_prompt_server(
    shared_prompt: SharedPrompt,
    replay_controller: ReplayController = None,
    host: str = "0.0.0.0",
    port: int = 8300,
):
    """在后台线程中启动 FastAPI 服务，与客户端共享 SharedPrompt 变量"""
    app = build_prompt_app(shared_prompt, replay_controller)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="prompt-server")
    thread.start()
    return server, thread

class RTCClient:
    """RealMan control loop adapter for the no-auth custom34 HTTP inference server.

    The server speaks GET /health, POST /reset, POST /act, with state34 and
    absolute EEF20 action chunks. RTC, when enabled, uses server-side rectified-
    flow inpainting (committed [K,20] + freeze L) and client-side time alignment.
    """

    def __init__(
        self,
        server_uri: str,
        prompt: str,
        infer_type: bool,
        H: int = 32,
        s_min: int = 16,
        b: int = 10,
        beta: float = 5.0,
        action_dim: int = 20,
        use_rtc: bool = False,
        use_async: bool = False,
        control_hz: float = 10.0,
    ):
        self.server_uri = normalize_http_base_url(server_uri)
        self.prompt = prompt
        self._A_cur_prompt = prompt
        self.infer_type = infer_type
        self.H = int(H)
        self.s_min = int(s_min)
        self.beta = beta
        self.action_dim = int(action_dim)
        self.use_rtc = bool(use_rtc)
        self.use_async = bool(use_async)
        self.control_hz = float(control_hz)
        if self.H <= 1 or not (1 <= self.s_min < self.H):
            raise ValueError(f"invalid H={self.H} / s_min={self.s_min}; require 1 <= s_min < H")
        if self.action_dim != 20:
            raise ValueError(f"custom34 server requires action_dim=20; got {self.action_dim}")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be >0")
        self.http_timeout_s = float(os.environ.get("AGENTICWAM_HTTP_TIMEOUT", "60"))
        self.rtc_soft_rows = max(0, int(os.environ.get("AGENTICWAM_RTC_SOFT_ROWS", "6")))
        self.rtc_freeze_margin = max(0, int(os.environ.get("AGENTICWAM_RTC_FREEZE_MARGIN", "2")))
        self._server_rtc = False
        self._server_prompt: str | None = None
        self._request_lock = asyncio.Lock()  # one model, do not overlap /act and /reset

        self.save_viz = self.use_async
        self.viz_dir = "action_comparisons"
        self.inference_count = 0
        if self.save_viz:
            os.makedirs(self.viz_dir, exist_ok=True)

        self.t = 0  # number of actions consumed from current chunk
        self.A_cur = np.zeros((self.H, 20), dtype=np.float32)
        self.o_cur = None
        self._obs_step = 0   # row index at observation CAPTURE time, not request send time
        self._obs_generation = 0
        self._last_requested_obs_generation = 0
        self._control_epoch = 0  # invalidate in-flight inference when paused/restarted
        self._hold_action = None

        self.delay_buffer = deque([2] * max(1, int(b)), maxlen=max(1, int(b)))
        self.lock = asyncio.Lock()
        self.cond = asyncio.Condition(self.lock)
        self.new_actions_ready = asyncio.Event()
        self._need_reinference = False
        self._reinference_pending = False
        self._suppress_rtc_once = False
        self._last_sent_prefix_len = 0
        self._last_sent_committed_len = 0
        self._last_latency_ms = 0.0
        self.running = False
        self._inference_task = None
        self.rgb_history = None

    @staticmethod
    def _request_sync(method: str, url: str, body: dict | None, timeout: float) -> dict:
        """Blocking HTTP I/O; ONLY called via asyncio.to_thread()."""
        raw = json.dumps(body, allow_nan=False).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if raw is not None else {}
        req = Request(url, data=raw, method=method, headers=headers)
        try:
            with urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                detail = payload.get("message", str(payload))
            except (ValueError, UnicodeError, OSError):
                detail = str(exc)
            raise ConnectionError(f"HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            # Never silently retry timed-out /act: the server may have already
            # completed it, potentially advancing per-episode code/step state.
            raise ConnectionError(f"HTTP request failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ConnectionError("server returned non-object JSON")
        return parsed

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        return await asyncio.to_thread(
            self._request_sync, method, self.server_uri + path, body, self.http_timeout_s
        )

    async def _handshake(self) -> None:
        health = await self._request("GET", "/health")
        if health.get("status") != "success":
            raise ConnectionError(f"GET /health failed: {health}")
        for name, expected in (("state_dim", 34), ("action_dim", 20), ("model_dim", 64)):
            if int(health.get(name, -1)) != expected:
                raise ValueError(f"server {name}={health.get(name)!r}, expected {expected}")
        actual_h = int(health.get("chunk_size", -1))
        if actual_h != self.H:
            raise ValueError(
                f"RTC config H={self.H} but server chunk_size={actual_h}; "
                "set cfg.rtc.H to the training action horizon"
            )
        self._server_rtc = bool(health.get("rtc_enabled", False))
        if self.use_rtc and self.use_async and not self._server_rtc:
            raise ValueError(
                "cfg.rtc.use_rtc=True but server RTC is disabled; "
                "start custom34_server.py with --fast-joint --rtc, or disable RTC in client"
            )
        if self.use_rtc and not self.use_async:
            logger.warning("[RTC] 同步模式下无已执行前缀；请求会使用普通 Action Chunking")
        logger.warning(
            "[HTTP] server health: state34 -> model64 -> action20, horizon=%d, rtc=%s",
            actual_h, self._server_rtc,
        )
        reset = await self._request("POST", "/reset", {})
        if reset.get("status") != "success":
            raise ConnectionError(f"POST /reset failed: {reset}")
        self._server_prompt = None
        logger.warning("[HTTP] remote episode reset OK")

    async def reset_episode(self) -> None:
        """Reset remote episode while serialized with inference requests."""
        async with self._request_lock:
            result = await self._request("POST", "/reset", {})
            if result.get("status") != "success":
                raise ConnectionError(f"POST /reset failed: {result}")
            self._server_prompt = None
        logger.warning("[HTTP] new episode reset OK")

    async def connect(self) -> None:
        logger.warning("[HTTP] Connecting to %s", self.server_uri)
        async with self._request_lock:
            await self._handshake()
        self.running = True
        if self.use_async:
            self._inference_task = asyncio.create_task(self._inference_loop(), name="wam-replan-loop")

    async def guided_inference(
        self, o, prefix_action=None, prefix_len=0, *, prompt: str | None = None
    ) -> dict:
        """Return {'action': np.ndarray[H,20]} from POST /act; no robot I/O here.

        prefix_action is already aligned to the observation's capture time:
        row k corresponds to observation_time + k/control_hz.  prefix_len is the
        hard-frozen portion; the rest of prefix_action is the soft guided tail.
        """
        if o is None:
            raise ConnectionError("no observation for inference")
        instruction = self.prompt if prompt is None else prompt
        if not isinstance(instruction, str) or not instruction.strip():
            raise ConnectionError("task instruction is empty")
        # Encode PNG in a worker thread: do not block actuator event loop.
        proc_o = convert_obs(o)
        payload = await asyncio.to_thread(make_zero_observation, proc_o, instruction)
        self._last_sent_prefix_len = 0
        self._last_sent_committed_len = 0
        if self._suppress_rtc_once:
            logger.warning("[RTC] 旧动作已失效，本次不发送 RTC 前缀")
            self._suppress_rtc_once = False
        elif prefix_action is not None and prefix_len > 0 and self.use_rtc:
            committed = np.asarray(prefix_action, dtype=np.float32)
            K = committed.shape[0] if committed.ndim == 2 else 0
            freeze = int(prefix_len)
            if committed.ndim != 2 or committed.shape[1] != 20 or not np.isfinite(committed).all():
                raise ValueError(f"RTC committed must be finite [K,20], got {committed.shape}")
            if not (1 <= freeze <= K <= min(24, self.H - 1)):
                raise ValueError(f"RTC requires 1<=freeze<=K<=min(24,H-1), got L={freeze} K={K}")
            if not self._server_rtc:
                raise ConnectionError("server RTC disabled, refusing silently unguided motion")
            payload["committed"] = committed.tolist()
            payload["freeze"] = freeze
            self._last_sent_prefix_len = freeze
            self._last_sent_committed_len = K

        async with self._request_lock:
            # The policy retains code-cache/episode state; a new task must not
            # silently inherit the previous task's program or code history.
            if self._server_prompt is not None and self._server_prompt != instruction:
                reset = await self._request("POST", "/reset", {})
                if reset.get("status") != "success":
                    raise ConnectionError(f"task-switch remote reset failed: {reset}")
                logger.warning("[HTTP] task changed; remote model episode reset")
                self._server_prompt = None
            start = time.monotonic()
            result = await self._request("POST", "/act", payload)
            self._last_latency_ms = 1000 * (time.monotonic() - start)
            if result.get("status") == "success":
                self._server_prompt = instruction
        if result.get("status") != "success":
            raise ConnectionError(f"server /act error: {result.get('message', result)}")
        action = np.asarray(result.get("action"), dtype=np.float32)
        if action.shape != (self.H, 20) or not np.isfinite(action).all():
            raise ConnectionError(f"invalid action shape/values: {action.shape}, expected ({self.H},20)")
        self.inference_count += 1
        estimated_delay = max(1, int(np.ceil(self._last_latency_ms * self.control_hz / 1000.0)))
        self.delay_buffer.append(estimated_delay)
        logger.warning(
            "[HTTP] inference #%d: server %.0fms / RTT %.0fms, action=%s, rtc_freeze=%d K=%d",
            self.inference_count, float(result.get("latency", 0)), self._last_latency_ms,
            action.shape, self._last_sent_prefix_len, self._last_sent_committed_len,
        )
        return {"action": action, "latency": self._last_latency_ms}

    async def _inference_loop(self):
        while self.running:
            async with self.cond:
                try:
                    await asyncio.wait_for(
                        self.cond.wait_for(lambda:
                            self._need_reinference or
                            (self.t >= self.s_min and self.o_cur is not None
                             and self._obs_generation > self._last_requested_obs_generation)
                        ),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    continue
                was_reinference = self._need_reinference
                if was_reinference:
                    self._need_reinference = False
                if self.o_cur is None:
                    logger.warning("[RTC] no fresh observation; skip replan")
                    continue
                o = self.o_cur
                s = self._obs_step  # use obs CAPTURE index, not the current t
                epoch = self._control_epoch
                prompt = self.prompt
                requested_gen = self._obs_generation
                self._last_requested_obs_generation = requested_gen
                previous_actions = self.A_cur.copy()
                K = 0
                freeze = 0
                prefix_action = None
                if self.use_rtc and not was_reinference and prompt == self._A_cur_prompt:
                    available = max(0, self.H - s)
                    if available > 0:
                        pred_delay = max(self.delay_buffer) + self.rtc_freeze_margin
                        freeze = min(int(pred_delay), available, 24, self.H - 1)
                        # Give the new chunk a soft continuation region, NOT just hard prefix.
                        K = min(available, 24, self.H - 1, freeze + self.rtc_soft_rows)
                        if freeze > 0:
                            prefix_action = previous_actions[s:s + K].copy()
                if self.use_rtc and prompt != self._A_cur_prompt:
                    logger.warning("[RTC] task changed; discard previous task's committed actions")

            try:
                result = await self.guided_inference(
                    o, prefix_action=prefix_action, prefix_len=freeze, prompt=prompt
                )
            except (ConnectionError, OSError, ValueError) as exc:
                logger.error("[HTTP] background inference failed: %s; keep old action/hold", exc)
                if was_reinference:
                    async with self.cond:
                        self._need_reinference = True
                        self.cond.notify_all()
                await asyncio.sleep(1.0)
                continue

            new_actions = result["action"]
            async with self.cond:
                if epoch != self._control_epoch or prompt != self.prompt:
                    logger.warning("[RTC] stale response (pause/task-switch); discarded")
                    continue
                if was_reinference:
                    self.A_cur = new_actions
                    self._A_cur_prompt = prompt
                    self.t = 0
                    self._reinference_pending = False
                    self._hold_action = None
                    self.new_actions_ready.set()
                    logger.warning("[RTC] explicit reinference completed, starting fresh chunk")
                    continue
                elapsed = max(0, self.t - s)
                self.delay_buffer.append(elapsed)
                if elapsed >= self.H:
                    logger.error("[RTC] result stale: %d executed rows >= new H=%d; discarded", elapsed, self.H)
                    continue
                if prefix_action is not None and elapsed > freeze:
                    # The unseen execution delay exceeded the hard-frozen prefix.
                    # Using an unguided B[elapsed] could jump the robot.
                    logger.error(
                        "[RTC] actual delay=%d > frozen=%d (K=%d); reject chunk and hold old action. "
                        "Reduce inference latency/control rate or increase horizon.", elapsed, freeze, K
                    )
                    continue
                self.A_cur = new_actions
                self._A_cur_prompt = prompt
                # Skip all B rows whose timestamps have already passed while inferring.
                self.t = elapsed
                self.new_actions_ready.set()
                logger.warning(
                    "[RTC] switched chunk at obs row s=%d, real delay=%d, new cursor=%d, "
                    "frozen=%d soft_end=%d", s, elapsed, self.t, freeze, K,
                )

    async def pause(self) -> None:
        """Invalidate in-flight replan and prevent stale response activating on resume."""
        async with self.cond:
            self._control_epoch += 1
            self.t = 0
            self.o_cur = None
            self._last_requested_obs_generation = self._obs_generation
            self._need_reinference = False
            self._reinference_pending = False
            self.cond.notify_all()

    async def get_action(self, o_next: dict = None) -> np.ndarray:
        async with self.cond:
            if self._reinference_pending and self._hold_action is not None:
                if o_next is not None:
                    self.o_cur = o_next
                    self._obs_step = self.t
                    self._obs_generation += 1
                return self._hold_action.copy()
            # o_next is a snapshot collected AFTER t actions of the current chunk.
            if o_next is not None:
                self.o_cur = o_next
                self._obs_step = self.t
                self._obs_generation += 1
            idx = min(self.t, self.H - 1)
            self.t += 1
            self.cond.notify_all()
            action = self.A_cur[idx].copy()
            self._hold_action = action
            if self.t > self.H:
                logger.warning("[RTC] chunk exhausted; repeating last target (hold), not extrapolating")
            return action

    def suppress_rtc_next(self) -> None:
        self._suppress_rtc_once = True
        logger.warning("[RTC] next guidance suppressed")

    async def trigger_reinference(self, obs=None):
        self.new_actions_ready.clear()
        async with self.cond:
            self._need_reinference = True
            self._reinference_pending = True
            if obs is not None:
                self.o_cur = obs
                self._obs_step = self.t
                self._obs_generation += 1
            self.cond.notify_all()
        logger.warning("[RTC] emergency replan requested; hold current target until new chunk")

    async def request_reinference(self, obs=None):
        await self.trigger_reinference(obs=obs)
        timeout = float(os.environ.get("AGENTICWAM_REINFERENCE_TIMEOUT", "120"))
        end = time.monotonic() + timeout
        while self._reinference_pending:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise ConnectionError("emergency reinference timed out; do not resume robot motion")
            try:
                await asyncio.wait_for(self.new_actions_ready.wait(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise ConnectionError("emergency reinference timed out; hold robot") from exc
            self.new_actions_ready.clear()
        logger.warning("[RTC] emergency replan ready")

    async def stop(self):
        self.running = False
        async with self.cond:
            self.cond.notify_all()
        if self._inference_task is not None:
            self._inference_task.cancel()
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass
            self._inference_task = None


@draccus.wrap()
def main(cfg: BaseConfig):
    # 可选 RViz 可视化
    if cfg.env.rviz_visualization:
        import rospy
        rospy.init_node("rviz_visualization")
    if cfg.which_robot == 'c':
        from environment.simu_controller import RMClient
    else:
        # from robot_client.rm_client import RMClient
        from environment.rm_controller import RMClient
    rm_client = RMClient(cfg)
    ik_solver = IKSolver(cfg)
    env = RealManGym(
        rm_client, ik_solver,
        im_width=cfg.env.im_w, im_height=cfg.env.im_h,
        obs_which=cfg.env.obs_which, action_which=cfg.env.action_which,
        debug=cfg.env.debug, rviz_visualization=cfg.env.rviz_visualization,
    )
    rtc = RTCClient(
        server_uri=cfg.env.post_url,
        prompt=cfg.rtc.prompt,
        infer_type=cfg.rtc.infer_type,
        H=cfg.rtc.H,
        s_min=cfg.rtc.s_min,
        b=cfg.rtc.b,
        beta=cfg.rtc.beta,
        action_dim=cfg.rtc.action_dim,
        use_rtc=cfg.rtc.use_rtc,
        use_async=cfg.rtc.async_mode,
        control_hz=cfg.env.max_hz,
    )

    # 共享任务指令变量 + FastAPI 服务（规划模型通过 HTTP 实时修改任务指令）
    # 初始值仍取配置文件中的 cfg.rtc.prompt，可传空字符串让客户端启动后处于空闲等待
    shared_prompt = SharedPrompt(initial_prompt=cfg.rtc.prompt)

    async def get_obs_with_retry(max_retry_seconds: float = 600.0):
        """获取观测，失败时每 2 秒重试一次，持续 10 分钟仍失败则抛出异常。"""
        start = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                return await asyncio.to_thread(env.get_obs)
            except Exception as e:
                elapsed = time.monotonic() - start
                if elapsed >= max_retry_seconds:
                    logger.error(
                        f"[OBS] 获取观测持续失败 {elapsed:.0f}s (共尝试 {attempt} 次)，"
                        f"放弃重试并抛出异常"
                    )
                    raise
                logger.warning(
                    f"[OBS] 获取观测失败 (第 {attempt} 次): {type(e).__name__}: {e}，"
                    f"2s 后重试 (已耗时 {elapsed:.0f}s / 上限 {max_retry_seconds:.0f}s)"
                )
                await asyncio.sleep(2)

    # 子任务轨迹回放控制器（复用 env 已连接的机械臂）
    replay_ctrl = ReplayController(env)
    prompt_host = os.environ.get("PROMPT_SERVER_HOST", "127.0.0.1")
    prompt_port = int(os.environ.get("PROMPT_SERVER_PORT", "8300"))
    start_prompt_server(shared_prompt, replay_ctrl, host=prompt_host, port=prompt_port)
    logger.warning(
        f"[PROMPT] 指令服务已启动: http://{prompt_host}:{prompt_port} "
        f"(POST /prompt 设置, GET /prompt 读取, DELETE /prompt 清空, "
        f"POST /replay 触发子任务轨迹回放, GET /replay 查询回放状态), "
        f"当前指令: '{shared_prompt.get()}'"
    )

    # 定义夹爪状态稳定器
    class GripperStateStabilizer:
        def __init__(self, lock_duration=15):
            self.lock_duration = lock_duration  # 状态转换后锁定的步数
            self.last_state = None  # 上一次的夹爪状态，'open'或'close'
            self.lock_counter = 0  # 当前锁定计数
            self.locked_state = None  # 被锁定的状态

        def get_stable_state(self, action_value):
            """
            获取稳定的夹爪状态
            action_value: 夹爪动作值，>0.5表示open，<=0.5表示close
            返回: 稳定的夹爪状态，'open'或'close'
            """
            current_state = 'open' if action_value > 0.5 else 'close'

            # 如果正在锁定状态，减少计数器并返回锁定状态
            if self.lock_counter > 0:
                self.lock_counter -= 1
                return self.locked_state
            # 检测状态变化
            if self.last_state is not None and self.last_state != current_state:
                # 状态发生变化，锁定新状态
                self.locked_state = current_state
                self.lock_counter = self.lock_duration  # 重置锁定计数器
                self.last_state = current_state
                return current_state

            # 没有状态变化，更新状态并返回
            self.last_state = current_state
            return current_state

        def reset(self):
            """重置稳定器状态"""
            self.last_state = None
            self.lock_counter = 0
            self.locked_state = None

    async def run_loop_async():
        # 初始化左右夹爪的状态稳定器
        left_gripper_stabilizer = GripperStateStabilizer(lock_duration=0)
        right_gripper_stabilizer = GripperStateStabilizer(lock_duration=0)

        await rtc.connect()
        obs_init, _ = await asyncio.to_thread(env.reset)
        rgb_history = await asyncio.to_thread(env.get_rgb_history_realman)
        rtc.rgb_history = rgb_history

        obs_for_next_step = None
        # 保存首次推理时的初始姿态，用于夹具闭合度异常时恢复
        initial_pose_obs = None
        # 启动后先检查共享指令：为空则空闲等待，非空则在循环内做首次全量重规划
        was_idle = True

        for step in range(1, cfg.env.num_steps + 1):
            step_time = 1.0 / cfg.env.max_hz if cfg.env.max_hz > 0 else 0
            
            logger.info(f"step: {step}")

            # 子任务轨迹回放进行中：暂停主循环（不推理、不动作），等回放结束后重新规划
            if replay_ctrl.is_replaying():
                if not was_idle:
                    logger.warning("[REPLAY] 轨迹回放进行中，主循环暂停执行")
                was_idle = True
                await rtc.pause()  # 清除在途重规划结果，防止旧轨迹在恢复后执行
                obs_for_next_step = None
                await asyncio.sleep(0.1)
                continue

            # 读取共享任务指令：为空则不执行任何动作，进入空闲等待
            cur_prompt = shared_prompt.get()
            if not cur_prompt:
                if not was_idle:
                    logger.warning("[PROMPT] 任务指令为空，停止执行动作，进入空闲等待")
                was_idle = True
                await rtc.pause()  # 清除在途重规划结果，防止旧轨迹在恢复后执行
                obs_for_next_step = None
                await asyncio.sleep(0.1)
                continue

            # 运行中任务切换：先停止旧 chunk，再从新观测全量规划；
            # 不能把上个任务的绝对 EEF 目标继续发给机械臂。
            if not was_idle and cur_prompt != rtc.prompt:
                logger.warning("[PROMPT] task changed during execution; discard old chunk and replan")
                await rtc.pause()
                was_idle = True
                obs_for_next_step = None
                rtc.prompt = cur_prompt
                continue

            # 空闲恢复（含首次启动）：拿到指令后先做一次全量重规划（无 prefix）
            if was_idle:
                logger.warning(f"[PROMPT] 收到任务指令: {cur_prompt}，开始全量重规划")
                rtc.prompt = cur_prompt
                obs_resume = await get_obs_with_retry()
                # 保存整个程序首次推理时的初始姿态（仅保存一次，后续空闲恢复不覆盖）
                if initial_pose_obs is None:
                    initial_pose_obs = obs_resume
                try:
                    res = await rtc.guided_inference(obs_resume, prefix_action=None, prefix_len=0)
                except (ConnectionError, OSError, ValueError) as e:
                    # 重试耗尽仍失败: 保持空闲状态，稍后重新尝试全量重规划
                    logger.error(f"[PROMPT] 全量重规划推理失败: {e}，保持空闲稍后重试")
                    await asyncio.sleep(1.0)
                    continue
                actions = res['action']
                async with rtc.lock:
                    rtc.A_cur = actions
                    rtc.t = 0
                # 本次 A_cur 由新任务指令生成
                rtc._A_cur_prompt = cur_prompt
                was_idle = False

            current_t = rtc.t
            should_get_obs = (current_t >= rtc.s_min - 3)

            act = await rtc.get_action(obs_for_next_step)
            if cfg.env.infermode == 'rot6d':
                act_euler_left = np.concatenate([act[:3], trans_utils.rot6d_to_euler_angle(act[3:9])])
                act_euler_right = np.concatenate([act[9:12], trans_utils.rot6d_to_euler_angle(act[12:18])])

                # 使用稳定器获取稳定的夹爪状态
                stable_gripper_left = left_gripper_stabilizer.get_stable_state(act[18])
                stable_gripper_right = right_gripper_stabilizer.get_stable_state(act[19])

                # 后台推理进行中时，夹具保持打开，不执行夹爪动作
                if rtc._reinference_pending:
                    gripper_positions = ["open", "open"]
                else:
                    gripper_positions = [stable_gripper_left, stable_gripper_right]

                cmd = {'ee_pos_quat': np.concatenate([act_euler_left, act_euler_right]),
                       'gripper_positions': gripper_positions}
            else:
                # 使用稳定器获取稳定的夹爪状态
                stable_gripper_left = left_gripper_stabilizer.get_stable_state(act[14])
                stable_gripper_right = right_gripper_stabilizer.get_stable_state(act[15])

                # 后台推理进行中时，夹具保持打开，不执行夹爪动作
                if rtc._reinference_pending:
                    gripper_positions = ["open", "open"]
                else:
                    gripper_positions = [stable_gripper_left, stable_gripper_right]

                cmd = {'joint_positions': np.array(act[:14]),
                       'gripper_positions': gripper_positions}

            start = time.monotonic()

            if should_get_obs:
                # 每次获取观测时同步读取共享任务指令，供下一次后台推理使用
                latest_prompt = shared_prompt.get()
                if latest_prompt and latest_prompt != rtc.prompt:
                    logger.warning(f"[PROMPT] 任务指令实时更新: {latest_prompt}")
                    rtc.prompt = latest_prompt
                # 耗时操作：获取图像
                obs_new, _, done, truncated, _ = await asyncio.to_thread(env.step, cmd)
                obs_for_next_step = obs_new  # 存起来，给下一次 get_action 用
                logger.warning(f"should_get_obs：True: 获取新观测")
                logger.warning(f"current_t={current_t}, rtc.s_min={rtc.s_min}")
            else:
                _, _, done, truncated, _ = await asyncio.to_thread(env.step, cmd, get_obs=False)
                obs_for_next_step = None  # 下一次不需要更新 rtc.o_cur
                logger.warning(f"should_get_obs：False: 不获取新观测")
                logger.warning(f"current_t={current_t}, rtc.s_min={rtc.s_min}")

            # obs, _, done, truncated, _ = await asyncio.to_thread(env.step, cmd)
            exec_time = time.monotonic() - start

            # ===== 夹具闭合度判断 =====
            # 获取观测用于夹具闭合度判断
            if should_get_obs:
                obs = obs_new
            else:
                obs = await get_obs_with_retry()

            arm_actpos = obs.get("arm_actpos", {}) if obs else {}
            left_arm_actpos = []
            right_arm_actpos = []
            if arm_actpos:
                left_arm_actpos = arm_actpos.get("left", [])
                right_arm_actpos = arm_actpos.get("right", [])

            # 小夹具 0 ~ 1000，0 完全闭合，1000 完全打开
            if left_arm_actpos and left_arm_actpos[0] < 50:
                logger.warning("左夹具闭合度接近0，重新打开左夹具，重新推理")
                reopen_cmd = dict(cmd)
                reopen_cmd['gripper_positions'] = ["open", stable_gripper_right]
                # 先重新打开夹具并获取当前观测
                obs_new, _, done, truncated, _ = await asyncio.to_thread(env.step, reopen_cmd)
                # 恢复到首次推理时的初始姿态，保持不动
                if initial_pose_obs is not None:
                    if cfg.env.infermode == 'rot6d':
                        restore_cmd = {'ee_pos_quat': np.array(initial_pose_obs["ee_pos_quat"]),
                                       'gripper_positions': ["open", "open"]}
                    else:
                        restore_cmd = {'joint_positions': np.array(initial_pose_obs["joint_positions"][:14]),
                                       'gripper_positions': ["open", "open"]}
                    obs_new, _, done, truncated, _ = await asyncio.to_thread(env.step, restore_cmd)
                # 等待 2 秒让机械臂在初始姿态完全稳定，避免剧烈波动
                await asyncio.sleep(2)
                # 稳定后重新获取观测，确保推理使用初始姿态下的清晰图像
                obs_new = await get_obs_with_retry()
                obs_for_next_step = obs_new
                # 夹具异常: 下一次推理禁用 RTC 前缀 (旧 chunk 后续动作不再可信)
                rtc.suppress_rtc_next()
                # 发起新的推理请求并阻塞等待新动作返回:
                # 等待期间不执行旧 chunk 的任何后续动作, 新动作到达后从第一步恢复执行
                await rtc.request_reinference(obs=obs_new)
                continue

            if right_arm_actpos and right_arm_actpos[0] < 50:
                logger.warning("右夹具闭合度接近0，重新打开右夹具，重新推理")
                reopen_cmd = dict(cmd)
                reopen_cmd['gripper_positions'] = [stable_gripper_left, "open"]
                # 先重新打开夹具并获取当前观测
                obs_new, _, done, truncated, _ = await asyncio.to_thread(env.step, reopen_cmd)
                # 恢复到首次推理时的初始姿态，保持不动
                if initial_pose_obs is not None:
                    if cfg.env.infermode == 'rot6d':
                        restore_cmd = {'ee_pos_quat': np.array(initial_pose_obs["ee_pos_quat"]),
                                       'gripper_positions': ["open", "open"]}
                    else:
                        restore_cmd = {'joint_positions': np.array(initial_pose_obs["joint_positions"][:14]),
                                       'gripper_positions': ["open", "open"]}
                    obs_new, _, done, truncated, _ = await asyncio.to_thread(env.step, restore_cmd)
                # 等待 2 秒让机械臂在初始姿态完全稳定，避免剧烈波动
                await asyncio.sleep(2)
                # 稳定后重新获取观测，确保推理使用初始姿态下的清晰图像
                obs_new = await get_obs_with_retry()
                obs_for_next_step = obs_new
                # 夹具异常: 下一次推理禁用 RTC 前缀 (旧 chunk 后续动作不再可信)
                rtc.suppress_rtc_next()
                # 发起新的推理请求并阻塞等待新动作返回:
                # 等待期间不执行旧 chunk 的任何后续动作, 新动作到达后从第一步恢复执行
                await rtc.request_reinference(obs=obs_new)
                continue

            if done or truncated:
                logger.warning(f"程序停止！！！！truncated={truncated}")
                logger.warning("Episode finished, stopping...")
                await rtc.stop()
                return  # 或 break +

            # if rtc.t >= rtc.s_min and not rtc.use_rtc:
            #     logger.info("降低频率！！！！")
            #     step_time = 1.0 / 1 # 非 RTC 模式：当大于固定步数，将执行频率变成很低

            if exec_time < step_time:
                await asyncio.sleep(step_time - exec_time)

            if step % 10 == 0:
                logger.info(f"[ASYNC] Step {step}/{cfg.env.num_steps}, exec {exec_time:.3f}s")

        await rtc.stop()

    async def run_loop_sync():
        # 初始化左右夹爪的状态稳定器
        left_gripper_stabilizer = GripperStateStabilizer(lock_duration=15)
        right_gripper_stabilizer = GripperStateStabilizer(lock_duration=15)

        await rtc.connect()
        obs, _ = await asyncio.to_thread(env.reset)

        step_time = 1.0 / cfg.env.max_hz if cfg.env.max_hz > 0 else 0
        total_steps = cfg.env.num_steps
        step = 0
        # 启动后先检查共享指令：为空则空闲等待，非空则在循环内做首次全量重规划
        was_idle = True
        # 当前episode的恢复参考姿态；新任务/回放/环境重置后刷新
        initial_pose_obs = None
        needs_episode_anchor = True
        episode_reset_pending = False

        while step < total_steps:
            # 子任务轨迹回放进行中：暂停主循环（不推理、不动作），等回放结束后重新规划
            if replay_ctrl.is_replaying():
                if not was_idle:
                    logger.warning("[REPLAY] 轨迹回放进行中，主循环暂停执行")
                episode_reset_pending = True
                needs_episode_anchor = True
                was_idle = True
                await asyncio.sleep(0.1)
                continue

            # 读取共享任务指令：为空则不执行任何动作，进入空闲等待
            cur_prompt = shared_prompt.get()
            if not cur_prompt:
                if not was_idle:
                    logger.warning("[PROMPT] 任务指令为空，停止执行动作，进入空闲等待")
                episode_reset_pending = True
                needs_episode_anchor = True
                was_idle = True
                await asyncio.sleep(0.1)
                continue

            if was_idle:
                logger.warning(f"[PROMPT] 收到任务指令: {cur_prompt}，开始全量重规划")
            if cur_prompt != rtc.prompt:
                logger.warning(f"[PROMPT] 任务指令实时更新: {cur_prompt}")
                rtc.prompt = cur_prompt
                episode_reset_pending = True
                needs_episode_anchor = True
                was_idle = True
            # One fresh observation and ONE inference per executed chunk.
            # Retry alone does not create a new episode; task/replay/reset does.
            if episode_reset_pending:
                await rtc.reset_episode()
                episode_reset_pending = False
            obs = await get_obs_with_retry()
            if needs_episode_anchor:
                from copy import deepcopy
                initial_pose_obs = deepcopy(obs)
                left_gripper_stabilizer.reset()
                right_gripper_stabilizer.reset()
                needs_episode_anchor = False
            was_idle = False

            # 同步模式：推理期间机器人停下等待，没有已committed的步，
            # 所以 prefix_len=0，走全量重规划路径。
            try:
                result = await rtc.guided_inference(obs, prefix_action=None, prefix_len=0)
            except (ConnectionError, OSError, ValueError) as e:
                # 重试耗尽仍失败: 回到空闲等待，下一轮循环自动重新规划
                logger.error(f"[SYNC] 推理请求持续失败: {e}，回到空闲等待后自动重试")
                was_idle = True
                await asyncio.sleep(1.0)
                continue
            # logger.error(f"服务端响应为: {result}")
            actions = result['action']

            # 标记本轮是否触发夹具闭合度异常（仅在异常时等待2秒）
            gripper_anomaly_triggered = False
            for i in range(rtc.s_min):
                if step >= total_steps:
                    break
                # 指令清空或任务切换：立刻停止旧 chunk，重新采集观测/全量规划。
                current_prompt = shared_prompt.get()
                if not current_prompt or current_prompt != rtc.prompt:
                    logger.warning("[PROMPT] instruction changed/cleared; abort old chunk")
                    was_idle = True
                    episode_reset_pending = True
                    needs_episode_anchor = True
                    break
                start = time.monotonic()
                act = actions[i]
                logger.info(f"current action chunk: {actions}")
                act_euler_left = np.concatenate([act[:3], trans_utils.rot6d_to_euler_angle(act[3:9])])
                act_euler_right = np.concatenate([act[9:12], trans_utils.rot6d_to_euler_angle(act[12:18])])

                # 使用稳定器获取稳定的夹爪状态
                stable_gripper_left = left_gripper_stabilizer.get_stable_state(act[18])
                stable_gripper_right = right_gripper_stabilizer.get_stable_state(act[19])

                cmd = {'ee_pos_quat': np.concatenate([act_euler_left, act_euler_right]),
                       'gripper_positions': [stable_gripper_left, stable_gripper_right]}
                fresh_obs = None

                is_last_step_in_chunk = (i == rtc.s_min - 1)

                if is_last_step_in_chunk:
                    obs, _, done, truncated, _ = await asyncio.to_thread(env.step, cmd)
                    fresh_obs = obs
                else:
                    _, _, done, truncated, _ = await asyncio.to_thread(env.step, cmd, get_obs=False)

                # obs, _, done, truncated, _ = await asyncio.to_thread(env.step, cmd)
                exec_time = time.monotonic() - start
                step += 1

                # 处理环境重置
                if done or truncated:
                    logger.info(f"程序停止！！！！truncated={truncated}")
                    obs, _ = await asyncio.to_thread(env.reset)
                    left_gripper_stabilizer.reset()
                    right_gripper_stabilizer.reset()
                    episode_reset_pending = True
                    needs_episode_anchor = True
                    was_idle = True
                    break  # 跳出当前chunk循环，重新获取动作

                # ===== 夹具闭合度判断 =====
                arm_actpos = fresh_obs.get("arm_actpos", {}) if fresh_obs is not None else {}
                left_arm_actpos = []
                right_arm_actpos = []
                if arm_actpos:
                    left_arm_actpos = arm_actpos.get("left", [])
                    right_arm_actpos = arm_actpos.get("right", [])

                # 小夹具 0 ~ 1000，0 完全闭合，1000 完全打开
                if left_arm_actpos and left_arm_actpos[0] < 50:
                    logger.warning("左夹具闭合度接近0，重新打开左夹具，重新推理")
                    reopen_cmd = dict(cmd)
                    reopen_cmd['gripper_positions'] = ["open", stable_gripper_right]
                    _, _, done, truncated, _ = await asyncio.to_thread(env.step, reopen_cmd, get_obs=False)
                    # 保留原恢复动作，但参考姿态属于当前episode
                    if initial_pose_obs is not None:
                        restore_cmd = {'ee_pos_quat': np.array(initial_pose_obs["ee_pos_quat"]),
                                       'gripper_positions': ["open", "open"]}
                        _, _, done, truncated, _ = await asyncio.to_thread(env.step, restore_cmd, get_obs=False)
                    # 夹具异常: 下一次推理禁用 RTC 前缀 (旧 chunk 后续动作不再可信)
                    rtc.suppress_rtc_next()
                    gripper_anomaly_triggered = True
                    left_gripper_stabilizer.reset()
                    right_gripper_stabilizer.reset()
                    episode_reset_pending = True
                    needs_episode_anchor = True
                    was_idle = True
                    break

                if right_arm_actpos and right_arm_actpos[0] < 50:
                    logger.warning("右夹具闭合度接近0，重新打开右夹具，重新推理")
                    reopen_cmd = dict(cmd)
                    reopen_cmd['gripper_positions'] = [stable_gripper_left, "open"]
                    _, _, done, truncated, _ = await asyncio.to_thread(env.step, reopen_cmd, get_obs=False)
                    # 保留原恢复动作，但参考姿态属于当前episode
                    if initial_pose_obs is not None:
                        restore_cmd = {'ee_pos_quat': np.array(initial_pose_obs["ee_pos_quat"]),
                                       'gripper_positions': ["open", "open"]}
                        _, _, done, truncated, _ = await asyncio.to_thread(env.step, restore_cmd, get_obs=False)
                    # 夹具异常: 下一次推理禁用 RTC 前缀 (旧 chunk 后续动作不再可信)
                    rtc.suppress_rtc_next()
                    gripper_anomaly_triggered = True
                    left_gripper_stabilizer.reset()
                    right_gripper_stabilizer.reset()
                    episode_reset_pending = True
                    needs_episode_anchor = True
                    was_idle = True
                    break

                if exec_time < step_time:
                    await asyncio.sleep(step_time - exec_time)

                if step % 10 == 0:
                    logger.info(f"[SYNC] Step {step}/{total_steps}, exec {exec_time:.3f}s")

            # The next outer iteration fetches observations and performs inference.
            # Do not infer here: that result used to be overwritten at loop entry.
            if gripper_anomaly_triggered:
                await asyncio.sleep(2)

        await rtc.stop()

    # choose mode based on cfg.rtc.async_mode
    if cfg.rtc.async_mode:
        asyncio.run(run_loop_async())
    else:
        asyncio.run(run_loop_sync())


if __name__ == "__main__":
    main()
