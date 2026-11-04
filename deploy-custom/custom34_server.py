#!/usr/bin/env python3
"""Agentic-WAM custom robot deployment: raw state34 + RGB -> absolute EEF action20.

Run from repository root, using the *same* task/data/model configuration as training.
Uses existing CustomUnifiedEEFTransform; does NOT run YAM FK/IK or the 14D server.

POST /act JSON:
  {"instruction": str, "state": [34 floats],
   "images": {"head": {"format":"png", "data":"<base64>"}, "left":..., "right":...}}
  Optional RTC (ONLY with an asynchronous client): "committed": [[20 floats] * K], "freeze": int
Response: {"status": "success"|"error", "action": [[20 floats] * T]|null,
           "latency": float (milliseconds, server time), "message": str}

POST /reset {} ; GET /health.
Images and state are Python numpy arrays at the client API, encoded for JSON transport.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]  # file must be at <repo>/deploy/custom34_server.py
# Support a file placed elsewhere only with explicit AGENTICWAM_ROOT.
ROOT = Path(os.environ.get("AGENTICWAM_ROOT", str(ROOT))).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "robotwin" / "deploy"))
sys.path.insert(0, str(ROOT / "deploy"))

LOG = logging.getLogger("agenticwam.custom34")
Image.MAX_IMAGE_PIXELS = 16_000_000
ROLE_NAMES = ("head", "left", "right")
SIM_NAMES = ("head_camera", "left_camera", "right_camera")
CAMERA_ALIASES = {
    "cam_high": "head", "head_camera": "head", "head": "head", "top": "head",
    "cam_left_wrist": "left", "left_camera": "left", "left": "left",
    "cam_right_wrist": "right", "right_camera": "right", "right": "right",
}


def check_state(obj: Any) -> np.ndarray:
    v = np.asarray(obj, dtype=np.float32)
    if v.shape != (34,) or not np.isfinite(v).all():
        raise ValueError(f"state must contain 34 finite float32 values, got {v.shape}")
    for offset in (17, 26):
        check_rotation6d(v[offset:offset+6])
    return v


def check_rotation6d(value: np.ndarray) -> None:
    first, second = value[..., :3], value[..., 3:6]
    if (np.linalg.norm(first, axis=-1) < 1e-6).any() or (np.linalg.norm(np.cross(first, second), axis=-1) < 1e-6).any():
        raise ValueError('rotation6D columns must be nonzero and linearly independent')


def check_image(img: np.ndarray, role: str) -> np.ndarray:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"images.{role} must be HxWx3 uint8 RGB, got {img.shape} {img.dtype}")
    h, w, _ = img.shape
    if not (16 <= h <= 4096 and 16 <= w <= 4096):
        raise ValueError(f"images.{role}: unreasonable H/W {h}x{w}")
    return np.ascontiguousarray(img)


def decode_image(field: Any, role: str) -> np.ndarray:
    # Accept a base64 string or {format: png/jpeg, data: base64}.
    if isinstance(field, dict):
        fmt = str(field.get("format", "png")).lower()
        field = field.get("data")
    else:
        fmt = None
    if fmt is not None and fmt not in ("png", "jpeg", "jpg"):
        raise ValueError(f"images.{role}: format must be png or jpeg")
    if not isinstance(field, str):
        raise ValueError(f"images.{role}: expected base64 string or {{format,data}}")
    try:
        encoded = base64.b64decode(field, validate=True)
        if len(encoded) > 12 * 1024 * 1024:
            raise ValueError("encoded image too large")
        with Image.open(io.BytesIO(encoded)) as p:
            if p.format not in ("PNG", "JPEG"):
                raise ValueError(f"images.{role}: unsupported image encoding {p.format}")
            p.load()
            out = np.array(p.convert("RGB"), dtype=np.uint8)
    except (binascii.Error, UnidentifiedImageError, OSError) as e:
        raise ValueError(f"images.{role}: invalid encoded image: {e}") from e
    return check_image(out, role)


def expected_dim(value: Any) -> int:
    # shape_meta raw_shape may be an int (34), [34], or (34,).
    if isinstance(value, (int, np.integer)):
        return int(value)
    if len(value) == 1:
        return int(value[0])
    raise ValueError(f"expected vector dimension, got shape {value!r}")


def train_camera_order(cfg, override: str | None) -> tuple[str, str, str]:
    if override:
        order = tuple(s.strip().lower() for s in override.split(","))
    else:
        image_metas = cfg.data.train.shape_meta.images
        keys = [str(m.key) for m in image_metas]
        try:
            order = tuple(CAMERA_ALIASES[k] for k in keys)
        except KeyError as e:
            raise ValueError(
                f"unknown training image key {e}; pass --camera-order head,left,right "
                f"in EXACT training camera sequence (train keys={keys})") from e
    if len(order) != 3 or set(order) != set(ROLE_NAMES):
        raise ValueError(f"camera order must be a permutation of head,left,right, got {order}")
    return order


class Custom34Runtime:
    def __init__(self, args):
        if not ROOT.joinpath("robotwin", "deploy", "deploy_policy.py").is_file():
            raise FileNotFoundError(f"AGENTICWAM_ROOT={ROOT} does not contain robotwin/deploy/deploy_policy.py")
        ckpt = Path(args.ckpt).resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint absent: {ckpt} (refusing smoke-mode random actions)")
        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
        os.environ.setdefault("AGENTICWAM_EVAL_PREPROC", "parity")
        os.chdir(ROOT)

        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
        import deploy_policy as dp
        from agenticwam.datasets.lerobot.transforms.custom_unified_eef import (
            CustomUnifiedEEFTransform,
        )

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable but requested {args.device}")
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="train", overrides=[f"task={args.task}",
                    f"data.dataset_root={args.dataset}"] + args.override)
        OmegaConf.set_struct(cfg, False)
        self.cfg = cfg
        self.cam_order = train_camera_order(cfg, args.camera_order)

        # Fail early if the deployment recipe differs from the custom robot's training recipe.
        sm = cfg.data.train.shape_meta
        if [int(x) for x in cfg.data.train.video_size] != [224, 320]:
            raise ValueError("deploy_policy.py fixes the video size to 224x320; training video_size differs")
        if expected_dim(sm.state[0].raw_shape) != 34 or expected_dim(sm.action[0].raw_shape) != 20:
            raise ValueError("training shape_meta must have raw state=34/action=20")
        if expected_dim(sm.state[0].shape) != 64 or expected_dim(sm.action[0].shape) != 64:
            raise ValueError("training shape_meta must have unified state=64/action=64")
        if int(cfg.model.proprio_dim) != 64 or int(cfg.model.action_dim) != 64 or int(cfg.model.n_cameras) != 3:
            raise ValueError("model requires proprio_dim=action_dim=64 and n_cameras=3")
        if args.rtc and (not args.fast_joint or str(cfg.model.get("code_objective", "")) != "joint"):
            raise ValueError("RTC requires --fast-joint and model.code_objective=joint")
        targets = [str(t.get("_target_", "")) for t in cfg.data.train.processor.action_state_transforms]
        if not any(t.endswith("custom_unified_eef.CustomUnifiedEEFTransform") for t in targets):
            raise ValueError(f"training config does not use CustomUnifiedEEFTransform: {targets}")

        self.stats_file = str(args.dataset_stats or cfg.data.train.pretrained_norm_stats)
        if not Path(self.stats_file).is_file():
            raise FileNotFoundError(f"training normalization stats missing: {self.stats_file}")
        nf = int(cfg.data.train.num_frames)
        self.horizon = int(args.action_horizon or (nf - 1))
        if self.horizon != nf - 1:
            # Per-step action normalization is normally tied to training horizon.
            raise ValueError(f"action_horizon must match training horizon ({nf - 1})")
        fr = int(cfg.data.train.get("action_video_freq_ratio", 4))
        if (nf - 1) % fr:
            raise ValueError(f"num_frames={nf} must satisfy (num_frames-1) % action_video_freq_ratio={fr} == 0")
        n_v = (nf - 1) // fr + 1
        LOG.info("building 3-expert model (this may load large weights)")
        self.policy = dp.AgenticWAMCosmosRobotWinPolicy(
            model_cfg=cfg.model, processor_cfg=cfg.data.train.processor,
            checkpoint_path=str(ckpt), dataset_stats_path=self.stats_file,
            device=args.device, action_horizon=self.horizon, replan_steps=args.replan_steps,
            num_inference_steps=args.num_inference_steps, sigma_shift=None,
            seed=None, text_ckpt_dir=str(cfg.paths.text_encoder), text_max_len=128,
            n_cameras=3, num_video_frames=n_v,
        )
        transforms = self.policy.processor.action_state_transforms
        custom = [t for t in transforms if isinstance(t, CustomUnifiedEEFTransform)]
        if len(custom) != 1:
            raise RuntimeError(f"expected exactly one CustomUnifiedEEFTransform, got {transforms}")
        self.transform = custom[0]
        self.key = str(sm.action[0].key)
        if self.key != str(sm.state[0].key):
            raise ValueError("state/action keys must match for CustomUnifiedEEFTransform")
        self.lock = threading.Lock()
        self.fast_joint = bool(args.fast_joint)
        self.rtc_enabled = bool(args.rtc)
        self.rtc_count = 0
        if self.fast_joint:
            from common.fast_joint_infer import install, _exotic_mode_active
            if self.rtc_enabled and _exotic_mode_active(self.policy.model):
                raise ValueError("RTC requested but fast_joint is in exotic/eager fallback mode; "
                                 "RTC is not applied in the original inference path")
            install(self.policy.model)
        self.code_log_dir = Path(args.code_log_dir).expanduser().resolve() if args.code_log_dir else None
        if self.code_log_dir:
            self.code_log_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.server_id = uuid.uuid4().hex
        self._instruction = None
        self.code_objective = str(cfg.model.get('code_objective', 'joint'))
        self.history_modes = [name for name in ('AGENTICWAM_SLICE_MODE', 'AGENTICWAM_SLICE_SUMMARY') if os.environ.get(name)]
        self.policy.reset()
        LOG.info("READY: 34D raw state -> 64D normalized proprio; 64D delta -> 20D absolute EEF")
        LOG.info("camera order %s; chunk=%d; RTC server support=%s", self.cam_order, self.horizon, self.rtc_enabled)
        LOG.info('instance=%s code_objective=%s fast_joint=%s code_memory=%s history_modes=%s',
                 self.server_id, self.code_objective, self.fast_joint,
                 bool(getattr(self.policy.model, 'code_memory_enabled', False)), self.history_modes)

    def _prepare_rtc(self, raw_state: np.ndarray, body: dict):
        """Optional *inference-only* inpainting guidance. Caller must time-align committed rows.

        `committed[k]` must correspond to row `k` of the NEW chunk based at THIS observation.
        Do not blindly send the unshifted previous chunk.
        """
        import torch
        if "committed" not in body and "freeze" not in body:
            return False
        if not self.rtc_enabled:
            raise ValueError("RTC fields supplied but server RTC disabled; restart with --fast-joint --rtc")
        if "committed" not in body or "freeze" not in body:
            raise ValueError("RTC requires BOTH committed and freeze")
        rows = np.asarray(body["committed"], dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 20 or not np.isfinite(rows).all():
            raise ValueError("committed must be finite [K,20] absolute EEF targets")
        for offset in (3, 12):
            check_rotation6d(rows[:, offset:offset+6])
        k = int(rows.shape[0])
        if not (1 <= k <= min(24, self.horizon-1)):
            raise ValueError(f"committed K must be within 1..{min(24, self.horizon-1)}")
        freeze = body["freeze"]
        if isinstance(freeze, bool) or not isinstance(freeze, int) or not (0 <= freeze <= k):
            raise ValueError("freeze must be integer 0..K")
        # Convert absolute action20 -> delta64 anchored to the exact new observation state.
        raw_a = torch.as_tensor(rows, dtype=torch.float32)[None]   # [1,K,20]
        raw_s = torch.as_tensor(raw_state, dtype=torch.float32)[None]  # [1,34]
        delta = self.transform._action64(raw_a, raw_s)[0]
        norm = self.policy.processor.normalizer.normalizers["action"][self.key]
        scale, offset = norm.scale, norm.offset
        if scale.ndim == 2:
            scale, offset = scale[:k], offset[:k]
        targets = (delta * scale + offset).clamp(-5, 5)
        weights = np.zeros(k, dtype=np.float32)
        weights[:freeze] = 1.0
        if k > freeze:
            weights[freeze:] = np.linspace(0.8, 0.0, k - freeze, endpoint=False)
        self.policy.model._rtc_targets = targets
        self.policy.model._rtc_weights = torch.as_tensor(weights)
        return True

    def _delta64_to_abs20(self, delta64: np.ndarray, raw_state: np.ndarray) -> np.ndarray:
        import torch
        delta64 = np.asarray(delta64, dtype=np.float32)
        if delta64.shape != (self.horizon, 64) or not np.isfinite(delta64).all():
            raise ValueError(f"model output expected [{self.horizon},64] finite, got {delta64.shape}")
        # The model wrapper already reversed the action normalizer. DO NOT undo it twice.
        # The custom transform backward uses the absolute *un-normalized* unified64 state.
        base64 = self.transform._state64(torch.as_tensor(raw_state, dtype=torch.float32))[None, None]
        batch = {"state": {self.key: base64},
                 "action": {self.key: torch.as_tensor(delta64)[None]}}
        absolute20 = self.transform.backward_action20(batch)["action"][self.key][0]
        out = np.asarray(absolute20.numpy(), dtype=np.float32)
        if out.shape != (self.horizon, 20) or not np.isfinite(out).all():
            raise ValueError(f"conversion to raw action20 failed: {out.shape}")
        for offset in (3, 12):
            check_rotation6d(out[:, offset:offset+6])
        return out

    def _reset_episode_unlocked(self) -> None:
        self.policy.reset()
        self.policy._last_code = None
        self.policy.model._last_code_canvas = None
        self.policy.model._rtc_targets = None
        self.policy.model._rtc_weights = None
        self._instruction = None

    def infer(self, body: dict) -> tuple[np.ndarray, float, str | None]:
        start = time.perf_counter()
        if not isinstance(body, dict):
            raise ValueError("POST /act requires a JSON object")
        instruction = body.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty English task string")
        state = check_state(body.get("state"))
        imgs = body.get("images")
        if not isinstance(imgs, dict) or set(imgs) != set(ROLE_NAMES):
            raise ValueError("images must have exactly head, left, right")
        frames = {role: decode_image(imgs[role], role) for role in ROLE_NAMES}
        # Wrapper's SIM_NAMES are fixed, but we map slots in EXACT training camera order.
        ob = {"observation": {
            SIM_NAMES[i]: {"rgb": frames[role]}
            for i, role in enumerate(self.cam_order)},
            "joint_action": {"vector": state.tolist()}}
        with self.lock:
            staged = False
            try:
                if self._instruction is not None and self._instruction != instruction:
                    self._reset_episode_unlocked()
                # Custom stage labels are observation-dependent. The non-joint
                # policy's once-per-episode code cache must not pin an old stage.
                if self.code_objective != 'joint':
                    self.policy.code_cache.clear()
                staged = self._prepare_rtc(state, body)
                predicted_delta64 = self.policy._infer_action_chunk(ob, instruction)
                action20 = self._delta64_to_abs20(predicted_delta64, state)
                code = getattr(self.policy, "_last_code", None)
                self.policy.step_count += 1
                self.calls += 1
                self._instruction = instruction
                if staged:
                    self.rtc_count += 1
                # Log full newly generated code on EVERY inference, not only episode step 0.
                try:
                    print(f"\n===== POLICY CODE | call={self.calls} episode={self.policy.episode_count} "
                          f"step={self.policy.step_count} =====\n"
                          f"{code if code is not None else '[NO CODE GENERATED]'}\n"
                          f"===== END POLICY CODE =====\n", flush=True)
                except (OSError, ValueError) as exc:
                    LOG.warning('Console code log failed after successful inference: %s', exc)
                if self.code_log_dir is not None:
                    path = self.code_log_dir / f"episode{self.policy.episode_count:04d}_call{self.calls:06d}.txt"
                    try:
                        path.write_text(str(code) if code is not None else "", encoding="utf-8")
                    except OSError as exc:
                        LOG.warning('Code log write failed after successful inference: %s', exc)
            finally:
                # RTC staging must NEVER leak into later requests.
                self.policy.model._rtc_targets = None
                self.policy.model._rtc_weights = None
        latency_ms = 1000.0 * (time.perf_counter() - start)
        return action20, latency_ms, code

    def reset(self) -> None:
        with self.lock:
            self._reset_episode_unlocked()


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, help="Hydra training task name, e.g. posttrain_custom")
    ap.add_argument("--dataset", required=True, help="same dataset root as training")
    ap.add_argument("--dataset-stats", default=None)
    ap.add_argument("--override", action="append", default=[], help="repeat for Hydra overrides")
    ap.add_argument("--camera-order", default=None, help="exact training camera order: head,left,right")
    ap.add_argument("--action-horizon", type=int, default=None)
    ap.add_argument("--replan-steps", type=int, default=24)
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fast-joint", action="store_true", help="install compiled-step joint inference")
    ap.add_argument("--rtc", action="store_true", help="accept optional committed/freeze fields")
    ap.add_argument("--code-log-dir", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-body-mb", type=int, default=24)
    ap.add_argument("--max-wait-sec", type=float, default=5.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # No API authentication; default bind is loopback only.
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        LOG.warning("NO AUTHENTICATION: listen=%s:%s; restrict network access with firewall/VPN, never expose to the internet", args.host, args.port)
    rt = Custom34Runtime(args)
    gate = threading.BoundedSemaphore(1)
    max_body = args.max_body_mb * 1024 * 1024

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "AgenticWAM-Custom34/1"

        def log_message(self, fmt: str, *values):
            LOG.info("%s %s", self.address_string(), fmt % values)

        def send_json(self, status: int, payload: dict):
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def send_error_response(self, status: int, err: str, latency: float = 0.0):
            self.send_json(status, {"status": "error", "action": None,
                                    "latency": round(latency, 2), "message": err})

        def read_body(self) -> dict:
            length = self.headers.get("Content-Length", "")
            if not length.isdecimal() or not (0 < int(length) <= max_body):
                raise ValueError(f"Content-Length must be 1..{max_body} bytes")
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise ValueError("Content-Type must be application/json")
            obj = json.loads(self.rfile.read(int(length)))
            if not isinstance(obj, dict):
                raise ValueError("JSON body must be an object")
            return obj

        def do_GET(self):
            if self.path != "/health":
                return self.send_error_response(404, "GET /health only")
            self.send_json(200, {"status": "success", "state_dim": 34,
                                 "input_camera_order": rt.cam_order, "model_dim": 64,
                                 "action_dim": 20, "chunk_size": rt.horizon,
                                 "calls": rt.calls, "rtc_enabled": rt.rtc_enabled,
                                 "rtc_guided_calls": rt.rtc_count, "server_id": rt.server_id,
                                 "episode_count": rt.policy.episode_count,
                                 "code_objective": rt.code_objective, "fast_joint": rt.fast_joint,
                                 "code_memory": bool(getattr(rt.policy.model, 'code_memory_enabled', False)),
                                 "history_modes": rt.history_modes})

        def do_POST(self):
            t0 = time.perf_counter()
            if self.path not in ("/act", "/reset"):
                return self.send_error_response(404, "POST /act or /reset only")
            try:
                body = self.read_body()
            except (ValueError, json.JSONDecodeError) as exc:
                return self.send_error_response(400, str(exc), 1000*(time.perf_counter()-t0))
            if not gate.acquire(timeout=args.max_wait_sec):
                return self.send_error_response(503, "GPU busy", 1000*(time.perf_counter()-t0))
            try:
                if self.path == "/reset":
                    rt.reset()
                    return self.send_json(200, {"status":"success", "action": None,
                                                "latency": round(1000*(time.perf_counter()-t0), 2),
                                                "message": "episode reset", "server_id": rt.server_id,
                                                "episode_count": rt.policy.episode_count})
                action, _, _ = rt.infer(body)
                return self.send_json(200, {"status": "success", "action": action.tolist(),
                                            "latency": round(1000*(time.perf_counter()-t0), 2),
                                            "message": "", "server_id": rt.server_id,
                                            "episode_count": rt.policy.episode_count})
            except (ValueError, KeyError, TypeError) as exc:
                LOG.warning("invalid inference request: %s", exc)
                return self.send_error_response(400, str(exc), 1000*(time.perf_counter()-t0))
            except Exception as exc:
                LOG.error("inference error:\n%s", traceback.format_exc())
                return self.send_error_response(500, str(exc), 1000*(time.perf_counter()-t0))
            finally:
                gate.release()

    srv = InferenceServer((args.host, args.port), Handler)
    LOG.info("serving at http://%s:%d (POST /act, /reset; GET /health)", args.host, args.port)
    try:
        srv.serve_forever(poll_interval=0.1)
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
