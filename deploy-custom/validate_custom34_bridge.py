#!/usr/bin/env python3
"""GPU-free roundtrip verification using the SAME CustomUnifiedEEFTransform as training.
Run from project root: python deploy/validate_custom34_bridge.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agenticwam.datasets.lerobot.transforms.custom_unified_eef import (  # noqa: E402
    CustomUnifiedEEFTransform,
)


def eef(xyz, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    # R = [[c,-s,0],[s,c,0],[0,0,1]], six dims are first two columns.
    return [*xyz, c, s, 0.0, -s, c, 0.0]


def run_test():
    state = np.zeros(34, dtype=np.float32)
    state[14:23] = eef([0.2, -0.1, 0.15], 0.35)
    state[23:32] = eef([-0.2, 0.25, 0.12], -0.4)
    state[32:34] = [0.25, 0.75]
    target = np.zeros((32, 20), dtype=np.float32)
    for i in range(32):
        t = i / 31.0
        target[i, 0:9] = eef([0.2 + 0.04*t, -0.1 + 0.01*t, 0.15 + 0.02*t], 0.35 + 0.1*t)
        target[i, 9:18] = eef([-0.2 - 0.03*t, 0.25 + 0.02*t, 0.12], -0.4 + 0.2*t)
        target[i, 18:20] = [0.25 + 0.4*t, 0.75 - 0.5*t]

    transform = CustomUnifiedEEFTransform(keys=["default"])
    raw_state = torch.as_tensor(state).reshape(1, 1, 34)
    raw_action = torch.as_tensor(target).reshape(1, 32, 20)
    transformed = transform.forward({
        "state": {"default": raw_state.clone()},
        "action": {"default": raw_action.clone()},
    })
    u_state = transformed["state"]["default"]
    u_action = transformed["action"]["default"]
    assert u_state.shape == (1, 1, 64), u_state.shape
    assert u_action.shape == (1, 32, 64), u_action.shape
    restored = transform.backward_action20({
        "state": {"default": u_state.clone()},
        "action": {"default": u_action.clone()},
    })["action"]["default"][0]
    err = float((restored - raw_action[0]).abs().max())
    print(f"raw state:  {tuple(raw_state.shape)} -> unified state: {tuple(u_state.shape)}")
    print(f"raw action: {tuple(raw_action.shape)} -> unified action: {tuple(u_action.shape)}")
    print(f"action20 roundtrip maximum abs error: {err:.8f}")
    assert err < 1e-4, "raw20 -> delta64 -> absolute20 failed"
    # Hands must be zero in the canonical 64D output.
    assert torch.count_nonzero(u_action[..., 9:31]) == 0
    assert torch.count_nonzero(u_action[..., 41:63]) == 0
    print("PASS: dimensions, rotation6d, EEF delta/absolute inverse, hand-zero slots")


if __name__ == "__main__":
    run_test()
