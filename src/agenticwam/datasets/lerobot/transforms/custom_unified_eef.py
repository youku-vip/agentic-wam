"""Custom bimanual gripper robot: raw state 34D / action 20D <-> unified EEF 64D.

Drop this file into:
    src/agenticwam/datasets/lerobot/transforms/custom_unified_eef.py

Raw observation.state [34], in this exact order:
    [0:7]    left arm joints (unused in the unified EEF representation)
    [7:14]   right arm joints (unused in the unified EEF representation)
    [14:23]  left EEF [xyz(3), rotation6d(6)]
    [23:32]  right EEF [xyz(3), rotation6d(6)]
    [32]     left gripper
    [33]     right gripper

Raw action [20], ABSOLUTE EEF target:
    [0:9]    left target [xyz(3), rotation6d(6)]
    [9:18]   right target [xyz(3), rotation6d(6)]
    [18]     left gripper absolute target
    [19]     right gripper absolute target

Both rotation6d fields are the FIRST TWO COLUMNS of a rotation matrix,
ordered [column0.xyz, column1.xyz], in the SAME reference frame.

Unified 64D, identical to YamUnifiedEEFTransform:
    left:  EEF[0:9], hand[9:31] = 0, gripper[31]
    right: EEF[32:41], hand[41:63] = 0, gripper[63]

State uses ABSOLUTE EEF; action EEF is DELTA relative to the chunk's first
state, expressed in the first state's coordinate frame. Gripper targets stay
ABSOLUTE. A backward() call mirrors the project's YAM implementation:
it converts delta EEF actions to ABSOLUTE unified-64D EEF targets.
Call absolute_eef64_to_action20() afterwards if your robot accepts raw 20D
absolute end-effector targets. The live robot controller must separately
enforce safety limits, coordinate-frame consistency, and gripper conventions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


STATE_DIM = 34
ACTION_DIM = 20
UNIFIED_DIM = 64
HAND_DIM = 22

STATE_LEFT_EEF = slice(14, 23)
STATE_RIGHT_EEF = slice(23, 32)
STATE_LEFT_GRIP = slice(32, 33)
STATE_RIGHT_GRIP = slice(33, 34)

ACTION_LEFT_EEF = slice(0, 9)
ACTION_RIGHT_EEF = slice(9, 18)
ACTION_LEFT_GRIP = slice(18, 19)
ACTION_RIGHT_GRIP = slice(19, 20)

UNIFIED_LEFT_EEF = slice(0, 9)
UNIFIED_LEFT_GRIP = slice(31, 32)
UNIFIED_RIGHT_EEF = slice(32, 41)
UNIFIED_RIGHT_GRIP = slice(63, 64)


def _check_last_dim(tensor: torch.Tensor, dimension: int, name: str) -> None:
    if tensor.shape[-1] != dimension:
        raise ValueError(
            f"{name}: expected last dimension {dimension}, got {tuple(tensor.shape)}"
        )


def _rot6d_to_R(d6: torch.Tensor) -> torch.Tensor:
    """First two COLUMNS (6D) -> orthonormal rotation [..., 3, 3]."""
    _check_last_dim(d6, 6, "rotation6d")
    a1, a2 = d6[..., :3], d6[..., 3:6]
    eps = torch.finfo(d6.dtype).eps
    b1 = F.normalize(a1, dim=-1, eps=eps)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _eef9_to_Rt(pose9: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """EEF [xyz(3), rotation6d(6)] -> (rotation matrix, position)."""
    _check_last_dim(pose9, 9, "eef9")
    return _rot6d_to_R(pose9[..., 3:9]), pose9[..., 0:3]


def _Rt_to_eef9(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(rotation matrix, position) -> EEF [xyz, col0, col1]."""
    return torch.cat((t, R[..., :, 0], R[..., :, 1]), dim=-1)


def _matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


def _matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)


def _empty_hand(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_zeros(*ref.shape[:-1], HAND_DIM)


def absolute_eef64_to_action20(absolute64: torch.Tensor) -> torch.Tensor:
    """Extract native 20D ABSOLUTE targets from absolute unified 64D poses.

    This is NOT a joint-space controller or an inverse-kinematics solver.
    Only call it on absolute EEF outputs (e.g. AFTER transform.backward()).
    """
    _check_last_dim(absolute64, UNIFIED_DIM, "absolute EEF64")
    return torch.cat(
        (
            absolute64[..., UNIFIED_LEFT_EEF],
            absolute64[..., UNIFIED_RIGHT_EEF],
            absolute64[..., UNIFIED_LEFT_GRIP],
            absolute64[..., UNIFIED_RIGHT_GRIP],
        ),
        dim=-1,
    )


class CustomUnifiedEEFTransform:
    """AgenticWAMProcessor-compatible raw(34D,20D) -> unified64 transform."""

    def __init__(self, keys):
        self.keys = list(keys)

    def _state64(self, s: torch.Tensor) -> torch.Tensor:
        """Raw observation.state [...,34] -> unified state [...,64]."""
        _check_last_dim(s, STATE_DIM, "observation.state")
        # Re-orthogonalize noisy 6D rotation columns before encoding.
        RL, tL = _eef9_to_Rt(s[..., STATE_LEFT_EEF])
        RR, tR = _eef9_to_Rt(s[..., STATE_RIGHT_EEF])
        hand = _empty_hand(s)
        result = torch.cat(
            (
                _Rt_to_eef9(RL, tL),
                hand,
                s[..., STATE_LEFT_GRIP],
                _Rt_to_eef9(RR, tR),
                hand,
                s[..., STATE_RIGHT_GRIP],
            ),
            dim=-1,
        )
        _check_last_dim(result, UNIFIED_DIM, "unified state")
        return result

    def _action64(self, a: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Absolute targets [...,T,20], base state [...,34] -> delta [...,T,64].

        The base is the CURRENT state at the start of the action chunk.
        Translation and rotation deltas are both expressed in this base frame.
        """
        _check_last_dim(a, ACTION_DIM, "action")
        _check_last_dim(base, STATE_DIM, "action base state")
        if a.ndim != base.ndim + 1:
            raise ValueError(
                "Expected action [...,T,20] and base state [...,34]; "
                f"got {tuple(a.shape)} and {tuple(base.shape)}"
            )

        def relative_eef(action_pose: torch.Tensor, base_pose: torch.Tensor):
            Ra, ta = _eef9_to_Rt(action_pose)
            Rb, tb = _eef9_to_Rt(base_pose)
            # Insert the action time axis for broadcasting.
            Rb = Rb.unsqueeze(-3)
            tb = tb.unsqueeze(-2)
            inverse_base = Rb.transpose(-1, -2)
            delta_position = _matvec(inverse_base, ta - tb)
            delta_rotation = _matmul(inverse_base, Ra)
            return _Rt_to_eef9(delta_rotation, delta_position)

        left = relative_eef(a[..., ACTION_LEFT_EEF], base[..., STATE_LEFT_EEF])
        right = relative_eef(a[..., ACTION_RIGHT_EEF], base[..., STATE_RIGHT_EEF])
        hand = _empty_hand(a)
        result = torch.cat(
            (
                left,
                hand,
                a[..., ACTION_LEFT_GRIP],
                right,
                hand,
                a[..., ACTION_RIGHT_GRIP],
            ),
            dim=-1,
        )
        _check_last_dim(result, UNIFIED_DIM, "unified action")
        return result

    def forward(self, batch):
        """Transformer API consumed by AgenticWAMProcessor during training."""
        for key in self.keys:
            state = batch["state"][key]
            _check_last_dim(state, STATE_DIM, f"state[{key}]")
            if "action" in batch and key in batch["action"]:
                # Important: s[...,0,:], NOT the future / last observation.
                batch["action"][key] = self._action64(
                    batch["action"][key], state[..., 0, :]
                )
            batch["state"][key] = self._state64(state)
        return batch

    def backward(self, batch):
        """Unified delta EEF64 -> unified ABSOLUTE EEF64, matching YAM.

        The Processor calls this after de-normalizing predictions. Like the
        existing YAM transform, backward does not produce raw 20D directly.
        """
        for key in self.keys:
            state = batch["state"][key]  # [..., Tobs, 64]
            action = batch["action"][key]  # [..., Tact, 64]
            _check_last_dim(state, UNIFIED_DIM, f"state[{key}]")
            _check_last_dim(action, UNIFIED_DIM, f"action[{key}]")

            def absolute_eef(delta_pose: torch.Tensor, base_pose: torch.Tensor):
                Rd, td = _eef9_to_Rt(delta_pose)
                Rb, tb = _eef9_to_Rt(base_pose)
                Rb = Rb.unsqueeze(-3)
                tb = tb.unsqueeze(-2)
                return _Rt_to_eef9(
                    _matmul(Rb, Rd),
                    tb + _matvec(Rb, td),
                )

            left = absolute_eef(
                action[..., UNIFIED_LEFT_EEF],
                state[..., 0, UNIFIED_LEFT_EEF],
            )
            right = absolute_eef(
                action[..., UNIFIED_RIGHT_EEF],
                state[..., 0, UNIFIED_RIGHT_EEF],
            )
            batch["action"][key] = torch.cat(
                (
                    left,
                    action[..., 9:31],
                    action[..., UNIFIED_LEFT_GRIP],
                    right,
                    action[..., 41:63],
                    action[..., UNIFIED_RIGHT_GRIP],
                ),
                dim=-1,
            )
        return batch

    def backward_action20(self, batch):
        """Convenience for native EEF controllers: delta64 -> absolute raw20.

        Call only if the consumer actually expects the supplied 20D action
        layout. This helper is separate from backward() for YAM compatibility.
        """
        self.backward(batch)
        for key in self.keys:
            batch["action"][key] = absolute_eef64_to_action20(
                batch["action"][key]
            )
        return batch
