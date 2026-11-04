#!/usr/bin/env python3
"""LeRobot v3.0 packed-Parquet -> gripper events + post-release retreat -> AW-0 code labels.

Reads ONLY raw observation.state (34D) and task_index from packed Parquet.
Never decodes/re-encodes the videos and never edits the original dataset.

IMPORTANT: detected 'close'/'open' are gripper-command/sensor transitions,
not verified physical object contacts. Review audit JSONL and video samples.

Usage:
 python build_v3_gripper_code.py --dataset /data/my_dataset --open-is-high
 python build_v3_gripper_code.py --dataset /data/my_dataset --open-is-low --min-range 0.005
 python build_v3_gripper_code.py --dataset /data/my_dataset --open-is-high --knob-axis x
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np

TASKS = {
    0: ("left", "air fryer door handle", "open the air fryer"),
    1: ("right", "pink cake", "place the cake into the air fryer"),
    2: ("left", "temperature control knob", "turn the temperature knob"),
    3: ("left", "air fryer door handle", "close the air fryer"),
}


def iter_episodes(root: Path, batch_size: int = 4096):
    """Stream episodes in the Parquet's physical row order; fail on broken indices.

    An episode may cross file and record-batch boundaries. This assumes LeRobot's
    packed files store rows in ascending global `index` order, and verifies it.
    """
    import pyarrow.parquet as pq

    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No packed v3 Parquet files under {root / 'data'}")
    required = ["index", "episode_index", "frame_index", "task_index", "observation.state"]
    cur_ep = None
    cur_task = None
    parts = []
    last_global = -1
    local_frame = 0
    for path in files:
        pf = pq.ParquetFile(str(path))
        names = set(pf.schema_arrow.names)
        missing = set(required) - names
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for rb in pf.iter_batches(batch_size=batch_size, columns=required):
            raw = rb.to_pydict()
            states = np.asarray(raw["observation.state"], dtype=np.float32)
            if states.ndim != 2 or states.shape[1] != 34:
                raise ValueError(f"{path}: expected state [N,34], got {states.shape}")
            for j in range(len(states)):
                gidx = int(raw["index"][j])
                ep = int(raw["episode_index"][j])
                fi = int(raw["frame_index"][j])
                ti = int(raw["task_index"][j])
                if gidx != last_global + 1:
                    raise ValueError(f"Global indices not consecutive: previous={last_global}, current={gidx}. "
                                     "Check file sorting and index semantics before building labels.")
                last_global = gidx
                if cur_ep is None:
                    cur_ep, cur_task, local_frame = ep, ti, 0
                if ep != cur_ep:
                    if ep < cur_ep:
                        raise ValueError(f"Episode order went backward: {cur_ep} -> {ep}")
                    yield cur_ep, cur_task, np.concatenate(parts, axis=0)
                    parts = []
                    cur_ep, cur_task, local_frame = ep, ti, 0
                if ti != cur_task:
                    raise ValueError(f"Episode {ep} has task_index {cur_task} and {ti}. "
                                     "This script expects one task per episode; split multi-task episodes first.")
                if fi != local_frame:
                    raise ValueError(f"Episode {ep} has frame_index={fi}, expected={local_frame}")
                parts.append(states[j:j+1])
                local_frame += 1
    if cur_ep is not None:
        yield cur_ep, cur_task, np.concatenate(parts, axis=0)


def median_filter3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x.copy()
    p = np.pad(x, (1, 1), mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]], axis=0), axis=0)


def detect_events(x: np.ndarray, *, open_is_high: bool, min_range: float,
                  stable_frames: int):
    """Hysteresis + stable-state confirmation. Returns [(frame,'close'|'open')].

    Event frame is the LAST frame of a confirmed state run (not the contact frame).
    Detects abrupt binary changes as well as slower analog ramps.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite gripper values")
    if not len(x):
        return [], {"range": 0.0, "low": None, "high": None}
    xs = median_filter3(x)
    lo, hi = map(float, np.quantile(xs, [0.05, 0.95]))
    span = hi - lo
    if span < min_range:
        return [], {"range": round(span, 6), "low": lo, "high": hi,
                    "note": "insufficient gripper movement"}
    low, high = lo + 0.35 * span, lo + 0.65 * span
    state = "open" if (xs[0] > (lo + hi) / 2.0) == open_is_high else "closed"
    candidate = None
    count = 0
    events = []
    for i in range(1, len(xs)):
        if state == "open":
            opposite = xs[i] <= low if open_is_high else xs[i] >= high
            target = "closed"
        else:
            opposite = xs[i] >= high if open_is_high else xs[i] <= low
            target = "open"
        if opposite:
            if candidate != target:
                candidate, count = target, 1
            else:
                count += 1
            if count >= stable_frames:
                state = target
                events.append((i, "open" if state == "open" else "close"))
                candidate, count = None, 0
        else:
            candidate, count = None, 0
    return events, {"range": round(span, 6), "low": round(low, 6),
                    "high": round(high, 6), "initial": "open" if
                    ((xs[0] > (lo + hi) / 2.0) == open_is_high) else "closed"}


def rot6d_to_matrix(raw: np.ndarray) -> np.ndarray:
    """The user's rot6d is [R[:,0], R[:,1]] (two matrix COLUMNS)."""
    a, b = np.asarray(raw[:3], dtype=float), np.asarray(raw[3:6], dtype=float)
    if not np.isfinite(raw).all() or np.linalg.norm(a) < 1e-6:
        raise ValueError("Invalid rot6d first column")
    x = a / np.linalg.norm(a)
    y = b - np.dot(x, b) * x
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("Invalid rot6d second column")
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def matrix_to_rpy_deg(R: np.ndarray) -> list[float]:
    """Roll/pitch/yaw in degrees, R=Rz(yaw)@Ry(pitch)@Rx(roll)."""
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll, yaw = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return [round(math.degrees(v), 2) for v in (roll, pitch, yaw)]


def measured_motion(states: np.ndarray, side: str, start: int, end: int,
                    min_motion_m: float, min_rot_deg: float):
    """Observed EE end pose in raw state, not an invented target pose."""
    if not len(states):
        return None, {"error": "empty states"}
    i, j = int(np.clip(start, 0, len(states)-1)), int(np.clip(end, 0, len(states)-1))
    pos_slice = slice(14, 17) if side == "left" else slice(23, 26)
    rot_slice = slice(17, 23) if side == "left" else slice(26, 32)
    p0, p1 = states[i, pos_slice], states[j, pos_slice]
    details = {"xyz_displacement_m": round(float(np.linalg.norm(p1-p0)), 4)}
    try:
        R0 = rot6d_to_matrix(states[i, rot_slice])
        R1 = rot6d_to_matrix(states[j, rot_slice])
        relative = R0.T @ R1
        rotation_deg = math.degrees(math.acos(np.clip((np.trace(relative)-1)/2, -1.0, 1.0)))
        details["eef_rotation_deg"] = round(rotation_deg, 2)
        if details["xyz_displacement_m"] < min_motion_m and rotation_deg < min_rot_deg:
            details["note"] = "little wrist motion"
            return None, details
        pos = [round(float(v), 4) for v in p1]
        rpy = matrix_to_rpy_deg(R1)
        return f"robot.move_arm(side={side!r}, target_pos={pos}, target_rpy={rpy})", details
    except ValueError as e:
        details["error"] = str(e)
        return None, details



def rotation_vector_from_matrix(R: np.ndarray) -> np.ndarray:
    """Principal SO(3) logarithm (radians), valid away from 180-degree frame jumps.

    Because every input is a *consecutive sampled frame*, an approximately 180
    degree relative jump is not temporally identifiable; require review instead
    of pretending to know how many rotations occurred between frames.
    """
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(c)
    vee = np.array([R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]], dtype=float)
    if theta < 1e-5:
        return 0.5 * vee
    if math.pi - theta < math.radians(1.0):
        raise ValueError('one sampled step rotates almost 180 degrees; direction is ambiguous')
    return 0.5 * theta / math.sin(theta) * vee


def measured_knob_rotation(states: np.ndarray, start: int, end: int, opts):
    """Estimate gripper-local rotation while xyz is stationary, from state rot6D.

    Returns a `robot.rotate` expression or None plus the audit record.  The
    physical knob-axis and signed angle still require calibration: the robot API
    assumes rotation about the gripper's approach axis.

    Uses SUM of consecutive local SO(3) rotations, not R[start].T @ R[end],
    so motions exceeding 180 degrees across *multiple frames* are retained.
    """
    i = max(0, min(int(start), len(states) - 1))
    j = max(0, min(int(end), len(states) - 1))
    meta = {"knob_angle_source": "estimated from consecutive left EEF rot6d",
            "rotation_interval": [i, j],
            "knob_axis_frame": "initial/local gripper (approximate)"}
    if j <= i:
        meta["warning"] = "insufficient sampled frames to estimate wrist rotation"
        return None, meta
    positions = np.asarray(states[i:j + 1, 14:17], dtype=float)
    drift = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    meta["xyz_max_drift_m"] = round(drift, 5)
    if not np.all(np.isfinite(positions)) or drift > opts.knob_max_xyz_drift_m:
        meta["warning"] = "XYZ is not sufficiently stationary for a knob-in-place rotation"
        return None, meta
    try:
        Rs = [rot6d_to_matrix(states[k, 17:23]) for k in range(i, j + 1)]
        steps = np.asarray([rotation_vector_from_matrix(a.T @ b)
                            for a, b in zip(Rs[:-1], Rs[1:])])
    except ValueError as exc:
        meta["warning"] = str(exc)
        return None, meta
    motion_norms = np.linalg.norm(steps, axis=1)
    path_angle_deg = math.degrees(float(motion_norms.sum()))
    meta["rotation_path_deg"] = round(path_angle_deg, 2)
    if path_angle_deg < opts.knob_min_angle_deg:
        meta["warning"] = "rotation below minimum angle threshold"
        return None, meta

    if opts.knob_axis == "auto":
        # Principal axis of local rotations; canonical sign is mathematical,
        # not a guarantee of the robot rotate() physical positive convention.
        _, _, vh = np.linalg.svd(steps, full_matrices=False)
        axis = vh[0]
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis = -axis
        meta["knob_axis_mode"] = "auto (sign not physically calibrated)"
    else:
        axis = np.eye(3)[{"x": 0, "y": 1, "z": 2}[opts.knob_axis]]
        meta["knob_axis_mode"] = f"user specified local gripper {opts.knob_axis}"
    signed = steps @ axis
    coherence = float(np.sum(np.abs(signed)) / max(1e-12, np.sum(motion_norms)))
    angle_deg = math.degrees(float(np.sum(signed)))
    meta["knob_axis_local_xyz"] = [round(float(v), 4) for v in axis]
    meta["axis_coherence"] = round(coherence, 3)
    meta["estimated_wrist_angle_before_sign_deg"] = round(angle_deg, 2)
    # The robot.rotate() positive direction may be reversed relative to the
    # mathematical axis convention. Calibrate once, then set --knob-angle-sign.
    angle_deg *= opts.knob_angle_sign
    meta["knob_angle_sign_calibration"] = opts.knob_angle_sign
    meta["estimated_signed_angle_deg"] = round(angle_deg, 2)
    if coherence < opts.knob_min_axis_coherence:
        meta["warning"] = "rotation is not sufficiently concentrated about one gripper-local axis"
        return None, meta
    if abs(angle_deg) < opts.knob_min_angle_deg:
        meta["warning"] = "net signed knob rotation below minimum (possible rotate-and-return)"
        return None, meta
    if opts.knob_axis == "auto":
        meta["warning"] = ("axis inferred from wrist motion: confirm this is the gripper "
                           "approach axis and calibrate the signed-angle convention")
    return f"robot.rotate('temperature control knob', {angle_deg:.2f}, side='left')", meta


def make_code(lines: list[str], *, comment: str = "") -> str:
    if not lines:
        return ""
    body = ([f"    # {comment}"] if comment else [])
    for n, line in enumerate(lines):
        body.append("    " + ("return " if n == len(lines)-1 else "") + line)
    code = "def policy_chunk(obs, state, robot, skills):\n" + "\n".join(body) + "\n"
    ast.parse(code)
    return code


def code_for_event(task_id: int, kind: str, states: np.ndarray,
                   prev_boundary: int, event_frame: int, opts):
    side, object_name, meaning = TASKS[task_id]
    if kind == "close":
        method = "pick" if task_id == 1 else "grasp"
        return make_code([f"robot.{method}({object_name!r}, side={side!r})"]), {}
    if task_id == 1:
        return make_code(["robot.place('pink cake', 'air fryer', side='right')"]), {}

    lines = []
    meta = {}
    if task_id == 2:
        if opts.knob_angle_deg is not None:
            lines.append(f"robot.rotate('temperature control knob', {opts.knob_angle_deg:.2f}, side='left')")
            meta["knob_angle_source"] = "CLI user-provided physical angle"
        else:
            rotation, meta = measured_knob_rotation(
                states, max(0, prev_boundary - 1), event_frame, opts)
            if rotation:
                lines.append(rotation)
            # A failed rotation check must NOT silently become 'move_arm':
            # it cannot substantiate the knob semantics. Keep audit warnings.
    else:
        motion, meta = measured_motion(states, side, max(0, prev_boundary - 1),
                                      event_frame, opts.min_motion_m, opts.min_rot_deg)
        if motion:
            lines.append(motion)
    lines.append(f"robot.release(side={side!r})")
    return make_code(lines, comment=meaning), meta


def code_for_unreleased_tail(task_id: int, states: np.ndarray, boundary: int, opts):
    side, _, meaning = TASKS[task_id]
    if task_id == 1:
        motion, meta = measured_motion(states, side, max(0, boundary-1),
                                      len(states)-1, opts.min_motion_m, opts.min_rot_deg)
        return make_code([motion] if motion else [], comment="moving cake; no opening detected"), meta
    if task_id == 2:
        if opts.knob_angle_deg is not None:
            return make_code([f"robot.rotate('temperature control knob', {opts.knob_angle_deg:.2f}, side='left')"],
                             comment=meaning), {"knob_angle_source": "CLI user-provided physical angle"}
        rotation, meta = measured_knob_rotation(
            states, max(0, boundary - 1), len(states) - 1, opts)
        return make_code([rotation] if rotation else [], comment=meaning), meta
    motion, meta = measured_motion(states, side, max(0, boundary-1),
                                  len(states)-1, opts.min_motion_m, opts.min_rot_deg)
    return make_code([motion] if motion else [], comment=meaning), meta



def code_for_retreat(task_id: int, states: np.ndarray, open_frame: int, opts):
    """Find a measured post-open return/withdrawal motion and label only its span.

    Return (motion_span_or_none, audit_dict). A retreat MUST follow a measured
    gripper OPEN transition and have nontrivial XYZ travel. The trajectory is
    approximated by one or a few actual observed EEF waypoints. This is a
    SYMBOLIC code label, not a safe motion plan for direct robot execution.
    """
    side, _, _ = TASKS[task_id]
    N = len(states)
    meta = {"retreat_checked_after_open_frame": int(open_frame),
            "retreat_detected": False,
            "note": "post-release retreat only; no synthetic go_home target"}
    if open_frame >= N - 1:
        meta["warning"] = "no frames after open event to detect retreat"
        return None, meta

    p_slice = slice(14, 17) if side == "left" else slice(23, 26)
    r_slice = slice(17, 23) if side == "left" else slice(26, 32)
    # The anchor includes the final OPEN sample (which may precede the return).
    # Median filtering reduces spurious movements; never change the source poses
    # used in the emitted API targets.
    pts = np.asarray(states[open_frame:, p_slice], dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        meta["warning"] = "non-finite XYZ values in post-release segment"
        return None, meta
    if len(pts) >= 3:
        smooth = np.column_stack([median_filter3(pts[:, k]) for k in range(3)])
    else:
        smooth = pts.copy()
    displacement = np.linalg.norm(smooth - smooth[0], axis=1)
    increments = np.linalg.norm(np.diff(smooth, axis=0), axis=1)
    excursion = float(np.max(displacement))
    path = float(np.sum(increments))
    meta["retreat_max_excursion_m"] = round(excursion, 5)
    meta["post_open_path_length_m"] = round(path, 5)
    if excursion < opts.retreat_min_motion_m:
        meta["note"] = "no post-release XYZ excursion above retreat-min-motion-m"
        return None, meta

    # A stable, thresholded set of moving increments defines the beginning and
    # end of the return. If the robot moves slowly with each individual step
    # below the threshold but the accumulated excursion is real, fall back to
    # crossing one quarter of the excursion threshold.
    active_step = np.flatnonzero(increments >= opts.retreat_step_m)
    if len(active_step):
        first_local = int(active_step[0])
        last_local = int(active_step[-1]) + 1
    else:
        crossing = np.flatnonzero(displacement >= opts.retreat_min_motion_m / 4)
        if not len(crossing):
            meta["warning"] = "excursion found but onset cannot be located"
            return None, meta
        first_local = max(0, int(crossing[0]) - 1)
        last_local = int(np.argmax(displacement))
    begin = open_frame + first_local
    finish = open_frame + last_local
    if finish <= begin:
        meta["warning"] = "zero-length return stage after motion detection"
        return None, meta

    # Choose representative measured waypoints in cumulative XYZ arc length,
    # so 2-3 waypoints represent a curved withdrawal rather than a made-up
    # straight line. The final waypoint is the end of actual motion, not the
    # episode's padded stationary tail.
    path_pts = smooth[first_local:last_local + 1]
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path_pts, axis=0), axis=1))]
    n_wp = max(1, int(opts.retreat_max_waypoints))
    sampled = []
    for d in np.linspace(float(cumulative[-1]) / n_wp, float(cumulative[-1]), n_wp):
        local = int(np.searchsorted(cumulative, d, side="left"))
        idx = min(finish, begin + local)
        if idx not in sampled:
            sampled.append(idx)
    if not sampled or sampled[-1] != finish:
        sampled.append(finish)
    if len(sampled) > n_wp:
        sampled = sampled[-n_wp:]

    statements = []
    used_frames = []
    for frame in sampled:
        try:
            R = rot6d_to_matrix(states[frame, r_slice])
        except ValueError as exc:
            meta["warning"] = f"invalid retreat waypoint rotation: {exc}"
            return None, meta
        xyz = [round(float(v), 4) for v in states[frame, p_slice]]
        rpy = matrix_to_rpy_deg(R)
        statements.append(f"robot.move_arm(side={side!r}, target_pos={xyz}, target_rpy={rpy})")
        used_frames.append(int(frame))
    meta.update({"retreat_detected": True,
                 "retreat_start_frame": int(begin),
                 "retreat_end_frame": int(finish),
                 "retreat_waypoint_frames": used_frames,
                 "retreat_waypoints": len(used_frames),
                 "retreat_net_translation_m": round(float(np.linalg.norm(smooth[last_local] - smooth[first_local])), 5)})
    return {"start": int(begin), "end": int(finish + 1), "event": "retreat",
            "event_frame": None,
            "code": make_code(statements, comment="post-release arm retreat to observed pose(s)"),
            "motion": meta}, meta



def code_for_task3_open_push(states: np.ndarray, opts):
    """Conservative Task 3 fallback: recorded gripper remains OPEN while left EEF moves.

    This does NOT claim that a door handle was grasped, that contact occurred,
    or that the door actually closed. The episode instruction supplies intent;
    EEF poses supply measured motion. Visual verification remains mandatory.

    A retreat is emitted only if the EEF travels outward from the initial pose
    and then noticeably reduces its distance to that pose (return evidence).
    This is an intentionally conservative heuristic, not contact recognition.
    """
    T = len(states)
    audit = {"task3_mode": "open_gripper_push_motion_only",
             "task3_contact_verified": False, "task3_retreat_detected": False,
             "task3_visual_verification_required": True}
    if T < 4:
        audit["warning"] = "not enough frames for open-gripper motion segmentation"
        return [{"start": 0, "end": T, "event": "unverified", "event_frame": None,
                 "code": "", "motion": audit}], audit
    pos = np.asarray(states[:, 14:17], dtype=float)
    if not np.all(np.isfinite(pos)):
        audit["warning"] = "non-finite left-EEF positions"
        return [{"start": 0, "end": T, "event": "unverified", "event_frame": None,
                 "code": "", "motion": audit}], audit
    smoothed = np.column_stack([median_filter3(pos[:, k]) for k in range(3)])
    start_pos = np.median(smoothed[:min(T, 3)], axis=0)
    distance = np.linalg.norm(smoothed - start_pos, axis=1)
    peak = int(np.argmax(distance))
    excursion = float(distance[peak])
    end_distance = float(distance[-1])
    post_peak_path = float(np.sum(np.linalg.norm(np.diff(smoothed[peak:], axis=0), axis=1)))
    return_amount = excursion - end_distance
    audit.update({"task3_peak_frame": peak,
                  "task3_peak_excursion_m": round(excursion, 5),
                  "task3_final_distance_from_start_m": round(end_distance, 5),
                  "task3_distance_recovered_m": round(return_amount, 5),
                  "task3_post_peak_path_m": round(post_peak_path, 5)})
    if excursion < opts.min_motion_m or peak < 2:
        audit["warning"] = "insufficient measured EEF travel to ground Task 3"
        return [{"start": 0, "end": T, "event": "unverified", "event_frame": None,
                 "code": "", "motion": audit}], audit

    # A reversal toward the initial pose is evidence of return. No gripper
    # RELEASE is used, because sensor & action are constantly open in this set.
    has_return = (peak < T - 3 and
                  return_amount >= opts.retreat_min_motion_m and
                  post_peak_path >= opts.retreat_min_motion_m)
    if has_return:
        operation_end = peak
        retreat_begin = peak + 1
        # End at the last appreciable movement, rather than the padded idle tail.
        steps = np.linalg.norm(np.diff(smoothed[retreat_begin:], axis=0), axis=1)
        active = np.flatnonzero(steps >= opts.retreat_step_m)
        retreat_end = (retreat_begin + int(active[-1]) + 1
                       if len(active) else T - 1)
        retreat_end = max(retreat_begin, min(retreat_end, T - 1))
    else:
        operation_end = T - 1
        retreat_begin = retreat_end = None
        audit["warning"] = "no verified reversal; Task 3 not split into retreat"

    op, op_meta = measured_motion(states, "left", 0, operation_end,
                                  opts.min_motion_m, opts.min_rot_deg)
    if op is None:
        audit["warning"] = "unable to encode measured Task 3 motion"
        return [{"start": 0, "end": T, "event": "unverified", "event_frame": None,
                 "code": "", "motion": audit | op_meta}], audit

    operation_code = make_code([op], comment=(
        "Task 3: instruction says close air fryer with OPEN left gripper; "
        "measured motion only, contact unverified"))
    spans = [{"start": 0, "end": operation_end + 1, "event": "task3_open_push",
              "event_frame": None, "code": operation_code, "motion": op_meta}]
    if has_return:
        retreat, retreat_meta = measured_motion(states, "left", operation_end,
                                               retreat_end, opts.min_motion_m,
                                               opts.min_rot_deg)
        if retreat is not None:
            audit["task3_retreat_detected"] = True
            spans.append({"start": retreat_begin, "end": retreat_end + 1,
                          "event": "task3_retreat", "event_frame": None,
                          "code": make_code([retreat], comment=(
                              "Task 3: observed open-gripper arm withdrawal; "
                              "no release event")), "motion": retreat_meta})
            if retreat_end + 1 < T:
                spans.append({"start": retreat_end + 1, "end": T,
                              "event": "post_retreat_idle", "event_frame": None,
                              "code": "", "motion": {}})
        else:
            # Merge back when retreat's actual target pose is not reliable.
            spans[0]["end"] = T
            audit["warning"] = "return-like motion exists but retreat pose invalid"
    return spans, audit


def label_episode(ep: int, ti: int, states: np.ndarray, opts):
    if ti not in TASKS:
        raise ValueError(f"Unexpected task_index={ti} in episode {ep}")
    active, _, _ = TASKS[ti]
    ev, info = detect_events(states[:, 32 if active == "left" else 33],
                             open_is_high=opts.open_is_high, min_range=opts.min_range,
                             stable_frames=opts.stable_frames)
    other, _ = detect_events(states[:, 33 if active == "left" else 32],
                             open_is_high=opts.open_is_high, min_range=opts.min_range,
                             stable_frames=opts.stable_frames)
    # Optional display metric only: state gripper values are normalized 0..1
    # (1 open, 0 closed), with a nominal full opening of 70 mm.
    raw_gripper = states[:, 32 if active == "left" else 33]
    info["nominal_full_open_mm"] = opts.gripper_full_stroke_mm
    info["observed_opening_mm_p05"] = round(float(np.quantile(raw_gripper, 0.05) * opts.gripper_full_stroke_mm), 2)
    info["observed_opening_mm_p95"] = round(float(np.quantile(raw_gripper, 0.95) * opts.gripper_full_stroke_mm), 2)
    if raw_gripper.min() < -0.05 or raw_gripper.max() > 1.05:
        info["aperture_warning"] = "gripper values outside expected normalized [0, 1] range"
    # Task 3 special-case: verified dataset inspection shows both grippers
    # and commanded gripper targets stuck OPEN (1.0), so there are no real
    # close/open events. Only opt in when the source sensor is indeed open.
    if ti == 3 and getattr(opts, "task3_open_push", False) and not ev:
        values = np.asarray(states[:, 32], dtype=float)
        held_open = ((np.min(values) >= 0.95) if opts.open_is_high else
                     (np.max(values) <= 0.05))
        if held_open and not other:
            spans, motion_audit = code_for_task3_open_push(states, opts)
            assert sum(s["end"] - s["start"] for s in spans) == len(states)
            warnings = ["Task 3 inferred from task instruction and EEF kinematics; contact not verified"]
            if motion_audit.get("warning"):
                warnings.append(motion_audit["warning"])
            return spans, {"episode_index": ep, "task_index": ti,
                           "active_arm": active, "events": ev,
                           "other_arm_events": other, "n_frames": len(states),
                           "gripper_detection": info,
                           "task3_open_push_audit": motion_audit,
                           "retreat_detection": {}, "spans": spans,
                           "warnings": warnings, "warning": "; ".join(warnings)}
    spans = []
    start = 0
    for f, kind in ev:
        stop = min(f+1, len(states))  # the event belongs to the closing span
        if stop <= start:
            continue
        code, meta = code_for_event(ti, kind, states, start, f, opts)
        spans.append({"start": start, "end": stop, "event": kind,
                      "event_frame": int(f), "code": code, "motion": meta})
        start = stop
    retreat_audit = {}
    if start < len(states):
        if ev and ev[-1][1] == "open":
            # Distinct stage AFTER release/place, not bundled into its label.
            retreat, retreat_audit = code_for_retreat(ti, states, ev[-1][0], opts)
            if retreat is not None:
                # Event detection is delayed by stable-state confirmation: a
                # measured withdrawal may begin within the OPEN event's own
                # final samples. Never produce overlapping frame spans.
                if retreat["start"] < start:
                    retreat_audit["retreat_onset_overlaps_open_confirmation"] = True
                    retreat["start"] = start
                if retreat["end"] <= retreat["start"]:
                    retreat = None
                    retreat_audit["retreat_detected"] = False
                    retreat_audit["warning"] = "withdrawal entirely overlapped gripper open confirmation"
            if retreat is not None:
                if retreat["start"] > start:
                    spans.append({"start": start, "end": retreat["start"],
                                  "event": "post_open_idle", "event_frame": None,
                                  "code": "", "motion": {}})
                spans.append(retreat)
                if retreat["end"] < len(states):
                    spans.append({"start": retreat["end"], "end": len(states),
                                  "event": "post_retreat_idle", "event_frame": None,
                                  "code": "", "motion": {}})
            else:
                spans.append({"start": start, "end": len(states), "event": "tail",
                              "event_frame": None, "code": "", "motion": retreat_audit})
        else:
            code, meta = (code_for_unreleased_tail(ti, states, start, opts)
                          if ev and ev[-1][1] == "close" else ("", {}))
            spans.append({"start": start, "end": len(states), "event": "tail",
                          "event_frame": None, "code": code, "motion": meta})
    # Zero events -> unlabelled, avoid pretending a grasp happened.
    assert sum(s["end"]-s["start"] for s in spans) == len(states)
    warnings = []
    if not ev:
        warnings.append("no active-arm gripper event")
    if other:
        warnings.append("other arm also changed")
    if ti == 2 and any(s["motion"].get("warning") for s in spans if s["event"] == "open"):
        warnings.append("knob rotation needs review")
    if ev and ev[-1][1] == "open" and not retreat_audit.get("retreat_detected", False):
        warnings.append("no measured post-release retreat above threshold (or no frames after open)")
    if len([1 for _, k in ev if k == "open"]) > 1:
        warnings.append("multiple grasp/release cycles: only final post-open retreat segmented separately")
    return spans, {"episode_index": ep, "task_index": ti, "active_arm": active,
                   "n_frames": len(states), "events": [[int(f), k] for f,k in ev],
                   "other_arm_events": [[int(f), k] for f,k in other],
                   "gripper_detection": info, "retreat_detection": retreat_audit,
                   "spans": spans, "warnings": warnings, "warning": "; ".join(warnings)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="LeRobot v3 root (contains meta/data/videos)")
    ap.add_argument("--out", type=Path, default=None, help="default: <dataset>/policy_code_idx_custom.pt")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--open-is-high", action="store_true", help="gripper larger value = more open")
    group.add_argument("--open-is-low", action="store_true", help="gripper smaller value = more open")
    ap.add_argument("--min-range", type=float, default=0.01,
                    help="minimum 5th-to-95th percentile gripper travel; adapt to sensor units")
    ap.add_argument("--stable-frames", type=int, default=2,
                    help="stable thresholded samples to confirm a gripper event")
    ap.add_argument("--task3-open-push", action="store_true",
                    help="Task 3 fallback ONLY if recorded left gripper stays open; "
                         "label measured left-EEF close/push motion and conservative reversal/retreat, "
                         "never invent grasp/release")
    ap.add_argument("--min-motion-m", type=float, default=0.015)
    ap.add_argument("--retreat-min-motion-m", type=float, default=0.015,
                    help="post-open max EEF XYZ excursion required to label a retreat; default 15 mm")
    ap.add_argument("--retreat-step-m", type=float, default=0.0015,
                    help="post-open per-sampled-step XYZ motion to find retreat onset/end; default 1.5 mm")
    ap.add_argument("--retreat-max-waypoints", type=int, choices=[1, 2, 3], default=1,
                    help="number of observed EEF waypoints in the retreat code (1-3); default 1")
    ap.add_argument("--min-rot-deg", type=float, default=8.0)
    ap.add_argument("--knob-angle-deg", type=float, default=None,
                    help="optional known physical knob angle; overrides automatic angle estimation")
    ap.add_argument("--knob-axis", choices=["auto", "x", "y", "z"], default="auto",
                    help="gripper-local rotation axis; use x/y/z after calibrating the approach axis; auto = PCA-based estimate")
    ap.add_argument("--knob-angle-sign", type=int, choices=[-1, 1], default=1,
                    help="physical robot.rotate sign calibration (1 or -1); positive angle convention must be verified")
    ap.add_argument("--gripper-full-stroke-mm", type=float, default=70.0,
                    help="nominal full gripper opening for audit only; assumes state 0..1 maps linearly to 0..70 mm")
    ap.add_argument("--knob-max-xyz-drift-m", type=float, default=0.02,
                    help="maximum allowed left EEF XYZ excursion during a knob rotation (metres)")
    ap.add_argument("--knob-min-angle-deg", type=float, default=8.0,
                    help="minimum signed rotation magnitude to emit robot.rotate")
    ap.add_argument("--knob-min-axis-coherence", type=float, default=0.8,
                    help="minimum consistency of incremental motion about the estimated/specified axis")
    ap.add_argument("--audit", type=Path, default=None,
                    help="default: <dataset>/gripper_code_audit.jsonl")
    args = ap.parse_args()
    if args.stable_frames < 1 or args.min_range < 0:
        ap.error("stable-frames must be >=1 and min-range >=0")
    if args.retreat_min_motion_m <= 0 or args.retreat_step_m <= 0:
        ap.error("retreat motion/step thresholds must be positive")
    if args.gripper_full_stroke_mm <= 0:
        ap.error("gripper-full-stroke-mm must be positive")
    if args.knob_max_xyz_drift_m < 0 or args.knob_min_angle_deg < 0:
        ap.error("knob drift/angle thresholds must be nonnegative")
    if not 0 <= args.knob_min_axis_coherence <= 1:
        ap.error("knob-min-axis-coherence must be in [0, 1]")
    root = args.dataset.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected v3.0, got {info.get('codebase_version')!r}")
    out = args.out or root / "policy_code_idx_custom.pt"
    audit = args.audit or root / "gripper_code_audit.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    import torch
    codes, code_lookup, ids, order = [], {}, [], []
    n_episodes, warning_count, n_events, nonempty_frames = 0, 0, 0, 0
    with audit.open("w") as fout:
        for ep, ti, states in iter_episodes(root):
            spans, rec = label_episode(ep, ti, states, args)
            base = len(ids)
            order.append({"trajectory_id": f"ep{ep:06d}", "episode_index": ep,
                          "offset": base, "length": len(states), "task_index": ti})
            for span in spans:
                code = span["code"]
                if code not in code_lookup:
                    code_lookup[code] = len(codes)
                    codes.append(code)
                k = code_lookup[code]
                n = span["end"]-span["start"]
                ids.extend([k] * n)
                if code:
                    nonempty_frames += n
            n_events += len(rec["events"])
            warning_count += bool(rec["warning"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_episodes += 1
            if n_episodes % 500 == 0:
                print(f"  processed episodes={n_episodes}, frames={len(ids)}, events={n_events}", flush=True)
    expected_eps, expected_frames = info.get("total_episodes"), info.get("total_frames")
    if expected_eps is not None and expected_eps != n_episodes:
        raise AssertionError(f"Expected {expected_eps} episodes; found {n_episodes}")
    if expected_frames is not None and expected_frames != len(ids):
        raise AssertionError(f"Expected {expected_frames} frames; found {len(ids)}")
    torch.save({"codes": codes, "code_id_by_idx": torch.tensor(ids, dtype=torch.int32)}, out)
    order_file = out.with_name("episode_order.json")
    order_file.write_text(json.dumps(order, indent=2, ensure_ascii=False))
    print(f"DONE episodes={n_episodes}, frames={len(ids)}, detected_events={n_events}")
    print(f"unique_codes={len(codes)}, labelled_frames={nonempty_frames} "
          f"({nonempty_frames/max(len(ids),1):.1%}), warnings={warning_count}")
    print(f"cache={out}\norder={order_file}\naudit={audit}")
    print("NEXT: manually review audit, open/close polarity, video alignment and the actual v3 loader frame order.")


if __name__ == "__main__":
    main()
