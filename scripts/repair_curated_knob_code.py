#!/usr/bin/env python3
"""Restore curated knob labels from retained, previously audited pose windows.

No dataset, threshold, optimizer or existing cache is modified. Original/curated
pose equality and angle reproduction are checked before writing a new cache.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_custom_trajectories import rotation6d_matrices, rotation_vectors
from rebuild_semantic_code_cache import calls, validate_inputs, without_comments

VERSION = "curated-knob-reference-repair-v1"


def read_audit(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if type(value) is not int or value in result:
            raise ValueError(f"Invalid or duplicate {key}: {value}")
        result[value] = row
    return result


def robot_calls(row):
    return [call for span in row["spans"] for call in calls(span["code"])]


def calibration(motion):
    mode, sign = motion.get("knob_axis_mode"), motion.get("knob_angle_sign_calibration")
    if not isinstance(mode, str) or type(sign) is not int or sign not in (-1, 1):
        raise ValueError("Reference lacks explicit axis/sign metadata")
    return mode, sign


def reference_rotation(row, mapping, current_calibration):
    if row["task_index"] != 2 or row["n_frames"] < mapping["source_end_frame"]:
        raise ValueError("Reference task/length does not match source mapping")
    found = []
    for span in row["spans"]:
        for node in ast.walk(ast.parse(span["code"])):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "robot"
                    and node.func.attr == "rotate"):
                found.append((node, span.get("motion", {})))
    if len(found) != 1:
        raise ValueError("Require exactly one audited source rotation")
    node, motion = found[0]
    if not motion.get("rotation_window_refined"):
        raise ValueError("Reference has no refined rotation window")
    window = motion.get("rotation_window")
    if (not isinstance(window, list) or len(window) != 2 or any(type(i) is not int for i in window)
            or not mapping["source_start_frame"] <= window[0] < window[1] < mapping["source_end_frame"]):
        raise ValueError("Audited rotation window is not fully retained")
    parsed = calls("def policy_chunk(obs, state, robot, skills):\n    return " + ast.unparse(node) + "\n")[0]
    if (parsed[0] != "rotate" or len(parsed[1]) != 2 or parsed[1][0] != "temperature control knob"
            or type(parsed[1][1]) not in (int, float) or not np.isfinite(parsed[1][1])
            or parsed[2] != {"side": "left"}):
        raise ValueError("Unexpected reference rotation API")
    if calibration(motion) != current_calibration:
        raise ValueError("Reference and current cache use different axis/sign settings")
    angle = motion.get("estimated_signed_angle_deg")
    if type(angle) not in (int, float) or not np.isfinite(angle) or abs(angle - parsed[1][1]) > .011:
        raise ValueError("Reference angle metadata/code mismatch")
    return dict(call=ast.unparse(node), angle_deg=float(parsed[1][1]),
                original_window=window, motion=deepcopy(motion))


def plan_repairs(records, mappings, references):
    current = index_unique(records, "episode_index")
    mapped = index_unique(mappings, "new_episode")
    original = index_unique(references, "episode_index")
    if set(current) != set(mapped) or sorted(current) != list(range(len(current))):
        raise ValueError("Mapping must cover contiguous current episode IDs")
    end = 0
    for episode in sorted(mapped):
        m, row = mapped[episode], current[episode]
        if (m["new_global_offset"] != end or m["frames"] != row["n_frames"] or
                m["frames"] != m["source_end_frame"] - m["source_start_frame"] or
                not 0 <= m["source_start_frame"] < m["source_end_frame"]):
            raise ValueError("Invalid mapping offsets/lengths")
        if m["source_episode"] not in original or original[m["source_episode"]]["task_index"] != row["task_index"]:
            raise ValueError("Source reference missing or task mismatch")
        end += m["frames"]
    signatures = {calibration(span["motion"]) for row in records if row["task_index"] == 2
                  for span in row["spans"] if "knob_axis_mode" in span.get("motion", {})}
    if len(signatures) != 1:
        raise ValueError("Current cache needs one explicit axis/sign recipe")
    signature = next(iter(signatures))
    plans = []
    for row in records:
        if row["task_index"] != 2:
            continue
        names = [call[0] for call in robot_calls(row)]
        missing_rotate, missing_release = "rotate" not in names, "release" not in names
        if not missing_rotate and not missing_release:
            continue
        m = mapped[row["episode_index"]]
        source = original[m["source_episode"]]
        reference = reference_rotation(source, m, signature)
        if missing_rotate and missing_release:
            raise ValueError("Ambiguous fragment lacks both rotation and release; review manually")
        if missing_rotate:
            candidates = [i for i, span in enumerate(row["spans"])
                          if calls(span["code"]) == [("release", [], {"side": "left"})]]
            if len(candidates) != 1:
                raise ValueError("Missing rotation requires one release-only span")
            kind, index = "restore_rotate", candidates[0]
        else:
            candidates = [i for i, span in enumerate(row["spans"])
                          if calls(span["code"]) == [("rotate", ["temperature control knob", reference["angle_deg"]], {"side": "left"})]
                          and span["end"] == row["n_frames"]]
            source_opens = [int(event[0]) for event in source["events"] if event[1] == "open"]
            if (len(candidates) != 1 or m["source_end_frame"] not in source_opens
                    or any(event[1] == "open" for event in row["events"])):
                raise ValueError("Boundary release lacks matching source confirmation")
            kind, index = "restore_boundary_release", candidates[0]
        plans.append(dict(new_episode=row["episode_index"], source_episode=m["source_episode"],
            kind=kind, span_index=index, mapping=deepcopy(m), reference=reference))
    return plans


def verify_evidence(plan, current_states, source_states):
    m, reference = plan["mapping"], plan["reference"]
    if current_states.shape != (m["frames"], 34) or source_states.ndim != 2 or source_states.shape[1] != 34:
        raise ValueError("Invalid custom34 state shape")
    start, stop = reference["original_window"]
    local = current_states[start - m["source_start_frame"]:stop - m["source_start_frame"] + 1, 14:23]
    source = source_states[start:stop + 1, 14:23]
    if not np.array_equal(local, source) or not np.isfinite(local).all():
        raise ValueError("Retained rotation poses differ from source poses")
    rotations, valid = rotation6d_matrices(local[:, 3:9])
    if not valid.all():
        raise ValueError("Invalid retained rotation6d")
    steps = rotation_vectors(rotations[:-1].transpose(0, 2, 1) @ rotations[1:])
    axis = np.asarray(reference["motion"].get("knob_axis_local_xyz"), dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis) - 1) > .001:
        raise ValueError("Invalid recorded source axis")
    angle = float(np.rad2deg((steps @ axis).sum()) * calibration(reference["motion"])[1])
    if abs(angle - reference["angle_deg"]) > .05:
        raise ValueError("Recorded angle cannot be reproduced from retained poses (0.05 degree tolerance)")
    evidence = dict(retained_pose_equality=True, reproduced_angle_deg=angle)
    if plan["kind"] == "restore_boundary_release":
        end = m["source_end_frame"]
        grip = source_states[end - 2:end + 1, 32]
        if (end >= len(source_states) or len(grip) != 3 or not np.isfinite(grip).all()
                or not (grip[0] <= .35 and np.all(grip[1:] >= .65))
                or not np.array_equal(current_states[-2:, 32], grip[:2])):
            raise ValueError("Need observed closed/open/open source context for boundary release")
        evidence.update(source_gripper_context=grip.tolist(), source_confirmed_open_frame=end,
                        retained_open_frame=m["frames"] - 1)
    return evidence


def policy_with_calls(expressions):
    return "def policy_chunk(obs, state, robot, skills):\n" + "".join(
        "    " + ("return " if i == len(expressions) - 1 else "") + expression + "\n"
        for i, expression in enumerate(expressions))


def apply_repairs(records, plans, evidence):
    result = deepcopy(records)
    index = index_unique(result, "episode_index")
    for plan in plans:
        episode = plan["new_episode"]
        proof = evidence.get(episode)
        if not proof or proof.get("retained_pose_equality") is not True:
            raise ValueError("Pose verification is required for every repair")
        if plan["kind"] == "restore_boundary_release" and "source_gripper_context" not in proof:
            raise ValueError("Source gripper confirmation is required")
        row = index[episode]
        span = row["spans"][plan["span_index"]]
        old_code = span["code"]
        span["code"] = policy_with_calls([plan["reference"]["call"], "robot.release(side='left')"])
        motion = span.setdefault("motion", {})
        motion["rotation_attempt_before_repair"] = {key: deepcopy(value) for key, value in motion.items()
            if key not in {"semantic_retreat_merge"}}
        reference = plan["reference"]
        for key in ("knob_axis_mode", "knob_axis_local_xyz", "knob_angle_sign_calibration", "axis_coherence",
                    "estimated_signed_angle_deg", "estimated_wrist_angle_before_sign_deg", "rotation_path_deg",
                    "rotation_window_xyz_drift_m", "rotation_window_active_steps", "warning"):
            if key in reference["motion"]:
                motion[key] = deepcopy(reference["motion"][key])
        motion["audited_rotation_reuse"] = dict(source_episode=plan["source_episode"],
            original_window=reference["original_window"], source_metadata=reference["motion"], evidence=proof)
        motion["rotation_window"] = [frame - plan["mapping"]["source_start_frame"] for frame in reference["original_window"]]
        motion["rotation_window_refined"] = True
        row.setdefault("code_label_repairs", []).append(dict(kind=plan["kind"], previous_code=old_code,
            source_episode=plan["source_episode"], window=reference["original_window"]))
        if plan["kind"] == "restore_boundary_release":
            frame = row["n_frames"] - 1
            row["events"].append([frame, "open"])
            span["source_event"] = span.get("event")
            span["event"], span["event_frame"] = "open_confirmed_from_source_context", frame
            row.setdefault("gripper_detection", {})["source_boundary_confirmation"] = proof
        # Preserve physical-axis/sign review warnings when restoring a label.
        warning = reference["motion"].get("warning")
        if warning and warning not in row.setdefault("warnings", []):
            row["warnings"].append(warning)
            row["warning"] = "; ".join(row["warnings"])
    for row in result:
        for span in row["spans"]:
            span["code"] = without_comments(span["code"])
    return result


def cache_from_records(records):
    codes, lookup, ids = [""], {"": 0}, []
    for row in records:
        for span in row["spans"]:
            code = span["code"]
            if code not in lookup:
                lookup[code] = len(codes)
                codes.append(code)
            ids.extend([lookup[code]] * (span["end"] - span["start"]))
    return dict(codes=codes, code_id_by_idx=torch.tensor(ids, dtype=torch.int32))


def read_states(reader, episodes):
    import pandas as pd
    grouped = defaultdict(list)
    for episode in sorted(set(episodes)):
        grouped[reader.data_file(episode)].append(episode)
    result = {}
    for path, selected in grouped.items():
        table = pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state"],
                                filters=[("episode_index", "in", selected)])
        for episode, frame in table.groupby("episode_index", sort=False):
            frame = frame.sort_values("frame_index")
            if len(frame) != reader.episode_length(episode) or not np.array_equal(frame.frame_index, np.arange(len(frame))):
                raise ValueError("Invalid pose frame order/length")
            result[int(episode)] = np.stack(frame["observation.state"]).astype(np.float64)
    if set(result) != set(episodes):
        raise ValueError("Missing pose evidence")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Curated dataset root")
    parser.add_argument("--cache-dir", type=Path, help="Current cache; default root/coding_v4")
    parser.add_argument("--reference-audit", type=Path, required=True, help="Original audit containing refined rotation windows")
    parser.add_argument("--source-root", type=Path, help="Default: root metadata's curation source_root")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.expanduser().resolve(), args.out_dir.expanduser().resolve()
    if out.exists():
        parser.error("Output must be a new directory; existing caches are never overwritten")
    cache_dir = args.cache_dir or root / "coding_v4"
    info_path, map_path = root / "meta/info.json", root / "curation_source_map.json"
    audit_path, order_path, cache_path = [cache_dir / name for name in
        ("gripper_code_audit.jsonl", "episode_order.json", "policy_code_idx_custom.pt")]
    info = json.loads(info_path.read_text(encoding="utf-8"))
    records, mapping, reference = read_audit(audit_path), json.loads(map_path.read_text(encoding="utf-8")), read_audit(args.reference_audit)
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    order = json.loads(order_path.read_text(encoding="utf-8"))
    validate_inputs(data, order, records, info)
    plans = plan_repairs(records, mapping, reference)
    print(f"[knob-repair] verified cache; proposed repairs={len(plans)}", file=sys.stderr, flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
    target_reader = V3EpisodeReader(root)
    source_root = args.source_root or info.get("trajectory_curation", {}).get("source_root")
    if not source_root:
        parser.error("Source dataset root is required for pose/gripper verification")
    source_reader = V3EpisodeReader(source_root)
    current_states = read_states(target_reader, [p["new_episode"] for p in plans])
    source_states = read_states(source_reader, [p["source_episode"] for p in plans])
    evidence = {}
    refs = index_unique(reference, "episode_index")
    for number, plan in enumerate(plans, 1):
        if source_reader.episode_length(plan["source_episode"]) != refs[plan["source_episode"]]["n_frames"]:
            raise ValueError("Reference source episode length changed")
        evidence[plan["new_episode"]] = verify_evidence(plan, current_states[plan["new_episode"]], source_states[plan["source_episode"]])
        if number % 50 == 0 or number == len(plans):
            print(f"[knob-repair] pose/angle/context verified {number}/{len(plans)}", file=sys.stderr, flush=True)
    result = apply_repairs(records, plans, evidence)
    rebuilt = cache_from_records(result)
    validate_inputs(rebuilt, order, result, info)
    before = Counter(call[0] for row in records for call in robot_calls(row))
    after = Counter(call[0] for row in result for call in robot_calls(row))
    changes = Counter(plan["kind"] for plan in plans)
    expected = before + Counter(rotate=changes["restore_rotate"], release=changes["restore_boundary_release"])
    if after != expected:
        raise ValueError("Unexpected API-call changes")
    inputs = [info_path, map_path, cache_path, order_path, audit_path, args.reference_audit]
    report = dict(version=VERSION, episodes=len(result), frames=len(rebuilt["code_id_by_idx"]), generated_code_comments=False,
        repairs_by_kind=dict(changes), calls_before=dict(before), calls_after=dict(after),
        repairs=[dict(new_episode=p["new_episode"], source_episode=p["source_episode"], kind=p["kind"],
            source_window=p["reference"]["original_window"], angle_deg=p["reference"]["angle_deg"], evidence=evidence[p["new_episode"]]) for p in plans],
        input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        verified=["Cache/audit/order consistency", "Audited rotation windows entirely retained", "Source/curated pose equality",
                  "Angles reproduced within 0.05 degree (recorded axis/angle rounding)", "Boundary release confirmed by source gripper context"],
        not_verified=["Physical knob engagement or signed-angle calibration", "Training improvement", "Tokenizer limits"],
        dataset_and_normalization_unchanged=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    try:
        torch.save(rebuilt, stage / "policy_code_idx_custom.pt")
        (stage / "episode_order.json").write_text(json.dumps(order, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "gripper_code_audit.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result), encoding="utf-8")
        (stage / "knob_repair_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps({key: value for key, value in report.items() if key not in {"repairs", "input_sha256"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
