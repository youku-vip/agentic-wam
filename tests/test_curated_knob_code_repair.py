"""Label repairs require retained pose evidence and confirmed source context."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tokenize

import numpy as np
import pandas as pd
import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/repair_curated_knob_code.py"
spec = importlib.util.spec_from_file_location("knob_repair", SCRIPT)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

AXIS_MODE = "auto (sign not physically calibrated)"
GRASP = "robot.grasp('temperature control knob', side='left')"
ROTATE = "robot.rotate('temperature control knob', 30.0, side='left')"
RELEASE = "robot.release(side='left')"


def motion():
    return dict(rotation_window_refined=True, rotation_window=[2, 4],
        knob_axis_mode=AXIS_MODE, knob_axis_local_xyz=[0, 0, 1],
        knob_angle_sign_calibration=1, estimated_signed_angle_deg=30.0,
        rotation_window_xyz_drift_m=0.001,
        warning="Confirm physical engagement and signed-angle calibration")


def span(start, stop, expressions, event="open", metadata=None):
    return dict(start=start, end=stop, code=repair.policy_with_calls(expressions),
                event=event, event_frame=stop - 1, motion=metadata or {})


def fixtures():
    reference = dict(episode_index=7, n_frames=8, task_index=2, events=[[2, "close"], [6, "open"]],
        spans=[span(0, 3, [GRASP], "close"), span(3, 7, [ROTATE, RELEASE], metadata=motion()),
               span(7, 8, [], "tail")])
    reference["spans"][-1]["code"] = ""
    boundary = deepcopy(reference)
    boundary.update(episode_index=11, n_frames=7, events=[[1, "close"], [6, "open"]])
    boundary["spans"] = [span(0, 2, [GRASP], "close"), span(2, 7, [ROTATE, RELEASE], metadata=motion())]
    boundary["spans"][1]["motion"]["rotation_window"] = [1, 4]
    current = dict(episode_index=0, n_frames=6, task_index=2, events=[[1, "close"], [5, "open"]],
        gripper_detection={}, active_arm="left", warnings=["Whole interval drift"], warning="Whole interval drift",
        spans=[span(0, 2, [GRASP], "close"), span(2, 6, [RELEASE], metadata={"warning": "Whole interval drift"})])
    truncated = deepcopy(current)
    truncated.update(episode_index=1, events=[[1, "close"]])
    truncated["spans"][1] = span(2, 6, [ROTATE], "tail", motion())
    mappings = [dict(new_episode=0, source_episode=7, source_start_frame=1, source_end_frame=7,
                     frames=6, new_global_offset=0),
                dict(new_episode=1, source_episode=11, source_start_frame=0, source_end_frame=6,
                     frames=6, new_global_offset=6)]
    source_states = {}
    for episode, yaw, grip in ((7, [0, 0, 0, 15, 30, 30, 30, 30], [1, 0, 0, 0, 0, 1, 1, 1]),
                               (11, [0, 0, 10, 20, 30, 30, 30], [1, 0, 0, 0, 0, 1, 1])):
        theta = np.deg2rad(yaw)
        state = np.zeros((len(yaw), 34), dtype=np.float32)
        state[:, 17:23] = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(len(yaw)),
                                           -np.sin(theta), np.cos(theta), np.zeros(len(yaw))))
        state[:, 26:32] = [1, 0, 0, 0, 1, 0]
        state[:, 32], state[:, 33] = grip, 1
        source_states[episode] = state
    current_states = {m["new_episode"]: source_states[m["source_episode"]][m["source_start_frame"]:m["source_end_frame"]].copy()
                      for m in mappings}
    return [current, truncated], mappings, [reference, boundary], current_states, source_states


def test_restore_only_missing_apis_and_keep_ranges_and_original_input():
    records, mapping, references, current, source = fixtures()
    snapshot = deepcopy(records)
    plans = repair.plan_repairs(records, mapping, references)
    assert [p["kind"] for p in plans] == ["restore_rotate", "restore_boundary_release"]
    proof = {p["new_episode"]: repair.verify_evidence(p, current[p["new_episode"]], source[p["source_episode"]]) for p in plans}
    fixed = repair.apply_repairs(records, plans, proof)
    assert records == snapshot
    for before, after in zip(records, fixed):
        assert [(s["start"], s["end"]) for s in before["spans"]] == [(s["start"], s["end"]) for s in after["spans"]]
        assert after["spans"][0] == before["spans"][0]
        assert [c[0] for c in repair.robot_calls(after)] == ["grasp", "rotate", "release"]
        assert "signed-angle calibration" in after["warning"]
    assert fixed[1]["events"] == [[1, "close"], [5, "open"]]
    assert proof[1]["source_gripper_context"] == [0, 1, 1]
    assert proof[1]["reproduced_angle_deg"] == pytest.approx(30, abs=1e-5)
    assert fixed[0]["spans"][1]["motion"]["rotation_window"] == [1, 3]
    assert repair.plan_repairs(fixed, mapping, references) == []
    for row in fixed:
        for stage in row["spans"]:
            assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(stage["code"]).readline))


@pytest.mark.parametrize("change", ["lost_window", "different_sign", "different_axis", "no_refined_window", "angle_mismatch"])
def test_reject_incompatible_reference(change):
    records, mapping, refs, _, _ = fixtures()
    metadata = refs[0]["spans"][1]["motion"]
    if change == "lost_window":
        metadata["rotation_window"] = [0, 4]
    elif change == "different_sign":
        metadata["knob_angle_sign_calibration"] = -1
    elif change == "different_axis":
        metadata["knob_axis_mode"] = "user specified local gripper x"
    elif change == "no_refined_window":
        metadata["rotation_window_refined"] = False
    else:
        metadata["estimated_signed_angle_deg"] = 90
    with pytest.raises(ValueError):
        repair.plan_repairs(records, mapping, refs)


def test_reject_pose_edits_and_nonreproducible_angles():
    records, mapping, refs, current, source = fixtures()
    plan = repair.plan_repairs(records, mapping, refs)[0]
    changed = current[0].copy()
    changed[2, 14] += .001
    with pytest.raises(ValueError, match="differ from source"):
        repair.verify_evidence(plan, changed, source[7])
    plan["reference"]["angle_deg"] += 1
    with pytest.raises(ValueError, match="cannot be reproduced"):
        repair.verify_evidence(plan, current[0], source[7])


@pytest.mark.parametrize("failure", ["unknown_future", "not_open", "no_preceding_close", "different_retained_grip"])
def test_boundary_release_needs_source_confirmation(failure):
    records, mapping, refs, current, source = fixtures()
    plan = repair.plan_repairs(records, mapping, refs)[1]
    state = source[11].copy()
    if failure == "unknown_future":
        state = state[:6]
    elif failure == "not_open":
        state[6, 32] = 0
    elif failure == "no_preceding_close":
        state[4, 32] = 1
    else:
        current[1][-1, 32] = 0
    with pytest.raises(ValueError, match="closed/open/open"):
        repair.verify_evidence(plan, current[1], state)


def write_dataset(root, states, source_root=None):
    for directory in ("data/chunk-000", "meta/episodes/chunk-000"):
        (root / directory).mkdir(parents=True)
    info = dict(codebase_version="v3.0", fps=10, total_episodes=len(states), total_frames=sum(len(v) for v in states.values()),
        features={"observation.state": dict(dtype="float32", shape=[34])},
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    if source_root:
        info["trajectory_curation"] = dict(source_root=str(source_root))
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps(dict(task_index=2, task="turn knob")) + "\n")
    (root / "dataset_stats_eef64.json").write_text('{"unchanged": true}\n')
    frames, metadata, offset = [], [], 0
    for episode, state in states.items():
        n = len(state)
        frames.append(pd.DataFrame(dict(episode_index=np.full(n, episode), frame_index=np.arange(n),
                                        **{"observation.state": list(state)})))
        metadata.append(dict(episode_index=episode, **{"data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": offset, "dataset_to_index": offset + n}))
        offset += n
    pd.concat(frames).iloc[::-1].to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(metadata).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)


def snapshot_files(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def test_cli_checks_real_parquet_and_writes_separate_consistent_cache(tmp_path):
    records, mapping, refs, current, source = fixtures()
    source_root, root, out = tmp_path / "original", tmp_path / "curated", tmp_path / "fixed"
    write_dataset(source_root, source)
    write_dataset(root, current, source_root)
    (root / "curation_source_map.json").write_text(json.dumps(mapping))
    reference_path = tmp_path / "reference.jsonl"
    reference_path.write_text("".join(json.dumps(r) + "\n" for r in refs))
    cache = root / "coding_v4"
    cache.mkdir()
    torch.save(repair.cache_from_records(records), cache / "policy_code_idx_custom.pt")
    (cache / "gripper_code_audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    order = [dict(episode_index=i, task_index=2, offset=i*6, length=6) for i in range(2)]
    (cache / "episode_order.json").write_text(json.dumps(order))
    before = snapshot_files(tmp_path)
    command = [sys.executable, str(SCRIPT), "--root", str(root), "--reference-audit", str(reference_path), "--out-dir", str(out)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stdout + result.stderr
    assert before.items() <= snapshot_files(tmp_path).items()
    report = json.loads((out / "knob_repair_report.json").read_text())
    assert report["repairs_by_kind"] == dict(restore_rotate=1, restore_boundary_release=1)
    assert report["calls_after"] == dict(grasp=2, rotate=2, release=2)
    info = json.loads((root / "meta/info.json").read_text())
    repair.validate_inputs(torch.load(out / "policy_code_idx_custom.pt", weights_only=True), order,
                           repair.read_audit(out / "gripper_code_audit.jsonl"), info)
    rerun = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    assert rerun.returncode != 0
    assert "never overwritten" in rerun.stderr
    assert not list(tmp_path.glob(".fixed.partial-*"))
