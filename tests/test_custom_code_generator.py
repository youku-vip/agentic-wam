"""New generation needs no prior labels and never fabricates cropped events."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tokenize
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import generate_custom_code_cache as generator
import build_v3_gripper_code_fixed as builder
from rebuild_semantic_code_cache import calls, validate_inputs


def options(**changes):
    args = SimpleNamespace(open_is_low=False, stable_frames=2, knob_axis="auto", knob_angle_sign=1,
        knob_max_xyz_drift_m=.02, knob_min_angle_deg=8., knob_min_axis_coherence=.8)
    opts = generator.options(args)
    for key, value in changes.items():
        setattr(opts, key, value)
    return opts


def states(n=32):
    result = np.zeros((n, 34), dtype=np.float32)
    result[:, 17:23] = result[:, 26:32] = [1, 0, 0, 0, 1, 0]
    result[:, 32:34] = 1
    result[6:24, 32] = 0
    return result


def set_yaw(state, yaw):
    radians = np.deg2rad(yaw)
    state[:, 17:23] = np.column_stack((np.cos(radians), np.sin(radians), np.zeros(len(state)),
                                       -np.sin(radians), np.cos(radians), np.zeros(len(state))))


def knob_states():
    result = states()
    result[:, 14] = .08
    result[10:21, 14] = 0
    set_yaw(result, np.r_[np.zeros(10), np.linspace(0, 90, 11), np.full(11, 90)])
    return result


def test_refines_retained_stationary_window_without_relaxing_drift():
    sample = knob_states()
    legacy, before = builder.measured_knob_rotation(sample, 7, 25, options(refine_knob_window=False))
    assert legacy is None
    assert before["xyz_max_drift_m"] == .08
    code, after = builder.measured_knob_rotation(sample, 7, 25, options())
    assert calls(builder.make_code([code])) == [("rotate", ["temperature control knob", 90.0], {"side": "left"})]
    assert after["rotation_window"] == [10, 20]
    assert after["rotation_window_xyz_drift_m"] <= .02
    assert after["full_interval_exceeds_xyz_tolerance"]
    assert after["rotation_window_candidate_count"] > 1
    assert "confirm knob engagement" in after["warning"]


def test_valid_full_interval_and_multi_turn_angle_are_preserved():
    sample = states(101)
    set_yaw(sample, np.linspace(0, 400, len(sample)))
    old = builder.measured_knob_rotation(sample, 0, 100, options(refine_knob_window=False))
    new = builder.measured_knob_rotation(sample, 0, 100, options())
    assert old == new
    assert new[1]["estimated_signed_angle_deg"] == pytest.approx(400, abs=.01)


def test_angle_sign_applies_to_new_window():
    code, metadata = builder.measured_knob_rotation(knob_states(), 7, 25, options(knob_angle_sign=-1))
    assert "-90.00" in code
    assert metadata["knob_angle_sign_calibration"] == -1


def test_refinement_does_not_hide_a_rotate_and_return_in_valid_full_interval():
    sample = states(41)
    set_yaw(sample, np.r_[np.linspace(0, 90, 21), np.linspace(90, 0, 21)[1:]])
    code, metadata = builder.measured_knob_rotation(sample, 0, 40, options())
    assert code is None
    assert "net signed" in metadata["warning"]


def test_rejects_ambiguous_sample_jump_and_no_stationary_rotation():
    sample = knob_states()
    set_yaw(sample, np.r_[np.zeros(15), np.full(17, 180)])
    code, metadata = builder.measured_knob_rotation(sample, 7, 25, options())
    assert code is None
    assert "180 degrees" in metadata["warning"]
    sample = knob_states()
    sample[:, 14] = np.arange(len(sample)) * .03
    code, metadata = builder.measured_knob_rotation(sample, 7, 25, options())
    assert code is None
    assert metadata["rotation_window_candidate_count"] == 0


def test_terminal_candidate_is_not_promoted_without_confirmed_context():
    sample = knob_states()
    sample[24, 32] = 0
    sample = sample[:26]
    _, row = builder.label_episode(0, 2, sample, options())
    assert row["events"] == [[7, "close"]]
    assert row["gripper_detection"]["terminal_candidate"]["state"] == "open"
    assert [c[0] for s in row["spans"] for c in calls(s["code"])] == ["grasp", "rotate"]
    assert "terminal gripper candidate lacks stable confirmation" in row["warnings"]


def test_missing_grasp_is_not_invented_for_a_place_only_fragment():
    sample = states()
    sample[:, 32] = 1
    sample[:20, 33] = 0
    _, row = builder.label_episode(0, 1, sample, options())
    assert [c[0] for s in row["spans"] for c in calls(s["code"])] == ["place"]


def write_dataset(root, episode_states, tasks, source_root=None, mapping=None):
    for directory in ("meta/episodes/chunk-000", "data/chunk-000"):
        (root / directory).mkdir(parents=True)
    info = dict(codebase_version="v3.0", total_episodes=len(episode_states),
        total_frames=sum(len(s) for s in episode_states.values()), fps=10,
        features={"observation.state": dict(dtype="float32", shape=[34])},
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    if source_root:
        info["trajectory_curation"] = dict(source_root=str(source_root))
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text("".join(json.dumps(dict(task_index=t, task=f"task {t}")) + "\n" for t in sorted(set(tasks.values()))))
    (root / "dataset_stats_eef64.json").write_text('{"keep": true}\n')
    metadata, frames, offset = [], [], 0
    for ep, state in episode_states.items():
        n = len(state)
        metadata.append(dict(episode_index=ep, **{"data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": offset, "dataset_to_index": offset + n}))
        frames.append(pd.DataFrame(dict(index=np.arange(offset, offset + n), episode_index=np.full(n, ep),
            task_index=np.full(n, tasks[ep]), frame_index=np.arange(n), **{"observation.state": list(state)})))
        offset += n
    pd.DataFrame(metadata).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.concat(frames).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    if mapping is not None:
        (root / "curation_source_map.json").write_text(json.dumps(mapping))
    return info


@pytest.fixture
def datasets(tmp_path):
    old, clean = tmp_path / "original", tmp_path / "curated"
    original = {7: states(), 11: states(), 20: knob_states(), 41: states(), 99: knob_states()}
    original[7][:, 14] = np.r_[np.linspace(0, .04, 24), np.linspace(.04, 0, 8)]
    original[11][:, 32] = 1
    original[11][:20, 33] = 0
    original[41][:, 32:34] = 1
    original[41][:, 14] = np.r_[np.linspace(0, .1, 16), np.linspace(.1, 0, 16)]
    original[99][24, 32] = 0
    source_tasks = {7: 0, 11: 1, 20: 2, 41: 3, 99: 2}
    write_dataset(old, original, source_tasks)
    current, mapping, tasks = {}, [], {}
    offset = 0
    for i, (ep, source) in enumerate(original.items()):
        n = 26 if ep == 99 else len(source)
        current[i] = source[:n].copy()
        tasks[i] = source_tasks[ep]
        mapping.append(dict(new_episode=i, source_episode=ep, source_start_frame=0, source_end_frame=n,
                            frames=n, new_global_offset=offset))
        offset += n
    info = write_dataset(clean, current, tasks, old, mapping)
    return old, clean, current, info


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_cli_generates_four_tasks_without_old_audits_and_preserves_inputs(datasets, tmp_path):
    old, root, _, info = datasets
    before_old, before_new = snapshot(old), snapshot(root)
    out = tmp_path / "regenerated"
    command = [sys.executable, str(REPO / "scripts/generate_custom_code_cache.py"),
               "--root", str(root), "--out-dir", str(out), "--confirm-curation-boundaries"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO,
                            env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot(old) == before_old
    assert snapshot(root) == before_new
    report = json.loads((out / "code_generation_report.json").read_text())
    assert report["old_cache_or_audit_required"] is False
    assert len(report["boundary_open_confirmations"]) == 1
    assert len(report["refined_rotation_windows"]) == 2
    assert report["calls"]["rotate"] == 2
    assert report["calls"]["release"] == 3
    assert report["calls"].get("pick", 0) == 0
    assert report["unresolved_terminal_candidates"] == []
    assert set(report["by_task"]) == {"0", "1", "2", "3"}
    assert report["by_task"]["3"]["calls"] == dict(move_arm=2)
    data = torch.load(out / "policy_code_idx_custom.pt", weights_only=True)
    rows = [json.loads(line) for line in (out / "gripper_code_audit.jsonl").read_text().splitlines()]
    validate_inputs(data, json.loads((out / "episode_order.json").read_text()), rows, info)
    for code in data["codes"]:
        assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(code).readline))
    rerun = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    assert rerun.returncode != 0
    assert "never overwritten" in rerun.stderr


def test_retained_only_generation_leaves_single_terminal_open_unconfirmed(datasets, tmp_path):
    old, root, _, _ = datasets
    old.rename(tmp_path / "unavailable_original")
    report = generator.generate(root, tmp_path / "strict", options(), confirm_boundaries=False)
    assert report["calls"]["rotate"] == 2
    assert report["calls"]["release"] == 2
    assert len(report["unresolved_terminal_candidates"]) == 1


@pytest.mark.parametrize("problem", ["not_stable", "source_missing", "retained_state_mismatch"])
def test_boundary_confirmation_needs_matching_states_and_actual_future_open(datasets, problem):
    old, root, current, info = datasets
    context = generator.CurationBoundaryContext(root, info)
    if problem == "source_missing":
        (old / "meta/info.json").unlink()
    elif problem == "not_stable":
        path = old / "data/chunk-000/file-000.parquet"
        table = pd.read_parquet(path)
        index = table.index[(table.episode_index == 99) & (table.frame_index == 26)][0]
        state = table.loc[index, "observation.state"].copy()
        state[32] = 0
        table.at[index, "observation.state"] = state
        table.to_parquet(path, index=False)
    else:
        current[4][10, 14] += .001
        with pytest.raises(ValueError, match="do not match curation map"):
            context.confirm(4, 2, current[4], options())
        return
    proof = context.confirm(4, 2, current[4], options())
    assert proof["confirmed"] is False
    _, record = builder.label_episode(4, 2, current[4], options(), terminal_open_context=proof)
    assert [c[0] for s in record["spans"] for c in calls(s["code"])] == ["grasp", "rotate"]


def shell_environment(root, tool, out):
    return dict(os.environ, PROJECT_DIR=str(REPO), PY=sys.executable, CLEAN_ROOT=str(root),
                TOOL_DIR=str(tool), OUT_DIR=str(out), CONFIRM_CURATED_BOUNDARIES="1",
                KNOB_AXIS="auto", KNOB_ANGLE_SIGN="1")


def test_shell_entrypoint_generates_verifies_and_preserves_existing_output(datasets, tmp_path):
    source, root, _, _ = datasets
    out = tmp_path / "complete-cache"
    before_source, before_root = snapshot(source), snapshot(root)
    command = ["bash", str(REPO / "scripts/generate_custom_code_data.sh")]
    environment = shell_environment(root, REPO, out)
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"saved_cache_checks": "passed"' in result.stdout
    assert "data.code_cache=" + str(out / "policy_code_idx_custom.pt") in result.stdout
    assert snapshot(source) == before_source
    assert snapshot(root) == before_root
    before_output = snapshot(out)
    repeated = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, env=environment)
    assert repeated.returncode != 0
    assert "输出目录已存在" in repeated.stderr
    assert snapshot(out) == before_output


def test_shell_bootstrap_installs_verified_zip_without_network(datasets, tmp_path):
    _, root, _, _ = datasets
    tool, out = tmp_path / "installed-tool", tmp_path / "bootstrapped-cache"
    environment = shell_environment(root, tool, out)
    environment["CODEGEN_PACKAGE_URL"] = (REPO / "deliveries/custom-code-generator-v2.zip").as_uri()
    result = subprocess.run(["bash", str(REPO / "scripts/generate_custom_code_data.sh")],
        capture_output=True, text=True, cwd=tmp_path, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tool / "MANIFEST.json").is_file()
    assert (out / "policy_code_idx_custom.pt").is_file()
    assert '"saved_cache_checks": "passed"' in result.stdout


def test_shell_bootstrap_rejects_bad_download_without_publishing_tools(datasets, tmp_path):
    _, root, _, _ = datasets
    bad = tmp_path / "bad-package.zip"
    bad.write_bytes(b"This is not the approved tool package")
    tool, out = tmp_path / "unpublished-tool", tmp_path / "unpublished-cache"
    environment = shell_environment(root, tool, out)
    environment["CODEGEN_PACKAGE_URL"] = bad.as_uri()
    result = subprocess.run(["bash", str(REPO / "scripts/generate_custom_code_data.sh")],
        capture_output=True, text=True, cwd=tmp_path, env=environment)
    assert result.returncode != 0
    assert "SHA256 不匹配" in result.stderr
    assert not tool.exists()
    assert not out.exists()
