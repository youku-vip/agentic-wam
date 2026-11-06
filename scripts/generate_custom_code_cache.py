#!/usr/bin/env python3
"""Generate comment-free custom34 code from dataset states, without old caches/audits.

Refine excessive-XYZ knob intervals using retained poses, then merge semantic
release/place retreat spans. Optionally confirm a cropped terminal opening
using source states located by curation_source_map.json; never read source
poses to generate targets, nor invent a release without stable confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_v3_gripper_code_fixed import TASKS, detect_events, iter_episodes, label_episode
from rebuild_semantic_code_cache import calls, merge_record, validate_inputs, without_comments

VERSION = "custom-code-from-retained-states-v2"


class CurationBoundaryContext:
    """Read source gripper context only for an unconfirmed cropped opening."""

    def __init__(self, root, info):
        self.map_path = root / "curation_source_map.json"
        rows = json.loads(self.map_path.read_text(encoding="utf-8"))
        self.mapping = {}
        for row in rows:
            episode = row["new_episode"]
            if type(episode) is not int or episode in self.mapping:
                raise ValueError("Duplicate or invalid curation episode ID")
            self.mapping[episode] = row
        self.source_root = info.get("trajectory_curation", {}).get("source_root")
        if not self.source_root:
            raise ValueError("Missing trajectory_curation.source_root; omit --confirm-curation-boundaries to use retained data only")
        if Path(self.source_root).resolve() == root:
            raise ValueError("Source and curated roots cannot be the same")
        self.reader = None

    def confirm(self, episode, task, states, opts):
        side = TASKS[task][0]
        column = 32 if side == "left" else 33
        events, info = detect_events(states[:, column], open_is_high=opts.open_is_high,
                                    min_range=opts.min_range, stable_frames=opts.stable_frames)
        pending = info.get("terminal_candidate", {})
        if pending.get("state") != "open" or not events or events[-1][1] != "close":
            return None
        mapping = self.mapping[episode]
        first, end = mapping["source_start_frame"], mapping["source_end_frame"]
        if (mapping["frames"] != len(states) or end - first != len(states) or not 0 <= first < end):
            raise ValueError(f"Episode {episode}: source mapping length mismatch")
        proof = dict(confirmed=False, retained_frames=len(states),
            source_episode=mapping["source_episode"], source_end_frame=end,
            pending_retained_samples=pending["samples"], required_samples=opts.stable_frames)
        try:
            if self.reader is None:
                from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
                self.reader = V3EpisodeReader(self.source_root)
            frame = self.reader.read_episode(mapping["source_episode"], columns=["observation.state", "task_index"])
        except FileNotFoundError as exc:
            return proof | dict(reason="source_context_unavailable", detail=str(exc))
        source = np.stack(frame["observation.state"]).astype(np.float32)
        if (source.ndim != 2 or source.shape[1] != 34 or end > len(source)
                or not np.all(frame["task_index"].to_numpy() == task)
                or not np.array_equal(source[first:end], states)):
            raise ValueError(f"Episode {episode}: source task/retained states do not match curation map")
        proof["retained_state_equality"] = True
        needed = opts.stable_frames - pending["samples"]
        if needed < 1 or end + needed > len(source):
            return proof | dict(reason="insufficient_source_samples_after_crop")
        future = source[end:end + needed, column]
        open_samples = future >= info["high"] if opts.open_is_high else future <= info["low"]
        if not np.isfinite(future).all() or not open_samples.all():
            return proof | dict(reason="source_opening_not_stable")
        extended = np.r_[states[:, column], future]
        extended_events, _ = detect_events(extended, open_is_high=opts.open_is_high,
            min_range=opts.min_range, stable_frames=opts.stable_frames)
        expected = events + [(len(extended) - 1, "open")]
        if extended_events != expected:
            return proof | dict(reason="extended_gripper_detection_disagrees")
        return proof | dict(confirmed=True, retained_open_frame=len(states) - 1,
            source_confirmed_open_frame=end + needed - 1,
            source_gripper_context=source[max(first, end - opts.stable_frames):end + needed, column].tolist(),
            note="Only gripper confirmation uses source context; targets/angles use retained poses")


def options(args):
    return SimpleNamespace(open_is_high=not args.open_is_low, min_range=.01,
        stable_frames=args.stable_frames, task3_open_push=True, min_motion_m=.015,
        retreat_min_motion_m=.015, retreat_step_m=.0015, retreat_max_waypoints=1,
        min_rot_deg=8., knob_angle_deg=None, knob_axis=args.knob_axis,
        knob_angle_sign=args.knob_angle_sign, gripper_full_stroke_mm=70.,
        knob_max_xyz_drift_m=args.knob_max_xyz_drift_m,
        knob_min_angle_deg=args.knob_min_angle_deg,
        knob_min_axis_coherence=args.knob_min_axis_coherence,
        refine_knob_window=True, knob_refine_min_frames=3)


def generate(root, out, opts, confirm_boundaries=False):
    root, out = root.expanduser().resolve(), out.expanduser().resolve()
    if out.exists():
        raise ValueError("Output must be a new directory; existing caches are never overwritten")
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Expected LeRobot v3.0 custom34 dataset")
    boundary = CurationBoundaryContext(root, info) if confirm_boundaries else None
    records, order, codes, lookup, ids = [], [], [""], {"": 0}, []
    counts, by_task = Counter(), {}
    refined, confirmed, pending = [], [], []
    merged_retreats = removed_moves = 0
    for ep, task, states in iter_episodes(root):
        if task not in TASKS:
            raise ValueError(f"Unknown task index {task}")
        context = boundary.confirm(ep, task, states, opts) if boundary else None
        _, record = label_episode(ep, task, states, opts, terminal_open_context=context)
        record, changes = merge_record(record)
        merged_retreats += len(changes)
        removed_moves += sum(change["removed_move_arm_calls"] for change in changes)
        task_counts = by_task.setdefault(str(task), dict(episodes=0, frames=0, labeled_frames=0, calls=Counter()))
        task_counts["episodes"] += 1
        task_counts["frames"] += len(states)
        names = []
        for span in record["spans"]:
            span["code"] = without_comments(span["code"])
            names.extend(name for name, _, _ in calls(span["code"]))
            if span["code"] not in lookup:
                lookup[span["code"]] = len(codes)
                codes.append(span["code"])
            length = span["end"] - span["start"]
            ids.extend([lookup[span["code"]]] * length)
            if span["code"]:
                task_counts["labeled_frames"] += length
            if span["motion"].get("rotation_window_refined"):
                refined.append(dict(episode_index=ep, window=span["motion"]["rotation_window"],
                    angle_deg=span["motion"]["estimated_signed_angle_deg"]))
        counts.update(names)
        task_counts["calls"].update(names)
        if context and context["confirmed"]:
            confirmed.append(dict(episode_index=ep, **context))
        if record["gripper_detection"].get("terminal_candidate") and not (context or {}).get("confirmed"):
            pending.append(dict(episode_index=ep, task_index=task,
                candidate=record["gripper_detection"]["terminal_candidate"], source_context=context))
        order.append(dict(trajectory_id=f"ep{ep:06d}", episode_index=ep, task_index=task,
                          offset=len(ids) - len(states), length=len(states)))
        records.append(record)
        if len(records) % 200 == 0:
            print(f"[codegen] episodes={len(records)} frames={len(ids)} refined_rotations={len(refined)}", file=sys.stderr, flush=True)
    if boundary and set(boundary.mapping) != {r["episode_index"] for r in records}:
        raise ValueError("Curation source map does not cover exactly the dataset episodes")
    data = dict(codes=codes, code_id_by_idx=torch.tensor(ids, dtype=torch.int32))
    validate_inputs(data, order, records, info)
    missing_rotate = [r["episode_index"] for r in records if r["task_index"] == 2
                      and not any(name == "rotate" for s in r["spans"] for name, _, _ in calls(s["code"]))]
    report = dict(version=VERSION, dataset_root=str(root), episodes=len(records), frames=len(ids),
        generated_code_comments=False, old_cache_or_audit_required=False,
        target_pose_source="retained dataset observation.state", calls=dict(counts), by_task=by_task,
        refined_rotation_windows=refined, boundary_open_confirmations=confirmed,
        unresolved_terminal_candidates=pending, task2_without_rotate=missing_rotate,
        merged_retreats=merged_retreats, removed_move_arm_calls=removed_moves,
        settings=vars(opts), source_context_requested=confirm_boundaries,
        dataset_and_statistics_unchanged=True,
        input_sha256={"meta/info.json": hashlib.sha256(info_path.read_bytes()).hexdigest()},
        verified=["Contiguous raw frame enumeration", "Generated cache/audit/order consistency",
                  "Rotation angles use retained poses and unchanged numerical thresholds"],
        not_verified=["Physical knob engagement and signed-angle calibration", "Video alignment", "Training improvement", "Tokenizer limits"])
    if boundary:
        report["input_sha256"]["curation_source_map.json"] = hashlib.sha256(boundary.map_path.read_bytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    try:
        torch.save(data, stage / "policy_code_idx_custom.pt")
        (stage / "episode_order.json").write_text(json.dumps(order, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "gripper_code_audit.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        (stage / "code_generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if out.exists():
            raise ValueError("Output appeared during generation; refusing to overwrite it")
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps(dict(version=VERSION, episodes=len(records), frames=len(ids), calls=dict(counts),
        refined_rotation_windows=len(refined), boundary_open_confirmations=len(confirmed),
        unresolved_terminal_candidates=len(pending), task2_without_rotate=missing_rotate,
        generated_code_comments=False, dataset_and_statistics_unchanged=True, output=str(out)), ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Primary dataset: an original or curated custom34 root")
    parser.add_argument("--out-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--confirm-curation-boundaries", action="store_true",
        help="Read source gripper context via curation metadata only for unconfirmed cropped openings; no old audit needed")
    parser.add_argument("--open-is-low", action="store_true", help="Default: normalized gripper 1=open, 0=closed")
    parser.add_argument("--stable-frames", type=int, default=2)
    parser.add_argument("--knob-axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--knob-angle-sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--knob-max-xyz-drift-m", type=float, default=.02)
    parser.add_argument("--knob-min-angle-deg", type=float, default=8.)
    parser.add_argument("--knob-min-axis-coherence", type=float, default=.8)
    args = parser.parse_args()
    if (args.stable_frames < 1 or not np.isfinite([args.knob_max_xyz_drift_m, args.knob_min_angle_deg, args.knob_min_axis_coherence]).all()
            or args.knob_max_xyz_drift_m < 0 or args.knob_min_angle_deg < 0
            or not 0 <= args.knob_min_axis_coherence <= 1):
        parser.error("Invalid stability, drift, angle or coherence settings")
    generate(args.root, args.out_dir, options(args), args.confirm_curation_boundaries)


if __name__ == "__main__":
    main()
