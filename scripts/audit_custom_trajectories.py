#!/usr/bin/env python3
"""Read-only custom34/action20 trajectory dynamics audit; no video or model loading."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np


EEF_UNITS = {
    "position_step": "m", "speed": "m/s", "velocity_change": "m/s",
    "acceleration": "m/s^2", "jerk": "m/s^3",
    "rotation_step": "deg", "angular_speed": "deg/s",
    "angular_velocity_change": "deg/s", "angular_acceleration": "deg/s^2",
    "angular_jerk": "deg/s^3",
}
SUPPORT = {"position_step": 1, "speed": 1, "velocity_change": 2,
           "acceleration": 2, "jerk": 3, "rotation_step": 1,
           "angular_speed": 1, "angular_velocity_change": 2,
           "angular_acceleration": 2, "angular_jerk": 3,
           "joint_step": 1, "joint_speed": 1, "joint_velocity_change": 2,
           "joint_acceleration": 2, "joint_jerk": 3}
EVENT_FIELDS = ["episode", "task_index", "source", "side", "metric", "value", "unit",
                "frame_start", "frame_end", "timestamp_start", "timestamp_end",
                "statistical_candidate", "limit_exceeded", "statistical_threshold",
                "configured_limit", "near_gripper_transition", "crosses_task_boundary", "joint_index"]


def rotation6d_matrices(values):
    """First two COLUMNS -> orthonormal matrices, with invalid frames masked."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    clean = np.where(np.isfinite(values), values, 0.0)
    first, second = clean[:, :3], clean[:, 3:]
    norm = np.linalg.norm(first, axis=1)
    valid &= norm > 1e-8
    x = first / np.maximum(norm[:, None], 1e-8)
    y = second - np.sum(x * second, axis=1)[:, None] * x
    norm_y = np.linalg.norm(y, axis=1)
    valid &= norm_y > 1e-8
    y /= np.maximum(norm_y[:, None], 1e-8)
    matrices = np.stack((x, y, np.cross(x, y)), axis=-1)
    matrices[~valid] = np.eye(3)
    return matrices, valid


def rotation_vectors(matrices):
    """SO(3) shortest-arc logarithm, stable near zero and pi; no SciPy dependency."""
    r = np.asarray(matrices, dtype=np.float64)
    qabs = np.sqrt(np.maximum(0, np.stack((
        1 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2],
        1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2],
        1 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2],
        1 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2]), axis=1)))
    a, b, c = r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]
    d, e, f = r[:, 1, 0] + r[:, 0, 1], r[:, 2, 0] + r[:, 0, 2], r[:, 2, 1] + r[:, 1, 2]
    candidates = np.stack((np.stack((qabs[:, 0]**2, a, b, c), axis=1),
        np.stack((a, qabs[:, 1]**2, d, e), axis=1),
        np.stack((b, d, qabs[:, 2]**2, f), axis=1),
        np.stack((c, e, f, qabs[:, 3]**2), axis=1)), axis=1)
    chosen = qabs.argmax(axis=1)
    q = candidates[np.arange(len(r)), chosen] / (2 * np.maximum(qabs[np.arange(len(r)), chosen, None], 1e-12))
    q /= np.maximum(np.linalg.norm(q, axis=1)[:, None], 1e-12)
    q[q[:, 0] < 0] *= -1
    length = np.linalg.norm(q[:, 1:], axis=1)
    angle = 2 * np.arctan2(length, np.clip(q[:, 0], 0, 1))
    scale = np.divide(angle, length, out=np.full_like(angle, 2.0), where=length > 1e-12)
    return q[:, 1:] * scale[:, None]


def time_edges(times, min_dt=1e-4, gap_factor=5.0):
    times = np.asarray(times, dtype=np.float64)
    dt = np.diff(times)
    usable = np.isfinite(times[:-1]) & np.isfinite(times[1:]) & (dt > min_dt)
    median_dt = float(np.median(dt[usable])) if usable.any() else None
    gap = np.zeros(len(dt), dtype=bool)
    if median_dt is not None and gap_factor > 0:
        gap = usable & (dt > median_dt * gap_factor)
    return usable & ~gap, {
        "nonfinite_timestamp_frames": np.flatnonzero(~np.isfinite(times)).tolist(),
        "nonpositive_or_tiny_dt_end_frames": (np.flatnonzero(~usable) + 1).tolist(),
        "long_gap_end_frames": (np.flatnonzero(gap) + 1).tolist(),
        "median_positive_dt_seconds": median_dt,
    }


def differences(values, times, valid_edges):
    """Vector finite differences on nonuniform time; invalid intervals never bridged.

    Acceleration = 2*(v_next-v_prev)/(t[i+2]-t[i]). Jerk is the third
    divided difference times 3!, exact for a cubic position trajectory.
    Arrays are aligned to the LAST support frame; leading entries are NaN.
    """
    n, dim = values.shape
    arrays = [np.full((n, dim), np.nan) for _ in range(4)]
    if n < 2:
        return arrays
    step = np.diff(values, axis=0)
    good = valid_edges & np.isfinite(values[:-1]).all(1) & np.isfinite(values[1:]).all(1)
    arrays[0][1:][good] = step[good]
    arrays[1][1:][good] = step[good] / np.diff(times)[good, None]
    if n >= 3:
        good2 = good[:-1] & good[1:]
        delta_v = np.diff(arrays[1][1:], axis=0)
        arrays[2][2:][good2] = 2 * delta_v[good2] / (times[2:] - times[:-2])[good2, None]
    if n >= 4:
        good3 = good[:-2] & good[1:-1] & good[2:]
        delta_a = np.diff(arrays[2][2:], axis=0)
        arrays[3][3:][good3] = 3 * delta_a[good3] / (times[3:] - times[:-3])[good3, None]
    return arrays


def vector_changes(vectors):
    result = np.full_like(vectors, np.nan)
    result[1:] = np.diff(vectors, axis=0)
    return result


def eef_metrics(pose, times, edges):
    matrices, valid_rotation = rotation6d_matrices(pose[:, 3:9])
    position = pose[:, :3]
    step, velocity, acceleration, jerk = differences(position, times, edges)
    angular_step = np.full((len(pose), 3), np.nan)
    valid = edges & valid_rotation[:-1] & valid_rotation[1:]
    relative = matrices[1:] @ matrices[:-1].transpose(0, 2, 1)
    angular_step[1:][valid] = np.rad2deg(rotation_vectors(relative[valid]))
    omega = np.full_like(angular_step, np.nan)
    omega[1:][valid] = angular_step[1:][valid] / np.diff(times)[valid, None]
    alpha = np.full_like(omega, np.nan)
    angular_jerk = np.full_like(omega, np.nan)
    if len(pose) >= 3:
        valid2 = valid[:-1] & valid[1:]
        delta = np.diff(omega[1:], axis=0)
        alpha[2:][valid2] = 2 * delta[valid2] / (times[2:] - times[:-2])[valid2, None]
    if len(pose) >= 4:
        valid3 = valid[:-2] & valid[1:-1] & valid[2:]
        delta = np.diff(alpha[2:], axis=0)
        angular_jerk[3:][valid3] = 3 * delta[valid3] / (times[3:] - times[:-3])[valid3, None]
    vectors = dict(position_step=step, speed=velocity, velocity_change=vector_changes(velocity),
        acceleration=acceleration, jerk=jerk, rotation_step=angular_step, angular_speed=omega,
        angular_velocity_change=vector_changes(omega), angular_acceleration=alpha, angular_jerk=angular_jerk)
    return {key: np.linalg.norm(value, axis=1) for key, value in vectors.items()}, {
        "invalid_position_frames": np.flatnonzero(~np.isfinite(position).all(1)).tolist(),
        "invalid_rotation6d_frames": np.flatnonzero(~valid_rotation).tolist(),
        "near_180_degree_rotation_step_frames": np.flatnonzero(np.linalg.norm(angular_step, axis=1) >= 175).tolist(),
    }


def analyze_episode(episode, state, action, times, tasks, min_dt=1e-4, gap_factor=5.0,
                    gripper_delta=0.5, joint_unit="stored"):
    state, action = np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)
    times, tasks = np.asarray(times, dtype=float), np.asarray(tasks)
    if not np.isfinite(tasks).all() or not np.equal(tasks, np.floor(tasks)).all():
        raise ValueError(f"episode {episode}: task_index must contain finite integers")
    tasks = tasks.astype(np.int64)
    n = len(times)
    if state.shape != (n, 34) or action.shape != (n, 20) or tasks.shape != (n,):
        raise ValueError(f"episode {episode}: expected state[N,34], action[N,20], timestamp[N], task_index[N]")
    edges, timing = time_edges(times, min_dt, gap_factor)
    tracks = []
    for source, values, pose_offsets, gripper_offsets in (
            ("state", state, (14, 23), (32, 33)), ("action", action, (0, 9), (18, 19))):
        for side, offset, grip_index in zip(("left", "right"), pose_offsets, gripper_offsets):
            scores, quality = eef_metrics(values[:, offset:offset + 9], times, edges)
            grip = values[:, grip_index]
            quality["nonfinite_gripper_frames"] = np.flatnonzero(~np.isfinite(grip)).tolist()
            changed = np.zeros(n, dtype=bool)
            changed[1:] = np.isfinite(grip[:-1]) & np.isfinite(grip[1:]) & (np.abs(np.diff(grip)) >= gripper_delta)
            tracks.append(dict(episode=episode, source=source, side=side, times=times, tasks=tasks,
                               scores=scores, quality=quality, grip=grip, grip_changed=changed,
                               units=EEF_UNITS, joint_indices={}))
    for side, offset in (("left", 0), ("right", 7)):
        values = state[:, offset:offset + 7]
        step, velocity, acceleration, jerk = differences(values, times, edges)
        vectors = dict(joint_step=step, joint_speed=velocity, joint_velocity_change=vector_changes(velocity),
                       joint_acceleration=acceleration, joint_jerk=jerk)
        scores = {key: np.max(np.abs(value), axis=1) for key, value in vectors.items()}
        indices = {key: np.argmax(np.where(np.isfinite(value), np.abs(value), -1), axis=1) + offset
                   for key, value in vectors.items()}
        units = {key: joint_unit + ("" if SUPPORT[key] == 1 and key == "joint_step" else
                 "/s" if key in ("joint_speed", "joint_velocity_change") else
                 "/s^2" if key == "joint_acceleration" else "/s^3") for key in scores}
        tracks.append(dict(episode=episode, source="state_joints", side=side, times=times, tasks=tasks,
                           scores=scores, quality={"invalid_joint_frames": np.flatnonzero(~np.isfinite(values).all(1)).tolist()},
                           grip=np.full(n, np.nan), grip_changed=np.zeros(n, dtype=bool), units=units,
                           joint_indices=indices))
    return tracks, timing


def summarize(tracks, limits=None, mad_k=6.0, top_k=30):
    limits = limits or {}
    groups, group_units = defaultdict(list), {}
    for track in tracks:
        for task in np.unique(track["tasks"]):
            for metric, scores in track["scores"].items():
                selected = scores[(track["tasks"] == task) & np.isfinite(scores)]
                if len(selected):
                    key = (int(task), track["source"], track["side"], metric)
                    groups[key].append(selected)
                    group_units[key] = track["units"][metric]
    baselines = {}
    for key, chunks in groups.items():
        values = np.concatenate(chunks)
        median = float(np.median(values))
        sigma = float(1.4826 * np.median(np.abs(values - median)))
        fallback = sigma <= 1e-12 * max(1, abs(median))
        threshold = float(np.quantile(values, .995)) if fallback else median + mad_k * sigma
        baselines[key] = dict(unit=group_units[key], samples=len(values), median=median, p95=float(np.quantile(values, .95)),
            p99=float(np.quantile(values, .99)), maximum=float(values.max()), threshold=threshold,
            threshold_method="q99.5 (zero-MAD fallback)" if fallback else f"median + {mad_k} * 1.4826*MAD",
            limit=limits.get(key[-1]), statistical_candidates=0, limit_exceeded=0)
    events = defaultdict(list)
    rows = []
    for track in tracks:
        ep_row = dict(episode=track["episode"], source=track["source"], side=track["side"],
                      task_indices=",".join(str(i) for i in np.unique(track["tasks"])),
                      frames=len(track["times"]), gripper_transitions=int(track["grip_changed"].sum()))
        for metric, values in track["scores"].items():
            finite = values[np.isfinite(values)]
            ep_row[metric + "_max"] = float(finite.max()) if len(finite) else None
            for task in np.unique(track["tasks"]):
                key = (int(task), track["source"], track["side"], metric)
                if key not in baselines:
                    continue
                base = baselines[key]
                tolerance = 1e-10 * max(1, abs(base["threshold"]))
                statistical = np.isfinite(values) & (values > base["threshold"] + tolerance)
                exceeded = np.zeros(len(values), dtype=bool) if base["limit"] is None else np.isfinite(values) & (values > base["limit"])
                in_task = track["tasks"] == task
                base["statistical_candidates"] += int(np.sum(statistical & in_task))
                base["limit_exceeded"] += int(np.sum(exceeded & in_task))
                candidates = np.flatnonzero((statistical | exceeded) & in_task)
                # Keep only the largest values per metric, including declared limit violations first.
                candidates = sorted(candidates, key=lambda i: (bool(exceeded[i]), values[i]), reverse=True)[:top_k]
                for frame in candidates:
                    start = max(0, int(frame) - SUPPORT[metric])
                    nearby = track["grip_changed"][max(0, start - 2):min(len(values), frame + 3)].any()
                    events[metric].append(dict(episode=track["episode"], task_index=int(task),
                        source=track["source"], side=track["side"], metric=metric, value=float(values[frame]),
                        unit=track["units"][metric], frame_start=start, frame_end=int(frame),
                        timestamp_start=float(track["times"][start]), timestamp_end=float(track["times"][frame]),
                        statistical_candidate=bool(statistical[frame]), limit_exceeded=bool(exceeded[frame]),
                        statistical_threshold=base["threshold"], configured_limit=base["limit"],
                        near_gripper_transition=bool(nearby),
                        crosses_task_boundary=bool(np.any(track["tasks"][start:frame + 1] != task)),
                        joint_index=int(track["joint_indices"][metric][frame]) if metric in track["joint_indices"] else None))
                events[metric] = sorted(events[metric], key=lambda item: (item["limit_exceeded"], item["value"]), reverse=True)[:top_k]
        rows.append(ep_row)
    return baselines, rows, [event for metric in sorted(events) for event in events[metric]]


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_episodes(reader, selected, time_source):
    """Read selected packed Parquet columns once per file, never decode a video."""
    import pandas as pd
    import pyarrow.parquet as pq
    files = defaultdict(list)
    for episode in selected:
        files[reader.data_file(episode)].append(episode)
    for path, episodes in files.items():
        columns = ["episode_index", "frame_index", "task_index", "observation.state", "action"]
        if time_source == "timestamp":
            columns.append("timestamp")
        missing = set(columns) - set(pq.ParquetFile(path).schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}; use --time-source fps only if fixed physical timing is known")
        table = pd.read_parquet(path, columns=columns, filters=[("episode_index", "in", episodes)])
        grouped = {int(ep): data for ep, data in table.groupby("episode_index", sort=False)}
        for episode in episodes:
            if episode not in grouped:
                raise ValueError(f"episode {episode} absent from {path}")
            frame = grouped[episode].sort_values("frame_index").reset_index(drop=True)
            if len(frame) != reader.episode_length(episode) or not np.array_equal(frame.frame_index.to_numpy(), np.arange(len(frame))):
                raise ValueError(f"episode {episode}: invalid frame count/order")
            yield episode, frame


def plot_episode(path, tracks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 2, figsize=(15, 15), sharex=True)
    for column, side in enumerate(("left", "right")):
        for track in tracks:
            if track["side"] != side or track["source"] == "state_joints":
                continue
            for row, metric in enumerate(("speed", "acceleration", "jerk", "angular_speed", "angular_acceleration", "angular_jerk")):
                axes[row, column].plot(track["times"], track["scores"][metric], label=track["source"], linewidth=1)
                axes[row, column].set_ylabel(f"{metric} ({EEF_UNITS[metric]})")
                for frame in np.flatnonzero(track["grip_changed"]):
                    axes[row, column].axvline(track["times"][frame], color="gray", alpha=.15)
                axes[row, column].grid(alpha=.2)
        axes[0, column].set_title(side)
        axes[0, column].legend()
        axes[-1, column].set_xlabel("sample time (s); derivatives aligned to last support frame")
    fig.suptitle(f"Episode {tracks[0]['episode']} — raw finite differences, no smoothing")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--time-source", choices=("timestamp", "fps"), default="timestamp")
    parser.add_argument("--fps", type=float, help="Explicit physical sample rate; only used with --time-source fps")
    parser.add_argument("--min-dt", type=float, default=1e-4)
    parser.add_argument("--gap-factor", type=float, default=5.0, help="Skip derivatives across dt > factor * median positive dt; 0 disables")
    parser.add_argument("--mad-k", type=float, default=6.0)
    parser.add_argument("--top-k", type=int, default=30, help="Maximum exported events per metric, not an episode deletion rule")
    parser.add_argument("--limits", type=Path, help="Optional JSON metric -> robot-specific upper limit, in report units")
    parser.add_argument("--joint-unit", choices=("stored", "rad", "deg"), default="stored", help="Declare stored joint units; does not convert or unwrap them")
    parser.add_argument("--gripper-delta", type=float, default=.5)
    parser.add_argument("--plot-episodes", type=int, nargs="+", help="Optional PNG plots; requires matplotlib")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output path already exists; choose a new --out directory")
    if args.time_source == "fps" and (args.fps is None or not np.isfinite(args.fps) or args.fps <= 0):
        parser.error("--time-source fps requires an explicit positive --fps; encoded video fps is not physical timing evidence")
    if args.time_source == "timestamp" and args.fps is not None:
        parser.error("--fps is only used with --time-source fps")
    if args.top_k < 1 or not all(np.isfinite(v) and v > 0 for v in (args.min_dt, args.mad_k, args.gripper_delta)) or not np.isfinite(args.gap_factor) or args.gap_factor < 0:
        parser.error("Invalid threshold parameters")
    if args.plot_episodes:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            parser.error("Plots require matplotlib: python -m pip install matplotlib")
    limits = {} if args.limits is None else json.loads(args.limits.read_text())
    if not isinstance(limits, dict) or any(key not in SUPPORT or isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for key, value in limits.items()):
        parser.error("Limits must be known metric names mapped to finite positive numbers")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
    reader = V3EpisodeReader(args.root)
    selected = reader.episode_ids if args.episodes is None else list(dict.fromkeys(args.episodes))
    if set(selected) - set(reader.episode_ids):
        parser.error("--episodes includes unknown episode IDs")
    if args.plot_episodes and set(args.plot_episodes) - set(selected):
        parser.error("--plot-episodes must be included in the selected episodes")
    tracks, timing = [], {}
    for number, (episode, frame) in enumerate(read_episodes(reader, selected, args.time_source), start=1):
        times = frame.timestamp.to_numpy(dtype=float) if args.time_source == "timestamp" else np.arange(len(frame)) / args.fps
        episode_tracks, timing[episode] = analyze_episode(episode, np.stack(frame["observation.state"]),
            np.stack(frame.action), times, frame.task_index.to_numpy(), args.min_dt, args.gap_factor,
            args.gripper_delta, args.joint_unit)
        tracks.extend(episode_tracks)
        if number % 100 == 0 or number == len(selected):
            print(f"[trajectory-audit] analyzed {number}/{len(selected)} episodes", file=sys.stderr, flush=True)
    baselines, rows, events = summarize(tracks, limits, args.mad_k, args.top_k)
    report = dict(episodes=len(selected), frames=sum(reader.episode_length(ep) for ep in selected),
        root=str(reader.root), time_source=args.time_source, assumed_fps=args.fps,
        task_names=reader.tasks, min_dt=args.min_dt, gap_factor=args.gap_factor, joint_unit=args.joint_unit,
        limits=limits, has_robot_limits=bool(limits), exported_events=len(events), event_cap_per_metric=args.top_k,
        baselines=[dict(task_index=key[0], source=key[1], side=key[2], metric=key[3], **value)
                   for key, value in sorted(baselines.items())],
        timing_by_episode=timing,
        pose_quality_issues=[dict(episode=t["episode"], source=t["source"], side=t["side"], **t["quality"])
                             for t in tracks if any(t["quality"].values())],
        notes=["Candidates are statistical outliers or declared-limit violations, not automatic bad-episode labels.",
               "Timestamp is assumed acquisition time in seconds; video encoding fps does not verify physical timing.",
               "State is measured pose/joints; action is an absolute setpoint, not measured velocity.",
               "SO(3) increments use the fixed reference frame and shortest arc; rotations above pi between samples are ambiguous.",
               "No smoothing, cross-episode differences, padding, normalization or video decoding is used.",
               "Joint angles are not unwrapped; confirm units and continuous-joint encoder wrapping.",
               "Candidate/violation counts are metric points; one frame can be counted by several metrics.",
               "Derivatives amplify quantization/noise; inspect neighboring frames and contact/gripper events before changing data."])
    args.out.mkdir(parents=True)
    write_csv(args.out / "episode_metrics.csv", rows)
    write_csv(args.out / "events.csv", events, EVENT_FIELDS)
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for episode in args.plot_episodes or []:
        current = [track for track in tracks if track["episode"] == episode]
        plot_episode(args.out / f"episode_{episode:06d}.png", current)
    print(json.dumps(dict(episodes=report["episodes"], frames=report["frames"],
        exported_events=len(events), has_robot_limits=bool(limits),
        limit_violations=sum(b["limit_exceeded"] for b in baselines.values()) if limits else None,
        timing_issue_episodes=sum(any(value for key, value in item.items() if key != "median_positive_dt_seconds") for item in timing.values()),
        pose_issue_tracks=len(report["pose_quality_issues"]),
        statistical_candidates=sum(b["statistical_candidates"] for b in baselines.values()),
        output=str(args.out.resolve()), note="Review candidates; no input data was changed."), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
