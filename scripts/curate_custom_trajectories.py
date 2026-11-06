#!/usr/bin/env python3
"""Paper-informed stability filtering for custom34/action20; original values stay intact.

Open-HumanScale, arXiv:2611.06817v1, section 2.2 (pp. 6-7), motivates
residual/dynamics checks followed by continuous-segment selection. The smoother,
thresholds, support-frame exclusion and robot boundary rules here are adaptations,
not parameters disclosed in the paper. No model or robot is run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_custom_trajectories import eef_metrics, read_episodes, time_edges

VERSION = "custom-paper-curation-v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/trajectory_curation.example.json"
METRICS = {"position_residual": (0, "m"), "position_step": (1, "m"),
           "rotation_step": (1, "deg"), "acceleration": (2, "m/s^2"),
           "jerk": (3, "m/s^3")}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config):
    if set(config) != {"description", "smoothing", "thresholds", "min_segment_frames", "idle"}:
        raise ValueError("Config must contain description, smoothing, thresholds, min_segment_frames and idle")
    smoothing = config["smoothing"]
    if set(smoothing) != {"window_frames", "polynomial_degree"}:
        raise ValueError("Invalid smoothing config")
    window, degree = smoothing["window_frames"], smoothing["polynomial_degree"]
    if (type(window) is not int or window < 3 or window % 2 != 1 or
            type(degree) is not int or not 0 <= degree < window):
        raise ValueError("Smoother requires an odd window >=3 and 0<=degree<window")
    thresholds = config["thresholds"]
    if set(thresholds) != set(METRICS):
        raise ValueError(f"Threshold keys must be {sorted(METRICS)}; use null to disable a check")
    for key, value in thresholds.items():
        if value is not None and (type(value) not in (float, int) or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key}: threshold must be finite positive or null")
    if not any(v is not None for v in thresholds.values()):
        raise ValueError("Enable at least one stability threshold")
    if type(config["min_segment_frames"]) is not int or config["min_segment_frames"] < 2:
        raise ValueError("min_segment_frames must be an integer >=2")
    idle = config["idle"]
    if set(idle) != {"enabled", "window_frames", "position_range_m", "min_active_fraction"}:
        raise ValueError("Invalid idle config")
    if (type(idle["enabled"]) is not bool or type(idle["window_frames"]) is not int or idle["window_frames"] < 2
            or type(idle["position_range_m"]) not in (int, float) or not np.isfinite(idle["position_range_m"])
            or idle["position_range_m"] <= 0 or type(idle["min_active_fraction"]) not in (int, float)
            or not np.isfinite(idle["min_active_fraction"]) or not 0 < idle["min_active_fraction"] <= 1):
        raise ValueError("Invalid idle window, motion threshold or active fraction")
    return config


def arrays(frame):
    state = np.stack(frame["observation.state"]).astype(np.float64)
    action = np.stack(frame["action"]).astype(np.float64)
    times = frame["timestamp"].to_numpy(dtype=np.float64)
    tasks = frame["task_index"].to_numpy(dtype=np.float64)
    n = len(frame)
    if state.shape != (n, 34) or action.shape != (n, 20) or n < 1:
        raise ValueError("Expected nonempty state[N,34] and action[N,20]")
    if (not np.array_equal(frame["frame_index"], np.arange(n)) or not np.isfinite(tasks).all()
            or not np.equal(tasks, np.floor(tasks)).all() or len(np.unique(tasks)) != 1):
        raise ValueError("Require contiguous frame_index from zero and one integer task per episode")
    return state, action, times, tasks.astype(np.int64)


def frame_hash(frame):
    digest = hashlib.sha256()
    for value in arrays(frame):
        dtype = "<i8" if value.dtype.kind in "iu" else "<f8"
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(value, dtype=dtype).tobytes())
    return digest.hexdigest()


def runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(padded) == 1).tolist(),
                    np.flatnonzero(np.diff(padded) == -1).tolist()))


def smooth_positions(positions, times, edges, window=7, degree=2):
    """Time-aware local polynomial fit; never span invalid time/position edges.

    This is used only to compute residuals. No fitted value is exported as a label.
    """
    n = len(times)
    result = np.full_like(positions, np.nan, dtype=np.float64)
    finite = np.isfinite(times) & np.isfinite(positions).all(axis=1)
    links = np.asarray(edges) & finite[:-1] & finite[1:]
    boundaries = [0, *(np.flatnonzero(~links) + 1), n]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        length = stop - start
        if length < degree + 1 or not finite[start:stop].all():
            continue
        width = min(window, length)
        # Uniform timestamps admit one shared least-squares stencil for the interior.
        if length >= window and np.allclose(np.diff(times[start:stop]), np.median(np.diff(times[start:stop])), rtol=1e-4, atol=1e-8):
            half = window // 2
            weights = np.linalg.pinv(np.vander(np.arange(-half, half + 1, dtype=float), degree + 1, increasing=True))[0]
            view = np.lib.stride_tricks.sliding_window_view(positions[start:stop], window, axis=0)
            result[start + half:stop - half] = np.einsum("ndw,w->nd", view, weights)
            indices = [*range(start, start + half), *range(stop - half, stop)]
        else:
            indices = range(start, stop)
        for index in indices:
            lo = max(start, min(index - width // 2, stop - width))
            hi = lo + width
            x = times[lo:hi] - times[index]
            scale = max(float(np.max(np.abs(x))), 1e-12)
            fit = np.linalg.lstsq(np.vander(x / scale, degree + 1, increasing=True), positions[lo:hi], rcond=None)[0]
            result[index] = fit[0]
    return result


def repeated_states(state):
    groups = defaultdict(list)
    for index, row in enumerate(state):
        if np.isfinite(row).all():
            groups[row.tobytes()].append(index)
    return [indices for indices in groups.values() if len(indices) > 1 and np.any(np.diff(indices) > 1)]


def trim_idle(state, start, stop, config):
    if not config["enabled"]:
        return start, stop, None, None
    width = config["window_frames"]
    if stop - start < width:
        return start, stop, None, "idle_window_too_short"
    active = []
    for index in range(start, stop - width + 1):
        ranges = [np.linalg.norm(np.ptp(state[index:index + width, off:off + 3], axis=0)) for off in (14, 23)]
        active.append(max(ranges) > config["position_range_m"])
    active = np.asarray(active)
    indices = np.flatnonzero(active)
    if not len(indices):
        return start, stop, 0.0, "idle_all_windows"
    fraction = float(active[indices[0]:indices[-1] + 1].mean())
    end = start + int(indices[-1]) + width
    start += int(indices[0])
    return start, end, fraction, "idle_active_fraction" if fraction < config["min_active_fraction"] else None


def analyze(episode, frame, config):
    state, action, times, tasks = arrays(frame)
    n = len(frame)
    edges, timing = time_edges(times)
    bad = ~np.isfinite(state).all(1) | ~np.isfinite(action).all(1) | ~np.isfinite(times)
    reasons = [set() for _ in range(n)]
    for index in np.flatnonzero(bad):
        reasons[index].add("nonfinite_row")
    for end in np.flatnonzero(~edges) + 1:
        bad[end - 1:end + 1] = True
        for index in (end - 1, end):
            reasons[index].add("invalid_time_edge")
    events = []
    smoothing = config["smoothing"]
    for source, values, offsets in (("state", state, (14, 23)), ("action", action, (0, 9))):
        for side, off in zip(("left", "right"), offsets):
            pose = values[:, off:off + 9]
            scores, quality = eef_metrics(pose, times, edges)
            fitted = smooth_positions(pose[:, :3], times, edges, smoothing["window_frames"], smoothing["polynomial_degree"])
            scores["position_residual"] = np.linalg.norm(pose[:, :3] - fitted, axis=1)
            for index in quality["invalid_rotation6d_frames"]:
                bad[index] = True
                reasons[index].add(f"{source}.{side}.invalid_rotation6d")
            for metric, (support, unit) in METRICS.items():
                threshold = config["thresholds"][metric]
                if threshold is None:
                    continue
                for end in np.flatnonzero(np.isfinite(scores[metric]) & (scores[metric] > threshold)):
                    start = max(0, int(end) - support)
                    bad[start:end + 1] = True
                    reason = f"{source}.{side}.{metric}"
                    for index in range(start, end + 1):
                        reasons[index].add(reason)
                    events.append(dict(episode=episode, task_index=int(tasks[0]), source=source, side=side,
                        metric=metric, value=float(scores[metric][end]), threshold=threshold, unit=unit,
                        support_start=start, support_end_exclusive=int(end) + 1))
    kept, dropped = [], []
    for start, stop in runs(~bad):
        begin, end, active_fraction, why = trim_idle(state, start, stop, config["idle"])
        if why is None and end - begin < config["min_segment_frames"]:
            why = "short_segment"
        item = dict(source_episode=episode, task_index=int(tasks[0]), start_frame=begin, end_frame=end,
                    frames=end - begin, active_fraction=active_fraction)
        if why:
            dropped.append(dict(**item, reason=why))
        else:
            kept.append(item)
    target = np.concatenate((state[1:, 14:34], state[-1:, 14:34]), axis=0)
    gripper_coverage = {}
    for side, column in (("left", 32), ("right", 33)):
        grip = state[:, column]
        transitions = np.flatnonzero(np.isfinite(grip[1:]) & np.isfinite(grip[:-1]) & (np.abs(np.diff(grip)) >= .5)) + 1
        missing = [int(index) for index in transitions if not any(
            segment["start_frame"] <= index - 1 and index < segment["end_frame"] for segment in kept)]
        gripper_coverage[side] = dict(original_transitions=len(transitions),
            retained_transitions=len(transitions) - len(missing), missing_transition_frames=missing)
    return dict(episode=episode, frames=n, task_index=int(tasks[0]), source_sha256=frame_hash(frame),
        next_state_action_exact=bool(np.array_equal(action, target)),
        bad_frames=np.flatnonzero(bad).tolist(), repeated_full_state_groups=repeated_states(state),
        retained_segments=kept, dropped_segments=dropped, timing=timing,
        gripper_transition_coverage=gripper_coverage), events, reasons


def load_raw(path):
    import pandas as pd
    document = json.loads(path.read_text(encoding="utf-8"))
    episodes = []
    seen = set()
    for item in document["episodes"]:
        episode = item["episode"]
        if type(episode) is not int or episode in seen:
            raise ValueError("Raw JSON requires unique integer episode IDs")
        seen.add(episode)
        frame = pd.DataFrame(item["frames"]).sort_values("frame_index").reset_index(drop=True)
        for key in ("observation.state", "action"):
            frame[key] = [np.asarray(row, dtype=np.float64) for row in frame[key]]
        episodes.append((episode, frame))
    return document, episodes


def slice_frame(frame, start, stop, new_episode, offset):
    """Retain original rows; new logical episode ends with a declared hold action."""
    part = frame.iloc[start:stop].copy()
    part = part.reset_index(drop=True)
    part["episode_index"] = np.full(len(part), new_episode, dtype=np.int64)
    part["frame_index"] = np.arange(len(part), dtype=np.int64)
    part["index"] = np.arange(offset, offset + len(part), dtype=np.int64)
    part["timestamp"] = (part["timestamp"] - part["timestamp"].iloc[0]).astype(frame["timestamp"].dtype)
    last_action = np.asarray(part["action"].iloc[-1])
    hold = np.asarray(part["observation.state"].iloc[-1])[14:34].astype(last_action.dtype).copy()
    changed = not np.array_equal(last_action, hold)
    part.at[len(part) - 1, "action"] = hold
    return part, changed


def validate_segments(document):
    records = {record["episode"]: record for record in document["episodes"]}
    if len(records) != len(document["episodes"]):
        raise ValueError("Duplicate source episodes in plan")
    last = {}
    for segment in document["segments"]:
        episode = segment["source_episode"]
        if episode not in records:
            raise ValueError("Segment references an unknown source episode")
        start, stop = segment["start_frame"], segment["end_frame"]
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= records[episode]["frames"]
                or start < last.get(episode, 0) or stop - start < 2):
            raise ValueError("Invalid, overlapping or unsorted half-open segment range")
        if any(start <= index < stop for index in records[episode]["bad_frames"]):
            raise ValueError("Segment includes rejected support frames")
        if segment["task_index"] != records[episode]["task_index"]:
            raise ValueError("Segment task differs from source task")
        last[episode] = stop
    return records


def preview_raw(document, frames_by_episode):
    records = validate_segments(document)
    output, offset = [], 0
    for new_episode, segment in enumerate(document["segments"]):
        episode = segment["source_episode"]
        if not records[episode]["next_state_action_exact"]:
            raise ValueError("Raw materialization requires verified action[t]=next_state[t] convention")
        start, stop = segment["start_frame"], segment["end_frame"]
        part, changed = slice_frame(frames_by_episode[episode], start, stop, new_episode, offset)
        output.append(dict(episode=new_episode, source_episode=episode, source_start_frame=start,
            source_end_frame=stop, terminal_action_changed=changed, frames=[dict(frame_index=int(row.frame_index),
                task_index=int(row.task_index), timestamp=float(row.timestamp),
                **{"observation.state": np.asarray(part["observation.state"].iloc[index]).tolist(),
                   "action": np.asarray(part["action"].iloc[index]).tolist()}) for index, row in enumerate(part.itertuples())]))
        offset += len(part)
    return dict(tool_version=VERSION, note="Selected original continuous samples; only new terminal actions become holds.", episodes=output)


def make_plan(args):
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    if args.idle_filter:
        config["idle"]["enabled"] = True
    reader, raw_document = None, None
    if args.root:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
        reader = V3EpisodeReader(args.root)
        chosen = reader.episode_ids if args.episodes is None else list(dict.fromkeys(args.episodes))
        if set(chosen) - set(reader.episode_ids):
            raise ValueError("Unknown selected episode IDs")
        inventory = read_episodes(reader, chosen, "timestamp")
        source = dict(kind="lerobot_v3", root=str(reader.root), info_sha256=sha_file(reader.root / "meta/info.json"), fps=reader.fps)
    else:
        raw_document, items = load_raw(args.raw_json)
        chosen = [episode for episode, _ in items] if args.episodes is None else list(dict.fromkeys(args.episodes))
        if set(chosen) - {episode for episode, _ in items}:
            raise ValueError("Unknown selected raw episode IDs")
        inventory = [(episode, frame) for episode, frame in items if episode in chosen]
        source = dict(kind="raw_json", path=str(args.raw_json.resolve()), sha256=sha_file(args.raw_json),
                      fps=raw_document.get("nominal_metadata_fps"))
    records, events, flags, saved_frames = [], [], [], {}
    for number, (episode, frame) in enumerate(inventory, 1):
        record, current_events, reasons = analyze(episode, frame, config)
        if reader:
            record["metadata_sha256"] = hashlib.sha256(json.dumps(reader.episode(episode), sort_keys=True, default=str).encode()).hexdigest()
        records.append(record)
        events.extend(current_events)
        for index, why in enumerate(reasons):
            if why:
                flags.append(dict(episode=episode, task_index=record["task_index"], frame_index=index, reasons=";".join(sorted(why))))
        if args.raw_json:
            saved_frames[episode] = frame
        if number % 100 == 0 or number == len(chosen):
            print(f"[curation] checked {number}/{len(chosen)} episodes", file=sys.stderr, flush=True)
    records.sort(key=lambda record: record["episode"])
    segments = [segment for record in records for segment in record["retained_segments"]]
    document = dict(version=VERSION, paper="arXiv:2611.06817v1 section 2.2, pages 6-7", source=source,
        script_sha256=sha_file(Path(__file__)), config=config, episodes=records, segments=segments,
        notes=["Smoother/thresholds are local adaptations; the paper does not disclose their numeric values.",
               "All supporting frames of a threshold violation are excluded; original poses are never replaced by the fit.",
               "Exact repeated states are reported only, not an unconditional deletion criterion.",
               "Both state/action and both arms are checked; binary gripper changes are not differentiated.",
               "Camera-pose/hand-visibility checks are unavailable without calibrated camera and visibility data.",
               "32-frame/70% position-only idle filtering is optional; it may reject legitimate robot holds or pure rotation.",
               "Timestamp is treated as physical time; nominal encoded timestamps cannot verify capture timing.",
               "New terminal actions use the final retained state as a hold, only after exact next-state alignment verification.",
               "Filtering can remove legitimate contacts or incomplete task stages; retained segments require review.",
               "Old code caches, frame/episode maps and normalization statistics must not be copied to new indices."])
    per_task = defaultdict(lambda: dict(source_episodes=0, source_frames=0, stability_rejected_frames=0, retained_segments=0, retained_frames=0))
    for record in records:
        row = per_task[record["task_index"]]
        row["source_episodes"] += 1
        row["source_frames"] += record["frames"]
        row["stability_rejected_frames"] += len(record["bad_frames"])
        row["retained_segments"] += len(record["retained_segments"])
        row["retained_frames"] += sum(item["frames"] for item in record["retained_segments"])
    summary = dict(version=VERSION, source_episodes=len(records), source_frames=sum(r["frames"] for r in records),
        stability_rejected_frames=len(flags), retained_segments=len(segments), retained_frames=sum(s["frames"] for s in segments),
        other_dropped_frames=sum(r["frames"] - len(r["bad_frames"]) - sum(s["frames"] for s in r["retained_segments"]) for r in records),
        idle_filter=config["idle"]["enabled"], by_task=dict(sorted(per_task.items())),
        events_by_metric=dict(Counter(event["metric"] for event in events)),
        next_state_aligned_episodes=sum(r["next_state_action_exact"] for r in records),
        episodes_with_missing_gripper_transitions=[r["episode"] for r in records
            if any(item["missing_transition_frames"] for item in r["gripper_transition_coverage"].values())],
        thresholds_are="Explicit local examples, not author settings or robot hardware limits.")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "segments.json", document)
    write_json(args.out / "summary.json", summary)
    for filename, rows, fields in (("frame_flags.csv", flags, ["episode", "task_index", "frame_index", "reasons"]),
            ("events.csv", events, ["episode", "task_index", "source", "side", "metric", "value", "threshold", "unit", "support_start", "support_end_exclusive"])):
        with (args.out / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if args.raw_json:
        if all(record["next_state_action_exact"] for record in records):
            write_json(args.out / "curated_raw_episodes.json", preview_raw(document, saved_frames))
        else:
            print("[curation] raw materialization skipped: next-state action convention not verified", file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def relative_payload(path, root):
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Payload escapes source root: {path}")
    return path.relative_to(root)


def export_dataset(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
    document = json.loads(args.plan.read_text(encoding="utf-8"))
    if document.get("version") != VERSION or document["source"]["kind"] != "lerobot_v3":
        raise ValueError("Export requires a v1 plan generated from this LeRobot v3 dataset, not raw JSON")
    records = validate_segments(document)
    if not document["segments"]:
        raise ValueError("Plan retains no segments; review thresholds before exporting")
    reader = V3EpisodeReader(args.root)
    if document["source"]["info_sha256"] != sha_file(reader.root / "meta/info.json"):
        raise ValueError("Source info changed since plan generation")
    # Validate all source columns and video references before creating the output.
    for episode, frame in read_episodes(reader, list(records), "timestamp"):
        record = records[episode]
        if frame_hash(frame) != record["source_sha256"]:
            raise ValueError(f"Episode {episode}: source values changed since plan generation")
        meta_hash = hashlib.sha256(json.dumps(reader.episode(episode), sort_keys=True, default=str).encode()).hexdigest()
        if meta_hash != record["metadata_sha256"]:
            raise ValueError(f"Episode {episode}: source video/index metadata changed")
        if record["retained_segments"] and not record["next_state_action_exact"]:
            raise ValueError(f"Episode {episode}: exact next-state action convention is required for terminal holds")
        for key in reader.video_keys:
            relative_payload(reader.video_file(episode, key), reader.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{args.out.name}.partial-", dir=args.out.parent))
    staging_args = argparse.Namespace(**vars(args))
    staging_args.out = stage
    try:
        report = materialize_dataset(staging_args, document, records, reader, args.out)
        stage.rename(args.out)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


def materialize_dataset(args, document, records, reader, published_out):
    import pandas as pd
    from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
    info = json.loads(json.dumps(reader.info))
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    metadata, mapping, offset, terminal_changes = [], [], 0, 0
    grouped = defaultdict(list)
    for new_episode, segment in enumerate(document["segments"]):
        grouped[reader.data_file(segment["source_episode"])].append((new_episode, segment))
    # Source global order follows sorted source episodes; each source file is read once.
    ordered_parts = {}
    for path, selected in grouped.items():
        ids = sorted({segment["source_episode"] for _, segment in selected})
        table = pd.read_parquet(path, filters=[("episode_index", "in", ids)])
        frames = {int(episode): data.sort_values("frame_index").reset_index(drop=True)
                  for episode, data in table.groupby("episode_index", sort=False)}
        for new_episode, segment in selected:
            episode = segment["source_episode"]
            frame = frames[episode]
            # Full-column read is rechecked too; do not trust a stale packed Parquet cache.
            if frame_hash(frame) != records[episode]["source_sha256"]:
                raise ValueError(f"Episode {episode}: source changed during export")
            ordered_parts[new_episode] = (segment, frame.iloc[segment["start_frame"]:segment["end_frame"]].copy())
    for new_episode in range(len(document["segments"])):
        segment, original_part = ordered_parts.pop(new_episode)
        episode, start, stop = segment["source_episode"], segment["start_frame"], segment["end_frame"]
        part, changed = slice_frame(original_part, 0, len(original_part), new_episode, offset)
        chunk, file_index = divmod(new_episode, 1000)
        path = args.out / info["data_path"].format(chunk_index=chunk, file_index=file_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False)
        meta = dict(episode_index=new_episode, length=len(part), tasks=[reader.tasks[segment["task_index"]]],
            **{"data/chunk_index": chunk, "data/file_index": file_index,
               "dataset_from_index": offset, "dataset_to_index": offset + len(part)})
        for key in reader.video_keys:
            old_meta = reader.episode(episode)
            base = f"videos/{key}"
            meta[f"{base}/chunk_index"] = int(old_meta[f"{base}/chunk_index"])
            meta[f"{base}/file_index"] = int(old_meta[f"{base}/file_index"])
            meta[f"{base}/from_timestamp"] = reader.video_start(episode, key) + start / reader.fps
            meta[f"{base}/to_timestamp"] = reader.video_start(episode, key) + stop / reader.fps
            source_path = reader.video_file(episode, key)
            target_path = args.out / relative_payload(source_path, reader.root)
            if not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if args.video_mode == "copy":
                    shutil.copy2(source_path, target_path)
                else:
                    target_path.symlink_to(source_path.resolve())
        metadata.append(meta)
        mapping.append(dict(new_episode=new_episode, source_episode=episode, source_start_frame=start, source_end_frame=stop,
            new_global_offset=offset, frames=len(part), terminal_action_changed=changed))
        terminal_changes += changed
        offset += len(part)
    meta_path = args.out / "meta/episodes/chunk-000/file-000.parquet"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata).to_parquet(meta_path, index=False)
    for name in ("tasks.parquet", "tasks.jsonl"):
        source_path = reader.root / "meta" / name
        if source_path.exists():
            shutil.copy2(source_path, args.out / "meta" / name)
    info.update(total_episodes=len(metadata), total_frames=offset,
        total_videos=len({str(p) for p in (args.out / "videos").rglob("*.mp4")}) if reader.video_keys else 0,
        total_chunks=(len(metadata) + 999) // 1000, chunks_size=1000, splits={"train": f"0:{len(metadata)}"})
    info.pop("total_tasks", None)
    info["total_tasks"] = len(reader.tasks)
    info["trajectory_curation"] = dict(version=VERSION, plan_sha256=sha_file(args.plan),
        video_mode=args.video_mode, source_root=str(reader.root), normalization_stats="must_recompute", code_cache="must_regenerate")
    write_json(args.out / "meta/info.json", info)
    write_json(args.out / "curation_source_map.json", mapping)
    # No old normalization/code/instruction caches are copied across new indices.
    checked = V3EpisodeReader(args.out)
    if checked.total_frames != offset or checked.episode_ids != list(range(len(mapping))):
        raise ValueError("Exported dataset reader verification failed")
    for item in mapping:
        frame = checked.read_episode(item["new_episode"])
        state, action, _, _ = arrays(frame)
        target = np.concatenate((state[1:, 14:34], state[-1:, 14:34]), axis=0)
        if not np.array_equal(action, target):
            raise ValueError("Exported next-state action verification failed")
    report = dict(version=VERSION, episodes=len(metadata), frames=offset, terminal_actions_changed=terminal_changes,
        video_mode=args.video_mode, output=str(published_out),
        source_untouched=True, required_before_training=["Recompute relative-action normalization statistics on this new root.",
        "Regenerate code and every frame/episode-indexed cache for the new enumeration.",
        "Verify video alignment, retained contact stages, task coverage and train/validation grouping."])
    write_json(args.out / "curation_export_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Read-only stability analysis and proposed continuous segments")
    source = plan.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path)
    source.add_argument("--raw-json", type=Path)
    plan.add_argument("--episodes", type=int, nargs="+")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--idle-filter", action="store_true", help="Enable optional position-only 32-frame / 70%% idle filtering")
    export = commands.add_parser("export", help="Materialize reviewed segments as an independent v3 dataset")
    export.add_argument("--root", type=Path, required=True)
    export.add_argument("--plan", type=Path, required=True, help="segments.json from plan --root")
    export.add_argument("--video-mode", choices=("symlink", "copy"), default="symlink")
    for command in (plan, export):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    if args.out.exists():
        parser.error("Output exists; choose a new --out directory")
    if args.root:
        args.root = args.root.expanduser().resolve()
        if args.out == args.root or args.out.is_relative_to(args.root):
            parser.error("Output must be outside the original dataset root")
    try:
        (make_plan if args.command == "plan" else export_dataset)(args)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
