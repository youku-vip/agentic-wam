import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/curate_custom_trajectories.py"
spec = importlib.util.spec_from_file_location("paper_curation", SCRIPT)
curation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curation)
sys.path.insert(0, str(REPO / "src"))


def config(**thresholds):
    value = json.loads(curation.DEFAULT_CONFIG.read_text())
    value["idle"]["enabled"] = False
    if thresholds:
        value["thresholds"] = {key: thresholds.get(key) for key in curation.METRICS}
        value["anomaly_rule"] = "any"
    return value


def frame(n=80, spike=None, yaw=None):
    state = np.zeros((n, 34), dtype=np.float32)
    for start in (17, 26):
        state[:, start:start + 6] = [1, 0, 0, 0, 1, 0]
    state[:, 14] = np.arange(n) * .0005
    state[:, 32:34] = 1
    state[20:60, 32] = 0
    if spike is not None:
        state[spike, 14] += .2
    if yaw is not None:
        r = np.deg2rad(np.asarray(yaw))
        state[:, 17:23] = np.stack((np.cos(r), np.sin(r), np.zeros(n), -np.sin(r), np.cos(r), np.zeros(n)), axis=1)
    action = np.r_[state[1:, 14:34], state[-1:, 14:34]].copy()
    return pd.DataFrame({"index": np.arange(n), "episode_index": np.full(n, 5), "frame_index": np.arange(n),
        "task_index": np.full(n, 2), "timestamp": np.arange(n, dtype=np.float32) / 10,
        "observation.state": list(state), "action": list(action)})


def cli(*args, success=True):
    result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, cwd=REPO)
    if success and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


def source_dataset(root):
    import av
    for directory in ("meta/episodes/chunk-000", "data/chunk-000", "videos/observation.images.head/chunk-000"):
        (root / directory).mkdir(parents=True)
    camera_starts = {"head": 1., "left": .8, "right": 1.2}
    info = dict(codebase_version="v3.0", fps=10, total_episodes=2, total_frames=160, total_tasks=2,
        features={"observation.state": dict(dtype="float32", shape=[34]), "action": dict(dtype="float32", shape=[20]),
                  **{f"observation.images.{camera}": dict(dtype="video", shape=[48, 64, 3]) for camera in camera_starts}},
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    (root / "meta/info.json").write_text(json.dumps(info))
    frames, metadata = [], []
    for slot, episode in enumerate((5, 9)):
        current = frame(spike=40 if slot == 0 else None)
        current["episode_index"] = episode
        current["task_index"] = 2 if slot == 0 else 0
        current["index"] += slot * 80
        frames.append(current)
        item = dict(episode_index=episode, length=80, **{
            "data/chunk_index": 0, "data/file_index": 0, "dataset_from_index": slot * 80,
            "dataset_to_index": (slot + 1) * 80})
        for camera, begin in camera_starts.items():
            prefix = f"videos/observation.images.{camera}"
            item.update({f"{prefix}/chunk_index": 0, f"{prefix}/file_index": 0,
                f"{prefix}/from_timestamp": begin + slot * 8., f"{prefix}/to_timestamp": begin + 8 + slot * 8.})
        metadata.append(item)
    pd.concat(frames).iloc[::-1].to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(metadata).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0, 2]}, index=pd.Index(["open air fryer", "turn knob"])).to_parquet(root / "meta/tasks.parquet")
    with av.open(str(root / "videos/observation.images.head/chunk-000/file-000.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(175):
            image = np.full((48, 64, 3), (index * 7) % 200 + 20, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video = root / "videos/observation.images.head/chunk-000/file-000.mp4"
    for camera in ("left", "right"):
        target = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        target.parent.mkdir(parents=True)
        target.write_bytes(video.read_bytes())
    return root


class NumericCurationTests(unittest.TestCase):
    def test_clean_motion_and_binary_gripper_transitions_stay_unchanged(self):
        original = frame()
        before = curation.frame_hash(original)
        result, events, _ = curation.analyze(5, original, config())
        self.assertEqual(result["bad_frames"], [])
        self.assertEqual(events, [])
        self.assertEqual(result["retained_segments"][0]["frames"], 80)
        self.assertTrue(result["next_state_action_exact"])
        self.assertEqual(curation.frame_hash(original), before)

    def test_local_polynomial_handles_nonuniform_time_and_edges(self):
        time = np.array([0, .04, .13, .25, .39, .62, .85, 1.1, 1.5])
        positions = np.column_stack((time**2, .5*time, np.ones(len(time))))
        smooth = curation.smooth_positions(positions, time, np.ones(len(time)-1, dtype=bool))
        np.testing.assert_allclose(smooth, positions, atol=1e-13)

    def test_uniform_fast_stencil_matches_time_aware_quadratic(self):
        time = np.arange(30) / 10.
        positions = np.column_stack((time**2, .5*time, time**2 + time))
        smooth = curation.smooth_positions(positions, time, np.ones(29, dtype=bool))
        np.testing.assert_allclose(smooth, positions, atol=1e-13)

    def test_spike_excludes_full_derivative_support_then_splits(self):
        result, events, _ = curation.analyze(5, frame(spike=40), config(jerk=150))
        support = {i for event in events for i in range(event["support_start"], event["support_end_exclusive"])}
        self.assertEqual(set(result["bad_frames"]), support)
        self.assertIn(40, support)
        self.assertEqual(len(result["retained_segments"]), 2)
        for segment in result["retained_segments"]:
            self.assertFalse(set(range(segment["start_frame"], segment["end_frame"])) & support)

    def test_large_orientation_changes_do_not_change_translation_selection(self):
        original = frame(n=40, yaw=[179, -90] * 20)
        result, events, _ = curation.analyze(5, original, config())
        self.assertEqual(result["bad_frames"], [])
        self.assertEqual(events, [])

    def test_exact_repeats_do_not_trigger_automatic_deletion(self):
        original = frame()
        state = np.stack(original["observation.state"])
        state[:, 14] = 0
        state[:, 0] = np.arange(80) % 2
        original["observation.state"] = list(state)
        original["action"] = list(np.r_[state[1:, 14:34], state[-1:, 14:34]])
        result, _, _ = curation.analyze(5, original, config())
        self.assertEqual(result["bad_frames"], [])
        self.assertEqual(len(result["retained_segments"]), 1)

    def test_invalid_time_is_cut_but_rotation_is_not_a_selection_criterion(self):
        original = frame()
        original.loc[40, "timestamp"] = original.loc[39, "timestamp"]
        rotation = original["observation.state"].iloc[50].copy()
        rotation[17:23] = 0
        original.at[50, "observation.state"] = rotation
        result, _, _ = curation.analyze(5, original, config())
        self.assertTrue({39, 40} <= set(result["bad_frames"]))
        self.assertNotIn(50, result["bad_frames"])

    def test_idle_is_optional_and_can_reject_pure_rotation(self):
        original = frame(yaw=np.linspace(0, 90, 80))
        state = np.stack(original["observation.state"])
        state[:, 14] = 0
        original["observation.state"] = list(state)
        original["action"] = list(np.r_[state[1:, 14:34], state[-1:, 14:34]])
        disabled = curation.analyze(5, original, config())[0]
        self.assertEqual(len(disabled["retained_segments"]), 1)
        enabled_config = config()
        enabled_config["idle"]["enabled"] = True
        enabled_config["idle"]["skip_task_indices"] = []
        enabled = curation.analyze(5, original, enabled_config)[0]
        self.assertEqual(enabled["retained_segments"], [])
        self.assertEqual(enabled["dropped_segments"][0]["reason"], "idle_all_windows")

    def test_default_idle_policy_preserves_configured_knob_task(self):
        original = frame(yaw=np.linspace(0, 90, 80))
        state = np.stack(original["observation.state"])
        state[:, 14] = 0
        original["observation.state"] = list(state)
        original["action"] = list(np.r_[state[1:, 14:34], state[-1:, 14:34]])
        settings = json.loads(curation.DEFAULT_CONFIG.read_text())
        result, _, _ = curation.analyze(5, original, settings)
        self.assertEqual(result["retained_segments"][0]["frames"], 80)
        original["task_index"] = 0
        result, _, _ = curation.analyze(5, original, settings)
        self.assertEqual(result["retained_segments"], [])

    def test_edge_only_idle_keeps_interior_pauses_below_70_percent_activity(self):
        state = np.zeros((300, 34))
        state[50:71, 14] = np.arange(21) * .001
        state[71:220, 14] = .02
        state[220:241, 14] = .02 + np.arange(21) * .001
        state[241:, 14] = .04
        settings = config()["idle"]
        settings["enabled"] = True
        start, stop, fraction, why = curation.trim_idle(state, 0, 300, settings)
        self.assertIsNone(why)
        self.assertGreater(start, 0)
        self.assertLess(stop, 300)
        self.assertLess(start, 100)
        self.assertGreater(stop, 200)
        self.assertLess(fraction, .7)
        settings["mode"] = "paper"
        self.assertEqual(curation.trim_idle(state, 0, 300, settings)[3], "idle_active_fraction")

    def test_a_single_dynamic_peak_without_residual_is_only_a_review_event(self):
        original = frame()
        state = np.stack(original["observation.state"])
        state[:, 14] = np.minimum(np.arange(80), 30) * .005
        original["observation.state"] = list(state)
        original["action"] = list(np.r_[state[1:, 14:34], state[-1:, 14:34]])
        settings = config()
        settings["thresholds"]["jerk"] = 1.
        result, events, _ = curation.analyze(5, original, settings)
        self.assertEqual(result["bad_frames"], [])
        self.assertTrue(any(event["metric"] == "jerk" for event in events))
        self.assertTrue(all(not event["used_for_exclusion"] for event in events))

    def test_default_candidates_require_both_residual_and_dynamic_support(self):
        original = frame(spike=40)
        result, events, _ = curation.analyze(5, original, config())
        residual = {event["support_start"] for event in events if event["metric"] == "position_residual"}
        dynamics = {index for event in events if event["metric"] != "position_residual"
                    for index in range(event["support_start"], event["support_end_exclusive"])}
        self.assertEqual(set(result["bad_frames"]), residual & dynamics)
        self.assertIn(40, result["bad_frames"])
        self.assertEqual({event["source"] for event in events}, {"state"})

    def test_idle_uses_either_arm_and_trims_only_continuous_ends(self):
        state = np.zeros((100, 34))
        state[40:70, 23] = np.arange(30) * .01
        state[70:, 23] = .29
        settings = config()["idle"]
        settings["enabled"] = True
        start, stop, fraction, why = curation.trim_idle(state, 0, 100, settings)
        self.assertLess(start, 40)
        self.assertGreater(stop, 70)
        self.assertEqual(fraction, 1)
        self.assertIsNone(why)

    def test_preview_preserves_all_poses_and_interior_actions(self):
        original = frame(spike=40)
        result, _, _ = curation.analyze(5, original, config())
        plan = dict(episodes=[result], segments=result["retained_segments"])
        output = curation.preview_raw(plan, {5: original})
        for segment in output["episodes"]:
            lo, hi = segment["source_start_frame"], segment["source_end_frame"]
            rows = segment["frames"]
            state = np.asarray([row["observation.state"] for row in rows])
            action = np.asarray([row["action"] for row in rows])
            np.testing.assert_array_equal(state, np.stack(original["observation.state"])[lo:hi])
            np.testing.assert_array_equal(action[:-1], np.stack(original["action"])[lo:hi-1])
            np.testing.assert_array_equal(action, np.r_[state[1:, 14:34], state[-1:, 14:34]])

    def test_report_identifies_a_release_lost_at_a_cut(self):
        original = frame(spike=40)
        state = np.stack(original["observation.state"])
        state[43:, 32] = 1
        original["observation.state"] = list(state)
        original["action"] = list(np.r_[state[1:, 14:34], state[-1:, 14:34]])
        record, _, _ = curation.analyze(5, original, config(jerk=150))
        coverage = record["gripper_transition_coverage"]["left"]
        self.assertEqual(coverage["original_transitions"], 2)
        self.assertEqual(coverage["retained_transitions"], 1)
        self.assertEqual(coverage["missing_transition_frames"], [43])
        gentle, _, _ = curation.analyze(5, original, config())
        self.assertLess(len(gentle["bad_frames"]), len(record["bad_frames"]))
        self.assertEqual(gentle["gripper_transition_coverage"]["left"]["missing_transition_frames"], [43])

    def test_invalid_config_and_mixed_tasks_fail(self):
        invalid = config()
        invalid["thresholds"]["jerk"] = -1
        with self.assertRaises(ValueError):
            curation.validate_config(invalid)
        mixed = frame()
        mixed.loc[40, "task_index"] = 3
        with self.assertRaises(ValueError):
            curation.analyze(5, mixed, config())


class ExportTests(unittest.TestCase):
    def test_failed_export_does_not_publish_a_partial_dataset(self):
        import argparse
        with TemporaryDirectory() as directory:
            base = Path(directory)
            root = source_dataset(base / "source")
            plan, out = base / "plan", base / "clean"
            cli("plan", "--root", root, "--out", plan)
            args = argparse.Namespace(root=root, plan=plan / "segments.json", out=out, video_mode="symlink")
            with patch.object(curation, "materialize_dataset", side_effect=RuntimeError("test write failure")):
                with self.assertRaisesRegex(RuntimeError, "test write failure"):
                    curation.export_dataset(args)
            self.assertFalse(out.exists())
            self.assertEqual(list(base.glob(".clean.partial-*")), [])

    def test_v3_split_reindex_video_alignment_and_source_immutability(self):
        from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
        from agenticwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
        from agenticwam.datasets.lerobot.v3.dataset import V3MultiLeRobotDataset
        with TemporaryDirectory() as directory:
            base = Path(directory)
            root = source_dataset(base / "source")
            before = {p.relative_to(root): curation.sha_file(p) for p in root.rglob("*") if p.is_file()}
            plan, out = base / "plan", base / "clean"
            cli("plan", "--root", root, "--out", plan)
            document = json.loads((plan / "segments.json").read_text())
            self.assertEqual(len(document["segments"]), 3)
            cli("export", "--root", root, "--plan", plan / "segments.json", "--out", out)
            original, exported = V3EpisodeReader(root), V3EpisodeReader(out)
            mapping = json.loads((out / "curation_source_map.json").read_text())
            self.assertEqual(exported.episode_ids, [0, 1, 2])
            self.assertFalse((out / "dataset_stats_eef64.json").exists())
            self.assertFalse((out / "coding_v4").exists())
            dataset = V3MultiLeRobotDataset([str(out)], delta_timestamps={
                "observation.state": [0, .1, .2], "action": [0, .1],
                **{key: [0, .1] for key in original.video_keys}}, video_backend="pyav")
            for item in mapping:
                frame_out = exported.read_episode(item["new_episode"])
                frame_in = original.read_episode(item["source_episode"])
                lo, hi = item["source_start_frame"], item["source_end_frame"]
                np.testing.assert_array_equal(np.stack(frame_out["observation.state"]), np.stack(frame_in["observation.state"])[lo:hi])
                np.testing.assert_array_equal(np.stack(frame_out["action"])[:-1], np.stack(frame_in["action"])[lo:hi-1])
                self.assertEqual(int(frame_out.task_index.iloc[0]), int(frame_in.task_index.iloc[0]))
                self.assertEqual(list(frame_out["index"]), list(range(item["new_global_offset"], item["new_global_offset"] + len(frame_out))))
                for index in (0, len(frame_out) - 1):
                    sample = dataset[item["new_global_offset"] + index]
                    for key in original.video_keys:
                        old_time = original.video_timestamps(item["source_episode"], key, [lo + index])
                        new_time = exported.video_timestamps(item["new_episode"], key, [index])
                        np.testing.assert_allclose(old_time, new_time, atol=1e-12)
                        old_image = decode_video_frames(original.video_file(item["source_episode"], key), old_time, .099, "pyav")
                        np.testing.assert_array_equal(sample[key][0].numpy(), old_image[0].numpy())
                    if index == len(frame_out) - 1:
                        self.assertTrue(sample["action_is_pad"][1])
                        self.assertTrue(all(sample[key + "_is_pad"][1] for key in original.video_keys))
            self.assertEqual(before, {p.relative_to(root): curation.sha_file(p) for p in root.rglob("*") if p.is_file()})
            self.assertNotEqual(cli("export", "--root", root, "--plan", plan / "segments.json", "--out", out, success=False).returncode, 0)

    def test_export_refuses_stale_source_before_creating_output(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            root = source_dataset(base / "source")
            plan, out = base / "plan", base / "clean"
            cli("plan", "--root", root, "--out", plan)
            path = root / "data/chunk-000/file-000.parquet"
            table = pd.read_parquet(path)
            row = table["observation.state"].iloc[0].copy()
            row[14] += .01
            table.at[0, "observation.state"] = row
            table.to_parquet(path, index=False)
            result = cli("export", "--root", root, "--plan", plan / "segments.json", "--out", out, success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source values changed", result.stderr)
            self.assertFalse(out.exists())

    def test_export_requires_next_state_convention_and_output_outside_source(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            root = source_dataset(base / "source")
            bad_output = cli("plan", "--root", root, "--out", root / "nested", success=False)
            self.assertNotEqual(bad_output.returncode, 0)
            path = root / "data/chunk-000/file-000.parquet"
            table = pd.read_parquet(path)
            action = table["action"].iloc[0].copy()
            action[0] += .001
            table.at[0, "action"] = action
            table.to_parquet(path, index=False)
            plan, out = base / "plan", base / "clean"
            cli("plan", "--root", root, "--out", plan)
            result = cli("export", "--root", root, "--plan", plan / "segments.json", "--out", out, success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("next-state action convention", result.stderr)
            self.assertFalse(out.exists())

    def test_plan_rejects_overlapping_ranges_and_bad_supports(self):
        original = frame(spike=40)
        record, _, _ = curation.analyze(5, original, config())
        document = dict(episodes=[record], segments=copy.deepcopy(record["retained_segments"]))
        document["segments"][1]["start_frame"] = 0
        with self.assertRaises(ValueError):
            curation.validate_segments(document)
        document["segments"] = [dict(source_episode=5, task_index=2, start_frame=0, end_frame=80)]
        with self.assertRaises(ValueError):
            curation.validate_segments(document)


if __name__ == "__main__":
    unittest.main()
