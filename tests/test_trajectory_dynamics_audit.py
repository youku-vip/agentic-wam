import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_custom_trajectories.py"
spec = importlib.util.spec_from_file_location("trajectory_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def arrays(times, x=None, yaw_degrees=None):
    n = len(times)
    state, action = np.zeros((n, 34)), np.zeros((n, 20))
    yaw = np.deg2rad(np.zeros(n) if yaw_degrees is None else yaw_degrees)
    rotation = np.stack((np.cos(yaw), np.sin(yaw), np.zeros(n), -np.sin(yaw), np.cos(yaw), np.zeros(n)), axis=1)
    for values, offsets in ((state, (14, 23)), (action, (0, 9))):
        for offset in offsets:
            values[:, offset + 3:offset + 9] = rotation
        if x is not None:
            values[:, offsets[0]] = x
    return state, action


def analyze(times, x=None, yaw=None, **kwargs):
    state, action = arrays(times, x, yaw)
    return audit.analyze_episode(0, state, action, times, np.zeros(len(times), dtype=int), **kwargs)


def left_state(tracks):
    return next(t for t in tracks if t["source"] == "state" and t["side"] == "left")


def fixture(root, moving=True, timestamp=True):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    n = 7
    info = dict(codebase_version="v3.0", fps=10, total_episodes=2, total_frames=2*n,
                features={"observation.state": {"dtype": "float32", "shape": [34]},
                          "action": {"dtype": "float32", "shape": [20]}},
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    (root / "meta/info.json").write_text(json.dumps(info))
    rows, episodes = [], []
    for ep in range(2):
        times = np.arange(n) * .07  # intentionally not the video's nominal 10 fps
        state, action = arrays(times, .1 * times if moving else None)
        state[n//2:, 32] = 1
        action[n//2:, 18] = 1
        for i in range(n):
            record = dict(episode_index=ep, frame_index=i, task_index=ep,
                          action=action[i], **{"observation.state": state[i]})
            if timestamp:
                record["timestamp"] = float(times[i])
            rows.append(record)
        episodes.append(dict(episode_index=ep, length=n, **{"data/chunk_index": 0, "data/file_index": 0,
                        "dataset_from_index": ep*n, "dataset_to_index": (ep+1)*n}))
    # Both episodes share one Parquet, with an adversarial physical row order.
    pd.DataFrame(rows[::-1]).to_parquet(root / "data/chunk-000/file-000.parquet")
    pd.DataFrame(episodes).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["task zero", "task one"])).to_parquet(root / "meta/tasks.parquet")
    return root


class DynamicsTests(unittest.TestCase):
    def test_constant_velocity_on_nonuniform_timestamps(self):
        t = np.array([0, .04, .13, .21, .37, .5])
        tracks, _ = analyze(t, x=.2*t)
        scores = left_state(tracks)["scores"]
        np.testing.assert_allclose(scores["speed"][1:], .2, atol=1e-12)
        np.testing.assert_allclose(scores["acceleration"][2:], 0, atol=1e-12)
        np.testing.assert_allclose(scores["jerk"][3:], 0, atol=1e-10)

    def test_acceleration_and_cubic_jerk_use_nonuniform_divided_differences(self):
        t = np.array([0, .05, .12, .23, .41, .55, .7])
        quadratic, _ = analyze(t, x=t**2)
        np.testing.assert_allclose(left_state(quadratic)["scores"]["acceleration"][2:], 2, atol=1e-10)
        cubic, _ = analyze(t, x=t**3)
        np.testing.assert_allclose(left_state(cubic)["scores"]["jerk"][3:], 6, atol=1e-10)

    def test_direction_reversal_is_detected_even_when_speed_norm_is_constant(self):
        tracks, _ = analyze(np.array([0, .1, .2]), x=np.array([0, .1, 0]))
        scores = left_state(tracks)["scores"]
        np.testing.assert_allclose(scores["speed"][1:], [1, 1])
        self.assertAlmostEqual(scores["velocity_change"][2], 2)
        self.assertAlmostEqual(scores["acceleration"][2], 20)

    def test_euler_wrap_is_two_degree_rotation_not_358_degree_jump(self):
        tracks, _ = analyze(np.array([0, .1, .2]), yaw=np.array([179, -179, -177]))
        scores = left_state(tracks)["scores"]
        np.testing.assert_allclose(scores["rotation_step"][1:], 2, atol=1e-10)
        np.testing.assert_allclose(scores["angular_speed"][1:], 20, atol=1e-10)
        np.testing.assert_allclose(scores["angular_acceleration"][2:], 0, atol=1e-9)

    def test_cubic_rotation_recovers_constant_angular_jerk_on_nonuniform_times(self):
        t = np.array([0, .05, .12, .23, .41, .55, .7])
        tracks, _ = analyze(t, yaw=10*t**3)
        np.testing.assert_allclose(left_state(tracks)["scores"]["angular_jerk"][3:], 60, atol=1e-8)

    def test_rotation_log_is_stable_near_pi_for_arbitrary_axes(self):
        def exponential(vectors):
            angle = np.linalg.norm(vectors, axis=1)
            axis = np.divide(vectors, angle[:, None], out=np.zeros_like(vectors), where=angle[:, None] > 1e-12)
            skew = np.zeros((len(vectors), 3, 3))
            skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
            skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
            skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
            return np.eye(3) + np.sin(angle)[:, None, None] * skew + (1-np.cos(angle))[:, None, None] * (skew @ skew)
        generator = np.random.default_rng(31)
        axes = generator.normal(size=(40, 3))
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        angles = np.linspace(0, np.pi, len(axes))
        matrices = exponential(axes*angles[:, None])
        recovered = audit.rotation_vectors(matrices)
        np.testing.assert_allclose(exponential(recovered), matrices, atol=1e-12)

    def test_duplicate_time_and_long_gap_do_not_create_huge_derivatives(self):
        t = np.array([0, .1, .1, .2, .3, 1.4, 1.5, 1.6])
        tracks, quality = analyze(t, x=t)
        scores = left_state(tracks)["scores"]
        self.assertEqual(quality["nonpositive_or_tiny_dt_end_frames"], [2])
        self.assertEqual(quality["long_gap_end_frames"], [5])
        self.assertTrue(np.isnan(scores["speed"][[2, 5]]).all())
        self.assertTrue(np.isnan(scores["jerk"][2:8]).all())
        np.testing.assert_allclose(scores["speed"][[1, 3, 4, 6, 7]], 1)

    def test_invalid_rotation_does_not_contaminate_position_or_other_arm(self):
        t = np.arange(7)*.1
        state, action = arrays(t, x=.1*t)
        state[3, 17:23] = 0
        tracks, _ = audit.analyze_episode(0, state, action, t, np.zeros(7))
        current = left_state(tracks)
        self.assertEqual(current["quality"]["invalid_rotation6d_frames"], [3])
        self.assertTrue(np.isnan(current["scores"]["angular_speed"][[3, 4]]).all())
        np.testing.assert_allclose(current["scores"]["speed"][1:], .1)

    def test_gripper_binary_transition_is_context_not_motion_anomaly(self):
        t = np.arange(7)*.1
        state, action = arrays(t)
        state[3:, 32] = 1
        action[3:, 18] = 1
        tracks, _ = audit.analyze_episode(0, state, action, t, np.zeros(7))
        self.assertEqual(left_state(tracks)["grip_changed"].sum(), 1)
        _, _, events = audit.summarize(tracks)
        self.assertEqual(events, [])

    def test_joint_wrap_is_reported_without_silently_unwrapping_encoder_values(self):
        t = np.arange(5)*.1
        state, action = arrays(t)
        state[:, 0] = [179, -179, -178, -177, -176]
        tracks, _ = audit.analyze_episode(0, state, action, t, np.zeros(5), joint_unit="deg")
        joints = next(t for t in tracks if t["source"] == "state_joints" and t["side"] == "left")
        self.assertEqual(joints["scores"]["joint_step"][1], 358)
        self.assertEqual(joints["units"]["joint_speed"], "deg/s")

    def test_only_command_spike_exceeds_limit_when_measured_state_is_smooth(self):
        t = np.arange(10)*.1
        state, action = arrays(t, x=.1*t)
        action[4, 0] += .1
        tracks, _ = audit.analyze_episode(0, state, action, t, np.zeros(10))
        _, _, events = audit.summarize(tracks, {"speed": .5})
        violations = [event for event in events if event["limit_exceeded"]]
        self.assertTrue(violations)
        self.assertTrue(all(event["source"] == "action" for event in violations))
        self.assertEqual({event["frame_end"] for event in violations}, {4, 5})

    def test_short_episodes_and_episode_boundaries_are_not_differenced(self):
        all_tracks = []
        for episode, value in ((0, 0), (1, 100)):
            t = np.array([0.0])
            state, action = arrays(t, x=np.array([value]))
            tracks, _ = audit.analyze_episode(episode, state, action, t, np.zeros(1))
            all_tracks.extend(tracks)
        _, rows, events = audit.summarize(all_tracks)
        self.assertEqual(events, [])
        self.assertTrue(all(row["speed_max"] is None for row in rows if row["source"] == "state"))

    def test_baselines_are_separate_for_tasks_sources_and_sides(self):
        t = np.arange(6)*.1
        first, _ = analyze(t, x=.01*t)
        state, action = arrays(t, x=10*t)
        second, _ = audit.analyze_episode(1, state, action, t, np.ones(6))
        bases, _, _ = audit.summarize(first+second)
        self.assertAlmostEqual(bases[(0, "state", "left", "speed")]["median"], .01)
        self.assertAlmostEqual(bases[(1, "state", "left", "speed")]["median"], 10)


class ReaderAndCliTests(unittest.TestCase):
    def test_shared_parquet_is_read_once_and_episode_order_is_preserved(self):
        from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
        with TemporaryDirectory() as tmp:
            reader = V3EpisodeReader(fixture(Path(tmp)))
            with patch("pandas.read_parquet", wraps=pd.read_parquet) as call:
                episodes = list(audit.read_episodes(reader, [1, 0], "timestamp"))
                self.assertEqual(call.call_count, 1)
            self.assertEqual([episode for episode, _ in episodes], [1, 0])
            self.assertEqual(episodes[0][1].frame_index.tolist(), list(range(7)))

    def test_cli_is_read_only_and_empty_events_still_have_headers(self):
        with TemporaryDirectory() as tmp:
            root = fixture(Path(tmp)/"dataset", moving=False)
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            out = Path(tmp)/"report"
            result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out/"summary.json").read_text())
            self.assertEqual(summary["frames"], 14)
            self.assertEqual(summary["exported_events"], 0)
            self.assertEqual(pd.read_csv(out/"events.csv").shape[0], 0)
            self.assertEqual({str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}, before)
            repeat = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--out", str(out)], capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)

    def test_missing_timestamp_requires_explicit_fixed_rate(self):
        with TemporaryDirectory() as tmp:
            root = fixture(Path(tmp)/"dataset", timestamp=False)
            base = [sys.executable, str(SCRIPT), "--root", str(root)]
            missing = subprocess.run(base+["--out", str(Path(tmp)/"bad")], capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            fixed = subprocess.run(base+["--out", str(Path(tmp)/"fixed"), "--time-source", "fps", "--fps", "10"], capture_output=True, text=True)
            self.assertEqual(fixed.returncode, 0, fixed.stderr)
            report = json.loads((Path(tmp)/"fixed/summary.json").read_text())
            self.assertEqual(report["assumed_fps"], 10)

    def test_cli_uses_recorded_time_instead_of_nominal_video_fps(self):
        with TemporaryDirectory() as tmp:
            root = fixture(Path(tmp)/"dataset")
            out = Path(tmp)/"report"
            result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root),
                                     "--out", str(out), "--episodes", "0"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out/"summary.json").read_text())
            measured = next(b for b in summary["baselines"] if b["source"] == "state" and b["side"] == "left" and b["metric"] == "speed")
            self.assertAlmostEqual(measured["median"], .1)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional matplotlib is not installed")
    def test_optional_plot_is_a_real_png_without_modifying_source_data(self):
        with TemporaryDirectory() as tmp:
            root = fixture(Path(tmp)/"dataset")
            out = Path(tmp)/"report"
            env = os.environ.copy()
            env["MPLCONFIGDIR"] = str(Path(tmp)/"mpl-cache")
            result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--out", str(out),
                                     "--episodes", "0", "--plot-episodes", "0"], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            png = (out/"episode_000000.png").read_bytes()
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(len(png), 10000)


if __name__ == "__main__":
    unittest.main()
