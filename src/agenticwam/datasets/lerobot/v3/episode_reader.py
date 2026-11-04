"""Read LeRobot v3.0 packed Parquet/MP4 data without changing the dataset.

Expected directory is a *single* LeRobot v3 dataset root (not the parent of
multiple train_shard_* datasets).  All frame windows are indexed by the local
episode's frame_index; the shared MP4 has an extra video start timestamp.

The recorded MP4 is nominally 10 fps in this project.  If it was created from
motion-selected frames from original 15-fps footage, do not feed the original
irregular physical timestamps to its decoder.  The frame's position in the
encoded 10-fps MP4 determines the video seek time; the original timestamps may
be retained separately for research, but this loader uses index/fps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


class V3EpisodeReader:
    """Metadata-driven access to individual episodes in LeRobot v3.0 shards."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        if not str(self.info.get("codebase_version", "")).startswith("v3."):
            raise ValueError(f"Expected LeRobot v3.x at {self.root}; got {self.info.get('codebase_version')}")
        self.fps = float(self.info["fps"])
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"Invalid dataset fps: {self.fps}")

        paths = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No LeRobot v3 episode metadata under {self.root / 'meta/episodes'}")
        table = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
        required = {
            "episode_index", "data/chunk_index", "data/file_index",
            "dataset_from_index", "dataset_to_index",
        }
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"v3 episode metadata missing columns: {sorted(missing)}")

        table["episode_index"] = table["episode_index"].astype("int64")
        table = table.sort_values("episode_index").reset_index(drop=True)
        if table["episode_index"].duplicated().any():
            raise ValueError("Duplicate episode_index in v3 episode metadata")

        self.episode_table = table
        self.episodes = {int(row["episode_index"]): row.to_dict() for _, row in table.iterrows()}
        self.episode_ids = list(self.episodes)
        self.total_episodes = len(self.episodes)
        self.total_frames = sum(self.episode_length(i) for i in self.episode_ids)
        if self.total_episodes != int(self.info.get("total_episodes", self.total_episodes)):
            raise ValueError("meta/info.json total_episodes does not match episode metadata")
        if self.total_frames != int(self.info.get("total_frames", self.total_frames)):
            raise ValueError("meta/info.json total_frames does not match episode metadata")
        self.tasks = self._load_tasks()
        self.features = self.info["features"]
        self.video_keys = [k for k, v in self.features.items() if v["dtype"] == "video"]


    def _load_tasks(self) -> dict[int, str]:
        meta = self.root / "meta"
        pq = meta / "tasks.parquet"
        jl = meta / "tasks.jsonl"
    
        if pq.is_file():
            df = pd.read_parquet(pq)
    
            # LeRobot v3: task text may be stored in the DataFrame index.
            # Your dataset has:
            #   index   = task text (unnamed)
            #   columns = ["task_index"]
            if "task" not in df.columns:
                if isinstance(df.index, pd.RangeIndex):
                    raise ValueError(
                        f"{pq} has no task column or task-text index"
                    )
                df = df.rename_axis("task").reset_index()
    
            if "task_index" not in df.columns or "task" not in df.columns:
                raise ValueError(
                    f"{pq} must provide task_index and task; "
                    f"actual columns={df.columns.tolist()}"
                )
    
            if df["task_index"].isna().any() or df["task"].isna().any():
                raise ValueError(f"{pq} contains null task metadata")
    
            if df["task_index"].duplicated().any():
                raise ValueError(f"{pq} contains duplicate task_index values")
    
            out = {
                int(r.task_index): str(r.task)
                for r in df.itertuples(index=False)
            }
    
        elif jl.is_file():
            out = {}
            for line in jl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    out[int(item["task_index"])] = str(item["task"])
    
        else:
            raise FileNotFoundError(
                f"Missing meta/tasks.parquet or meta/tasks.jsonl in {meta}"
            )
    
        return out


    def episode(self, episode_index: int) -> dict[str, Any]:
        try:
            return self.episodes[int(episode_index)]
        except KeyError as exc:
            raise IndexError(f"Unknown episode_index={episode_index}") from exc

    def episode_length(self, episode_index: int) -> int:
        ep = self.episode(episode_index)
        length = int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
        if length <= 0:
            raise ValueError(f"Episode {episode_index} has invalid length {length}")
        if "length" in ep and pd.notna(ep["length"]):
            if int(ep["length"]) != length:
                raise ValueError(f"Episode {episode_index} inconsistent length metadata")
        return length

    def data_file(self, episode_index: int) -> Path:
        ep = self.episode(episode_index)
        rel = self.info["data_path"].format(
            chunk_index=int(ep["data/chunk_index"]),
            file_index=int(ep["data/file_index"]),
        )
        path = self.root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def video_file(self, episode_index: int, video_key: str) -> Path:
        if video_key not in self.video_keys:
            raise KeyError(f"Unknown video key: {video_key}")
        ep = self.episode(episode_index)
        rel = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=int(ep[f"videos/{video_key}/chunk_index"]),
            file_index=int(ep[f"videos/{video_key}/file_index"]),
        )
        path = self.root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def video_start(self, episode_index: int, video_key: str) -> float:
        ep = self.episode(episode_index)
        start = float(ep[f"videos/{video_key}/from_timestamp"])
        if not math.isfinite(start) or start < 0:
            raise ValueError(f"Invalid video start timestamp for {episode_index}/{video_key}: {start}")
        return start

    def video_frame_window(self, episode_index: int, video_key: str) -> tuple[Path, int, int]:
        """For codegen LazyClip(path, first_frame=..., length=...)."""
        return (
            self.video_file(episode_index, video_key),
            round(self.video_start(episode_index, video_key) * self.fps),
            self.episode_length(episode_index),
        )

    def video_timestamps(self, episode_index: int, video_key: str, indices: Sequence[int]) -> list[float]:
        """Absolute *encoded-video* timestamps, not irregular source capture times."""
        length = self.episode_length(episode_index)
        if any(int(i) < 0 or int(i) >= length for i in indices):
            raise IndexError(f"video frame index outside episode {episode_index} (length={length})")
        base = self.video_start(episode_index, video_key)
        return [base + int(i) / self.fps for i in indices]

    def read_episode(self, episode_index: int, columns: list[str] | None = None) -> pd.DataFrame:
        """Return ONLY this episode, sorted by frame_index, independent of file packing.

        Parquet predicate pushdown may skip irrelevant row groups; the filter is
        always rechecked after reading.  Keeps no DataFrame cache to avoid
        duplicating a potentially large shard in multiple DataLoader workers.
        """
        p = self.data_file(episode_index)
        cols = None if columns is None else list(dict.fromkeys([*columns, "episode_index", "frame_index"]))
        df = pd.read_parquet(p, columns=cols, filters=[("episode_index", "==", int(episode_index))])
        if "episode_index" not in df:
            raise KeyError(f"{p} has no episode_index column")
        df = df.loc[df["episode_index"] == int(episode_index)]
        df = df.sort_values("frame_index").reset_index(drop=True)
        n = self.episode_length(episode_index)
        if len(df) != n:
            raise ValueError(f"Episode {episode_index}: loaded {len(df)} frames, expected {n}; file={p}")
        frame_indices = df["frame_index"].to_numpy()
        if not np.array_equal(frame_indices, np.arange(n)):
            raise ValueError(f"Episode {episode_index}: frame_index is not 0..{n-1}")
        return df


__all__ = ["V3EpisodeReader"]
