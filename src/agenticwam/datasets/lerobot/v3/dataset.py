"""LeRobot v3.0 -> Agentic-WAM BaseLerobotDataset compatibility adapter.

Drop this package at ``src/agenticwam/datasets/lerobot/v3/`` and switch
BaseLerobotDataset to V3LeRobotDatasetMetadata / V3MultiLeRobotDataset when
meta/info.json has codebase_version=v3.x (see companion integration patch).

No change to RobotVideoDataset, AgenticWAMProcessor, 34D/20D EEF transform,
or custom.yaml is required for this loader interface.
"""

from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..lerobot.datasets.video_utils import decode_video_frames
from .episode_reader import V3EpisodeReader


def _tensor_rows(frame, key: str, rows: Sequence[int]) -> torch.Tensor:
    """Read scalar/vector Parquet cells as [T, ...] torch tensors."""
    if key not in frame.columns:
        raise KeyError(f"Missing feature {key!r} in episode Parquet")
    values = frame.iloc[list(rows)][key].tolist()
    if not values:
        raise ValueError(f"Cannot sample empty feature {key!r}")
    # Each state/action cell is a vector of numpy floats, not a scalar.
    if isinstance(values[0], (list, tuple, np.ndarray)):
        array = np.stack([np.asarray(v) for v in values], axis=0)
    else:
        array = np.asarray(values)
    if array.dtype.kind in "OUS":
        raise TypeError(f"Feature {key!r} is not a numeric tensor (dtype={array.dtype})")
    return torch.from_numpy(np.ascontiguousarray(array))


class V3LeRobotDatasetMetadata:
    """Mirror only the v2 metadata properties that Agentic-WAM actually uses."""

    def __init__(self, repo_id: str, root: str | Path, **_: Any):
        self.repo_id = str(repo_id)
        self.root = Path(root)
        self.reader = V3EpisodeReader(self.root)
        self.info = self.reader.info
        self.tasks = self.reader.tasks
        self.episodes = self.reader.episodes
        self.features = self.reader.features
        self.total_episodes = self.reader.total_episodes
        self.total_frames = self.reader.total_frames

    @property
    def fps(self) -> float:
        return self.reader.fps

    def get_data_file_path(self, episode_index: int) -> Path:
        return self.reader.data_file(episode_index).relative_to(self.reader.root)

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        return self.reader.video_file(episode_index, video_key).relative_to(self.reader.root)


class _V3SingleDataset(torch.utils.data.Dataset):
    """One v3 dataset, with index-based windows clipped to episode boundaries."""

    def __init__(
        self,
        ds_root: str | Path,
        selected_episodes: list[int] | None,
        delta_timestamps: dict[str, list[float]],
        tolerance_s: float | None = None,
        video_backend: str | None = None,
    ):
        self.root = Path(ds_root)
        self.meta = V3LeRobotDatasetMetadata(str(ds_root), self.root)
        reader = self.meta.reader
        self.episodes = (
            list(reader.episode_ids)
            if selected_episodes is None
            else [int(i) for i in selected_episodes]
        )
        if len(set(self.episodes)) != len(self.episodes):
            raise ValueError(f"Duplicate episode IDs in selection for {self.root}")
        for ep_id in self.episodes:
            reader.episode(ep_id)

        lengths = [reader.episode_length(i) for i in self.episodes]
        ends = np.cumsum(lengths, dtype=np.int64)
        self._starts = [0, *map(int, ends[:-1])]
        self._ends = list(map(int, ends))
        self.episode_data_index = {
            "from": torch.tensor(self._starts, dtype=torch.long),
            "to": torch.tensor(self._ends, dtype=torch.long),
        }
        self.num_frames = self._ends[-1] if self._ends else 0
        self.num_episodes = len(self.episodes)

        self.video_backend = video_backend
        self.tolerance_s = float(tolerance_s or 1.0 / reader.fps - 1e-4)
        self.during_training = True
        self.delta_indices = {
            key: [round(float(dt) * reader.fps) for dt in deltas]
            for key, deltas in delta_timestamps.items()
        }
        self.features = reader.features
        for key in self.delta_indices:
            if key not in self.features:
                raise KeyError(f"Missing feature {key} in {self.root}/meta/info.json")

    @lru_cache(maxsize=64)
    def _table(self, episode_index: int):
        return self.meta.reader.read_episode(episode_index)

    def get_episode_data(self, local_episode_slot: int) -> dict[str, torch.Tensor]:
        ep_id = self.episodes[local_episode_slot]
        frame = self._table(ep_id)
        result = {}
        indices = range(len(frame))
        for key, spec in self.features.items():
            if spec.get("dtype") != "video" and key in frame.columns:
                result[key] = _tensor_rows(frame, key, indices)
        return result

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not (0 <= index < self.num_frames):
            raise IndexError(f"index {index} outside [0,{self.num_frames})")

        slot = bisect_right(self._ends, index)
        ep_id = self.episodes[slot]
        local_index = index - self._starts[slot]
        length = self.meta.reader.episode_length(ep_id)
        frame = self._table(ep_id)

        sample: dict[str, Any] = {}
        for key, deltas in self.delta_indices.items():
            unbounded = [local_index + delta for delta in deltas]
            clipped = [min(max(i, 0), length - 1) for i in unbounded]
            sample[f"{key}_is_pad"] = torch.tensor(
                [i < 0 or i >= length for i in unbounded], dtype=torch.bool
            )
            if key in self.meta.reader.video_keys:
                # The Parquet timestamp can reflect original irregular 15-fps
                # capture times.  The *encoded* MP4 is indexed by 10-fps frame
                # positions; use video from_timestamp + index / encoded fps.
                path = self.meta.reader.video_file(ep_id, key)
                query_timestamps = self.meta.reader.video_timestamps(ep_id, key, clipped)
                sample[key] = decode_video_frames(
                    path, query_timestamps, self.tolerance_s, self.video_backend
                )  # float32 [T,3,H,W] in [0,1]
            else:
                sample[key] = _tensor_rows(frame, key, clipped)

        raw = frame.iloc[local_index]
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            if key in frame.columns:
                sample[key] = torch.as_tensor(raw[key])
        task_index = int(raw["task_index"])
        if task_index not in self.meta.tasks:
            raise KeyError(f"task_index={task_index} missing from v3 task metadata")
        sample["task"] = self.meta.tasks[task_index]
        return sample


class V3MultiLeRobotDataset(torch.utils.data.Dataset):
    """Matching interface for BaseLerobotDataset's v2 MultiLeRobotDataset.

    Accepts dataset dirs in the same order as lerobot_shards resolver.  The
    training code's global code_id_by_idx is indexed in this concatenation
    order (not necessarily the Parquet's original global `index` field).
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict[str, list[int]] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        if not dataset_dirs:
            raise ValueError("At least one v3 dataset root is required")
        if image_transforms is not None:
            raise NotImplementedError("Use AgenticWAMProcessor for image transforms")
        if not download_videos:
            raise NotImplementedError("This local v3 adapter expects videos to exist")

        delta_timestamps = delta_timestamps or {}
        self.ds_names = [str(p) for p in dataset_dirs]
        self._datasets = [
            _V3SingleDataset(
                ds_root=name,
                selected_episodes=episodes.get(name) if episodes is not None else None,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerances_s.get(name) if tolerances_s else None,
                video_backend=video_backend,
            )
            for name in self.ds_names
        ]
        self._ends = list(np.cumsum([ds.num_frames for ds in self._datasets], dtype=np.int64))
        keys = [set(ds.features) for ds in self._datasets]
        common = set.intersection(*keys)
        if any(k not in common for k in delta_timestamps):
            raise ValueError("Not all v3 datasets provide the requested state/action/camera keys")
        self.disabled_features = set.union(*keys) - common

    @property
    def num_frames(self) -> int:
        return int(self._ends[-1]) if self._ends else 0

    @property
    def num_episodes(self) -> int:
        return sum(ds.num_episodes for ds in self._datasets)

    @property
    def fps(self) -> float:
        return self._datasets[0].meta.fps

    def set_during_training(self, flag: bool):
        for dataset in self._datasets:
            dataset.during_training = bool(flag)

    def get_episode_data(self, episode_index: int) -> dict[str, torch.Tensor]:
        episode_index = int(episode_index)
        if episode_index < 0:
            raise IndexError(episode_index)
        for ds in self._datasets:
            if episode_index < ds.num_episodes:
                return ds.get_episode_data(episode_index)
            episode_index -= ds.num_episodes
        raise IndexError(f"Global episode index out of range: {episode_index}")

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not (0 <= index < self.num_frames):
            raise IndexError(f"Global sample index {index} out of range")
        ds_index = bisect_right(self._ends, index)
        start = 0 if ds_index == 0 else int(self._ends[ds_index - 1])
        item = self._datasets[ds_index][index - start]
        item["dataset_index"] = torch.tensor(ds_index, dtype=torch.long)
        for key in self.disabled_features:
            item.pop(key, None)
        return item


__all__ = ["V3LeRobotDatasetMetadata", "V3MultiLeRobotDataset"]
