"""Build frame-to-task labels from actual selected v3 Parquet rows, without video."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass
class TaskSamplingIndex:
    task_ids: np.ndarray
    task_names: tuple[str, ...]
    sha256: str

    def counts(self):
        return np.bincount(self.task_ids, minlength=len(self.task_names)).tolist()


def build_task_sampling_index(dataset):
    import pandas as pd

    base = getattr(dataset, "lerobot_dataset", dataset)
    multi = getattr(base, "multi_dataset", base)
    singles = getattr(multi, "_datasets", None)
    if singles is None or not singles:
        raise TypeError("Task-balanced sampling requires the project's v3 map-style loader")
    names, name_to_id, selection = [], {}, []
    task_ids = np.full(len(dataset), -1, dtype=np.int64)
    global_start = 0
    for single in singles:
        reader = single.meta.reader
        local_map = {}
        for raw_id, name in sorted(reader.tasks.items()):
            if name not in name_to_id:
                name_to_id[name] = len(names)
                names.append(name)
            local_map[int(raw_id)] = name_to_id[name]
        files = defaultdict(list)
        for slot, episode in enumerate(single.episodes):
            length = reader.episode_length(episode)
            files[reader.data_file(episode)].append((episode, single._starts[slot], length))
        for path, episodes in files.items():
            frame = pd.read_parquet(path, columns=["episode_index", "frame_index", "task_index"],
                                    filters=[("episode_index", "in", [item[0] for item in episodes])])
            groups = {int(episode): rows for episode, rows in frame.groupby("episode_index", sort=False)}
            for episode, start, length in episodes:
                if episode not in groups:
                    raise ValueError(f"Missing episode {episode} in {path}")
                rows = groups[episode].sort_values("frame_index")
                if len(rows) != length or not np.array_equal(rows.frame_index.to_numpy(), np.arange(length)):
                    raise ValueError(f"Invalid frame order/length for episode {episode}")
                raw = rows.task_index.to_numpy()
                if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
                    raise ValueError(f"Invalid task_index values for episode {episode}")
                raw = raw.astype(np.int64)
                try:
                    labels = np.asarray([local_map[int(task)] for task in raw], dtype=np.int64)
                except KeyError as exc:
                    raise ValueError(f"task_index missing from metadata in {path}") from exc
                begin = global_start + int(start)
                task_ids[begin:begin + length] = labels
        selection.append(dict(root=str(single.root.resolve()), episodes=list(single.episodes),
                              lengths=[reader.episode_length(ep) for ep in single.episodes]))
        global_start += len(single)
    if global_start != len(dataset) or np.any(task_ids < 0):
        raise ValueError("Selected v3 sample order does not cover the full training dataset")
    present = np.unique(task_ids)
    remap = np.full(len(names), -1, dtype=np.int64)
    remap[present] = np.arange(len(present))
    task_ids = remap[task_ids]
    names = tuple(names[int(task)] for task in present)
    digest = hashlib.sha256(task_ids.astype("<i8").tobytes())
    digest.update(json.dumps(dict(selection=selection, task_names=names), sort_keys=True).encode())
    task_ids.flags.writeable = False
    return TaskSamplingIndex(task_ids=task_ids, task_names=names, sha256=digest.hexdigest())
