# Agentic-WAM: LeRobot v3.0 dataset adapter

Files in `v3/` are for `src/agenticwam/datasets/lerobot/v3/`.

These files implement metadata-driven v3 file lookup, per-episode Parquet
filtering, shared-MP4 start offsets, camera decoding, and padded 33-frame
windows for the original Agentic-WAM `BaseLerobotDataset` interface.

## Installation

1. In your Agentic-WAM checkout, create directory
   `src/agenticwam/datasets/lerobot/v3/` and copy all three files from `v3/`
   (`__init__.py`, `episode_reader.py`, `dataset.py`).

2. In `src/agenticwam/datasets/lerobot/base_lerobot_dataset.py`, replace the
   **single import**:

   ```python
   from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
   ```

   with:

   ```python
   from .v3.dataset import V3LeRobotDatasetMetadata as LeRobotDatasetMetadata
   from .v3.dataset import V3MultiLeRobotDataset as MultiLeRobotDataset
   ```

   This one-line logical switch makes this Agentic-WAM checkout **v3-only**.
   Do not use it in the same process to read v2.0/v2.1 datasets.  If you need
   mixed versions, select metadata/loader classes separately per version.

3. Put the previously supplied `custom_unified_eef.py` in
   `src/agenticwam/datasets/lerobot/transforms/` and `custom.yaml` in
   `configs/real_post_train/data/`.

4. Your `data.dataset_root` may point **directly to the v3 root** containing
   `meta/`, `data/`, `videos/`.  No `train_shard_0` wrapper is needed.
   The project's `${lerobot_shards:...}` resolver accepts that layout.

5. In your Agentic-WAM Python environment, install `torch`, `numpy`,
   `pandas`, `pyarrow`, `torchvision`, and a working video backend with AV1
   support (e.g. `av`/PyAV). Prefer project-locked compatible versions.

6. Put `smoke_test_v3.py` in the repository root and run:

   ```bash
   PYTHONPATH=src python smoke_test_v3.py \
       --root /absolute/path/to/my_v3_dataset --video-backend pyav
   ```

   The test validates 34D state, 20D action, three image streams, the total
   frame count, and episode-boundary padding.

   Before working with the real dataset, the standalone metadata test is:

   ```bash
   python agenticwam_v3/test_episode_reader_mock.py
   ```

   It uses in-memory mocked Parquet tables, so it doesn't require PyTorch or
   the real dataset; it does **not** verify video decoding.

## What this does NOT yet implement

The following Agentic-WAM scripts independently hardcode LeRobot v2.1 paths
and metadata. You must adapt them before completing a full training run:

- `utils/make_stats.py`: read each episode, not each whole packed Parquet;
  otherwise 32-step actions can cross episode boundaries.
- `codegen/posttrain/annotate_subtasks_gripper.py`: use `V3EpisodeReader`,
  34D state offsets 14:17, 23:26, grippers 32/33, and camera `head`.
- `codegen/pretrain/pipeline/codegen/embodiments/yam_sft.py`: use this reader;
  `codegen.data.frames.LazyClip` **already supports** `first_frame` and
  `length`, matching `reader.video_frame_window(...)`.
- `codegen/posttrain/build_trainable_cache.py` and
  `scripts/preflight_sft.py`: read v3 `meta/episodes/**/*.parquet` rather than
  `meta/episodes.jsonl`, and verify that `code_id_by_idx` aligns to the exact
  loader enumeration order.
- `scripts/preflight_sft.py` also needs a `custom` entry for the
  `CustomUnifiedEEFTransform` class.

The 10-fps MP4s came from motion-adaptively selected original 15-fps frames.
This adapter deliberately uses **encoded-video frame index/fps** for seeking,
and treats action horizons as indexed keyframe steps, not constant *physical*
time. That limitation matters for real-robot control.

## Useful episode reader methods

```python
from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader

reader = V3EpisodeReader('/data/my_v3_dataset')
frame_df = reader.read_episode(0)  # only episode 0, even in shared file
path, first_frame, length = reader.video_frame_window(
    0, 'observation.images.head'
)
# For code-generation scripts:
# clip = LazyClip(path, first_frame=first_frame, length=length)
```

## Runtime and verification status

The files are syntax-checked, and the mock episode-reader test passes.
The ChatGPT scratch environment does **not**
contain PyTorch or PyArrow, and the user's actual dataset has not been
uploaded; therefore real data/video decoding and full training have **not**
been tested here. Run the smoke test before launching a job.
