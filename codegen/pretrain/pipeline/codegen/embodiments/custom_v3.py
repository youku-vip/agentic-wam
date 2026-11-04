"""Custom34 LeRobot v3 data, original YAM API/prompts/library/grounding pipeline."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from codegen.skill_paths import library_dir

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / 'src'))
OUTPUTS = Path(__file__).resolve().parents[2] / 'outputs'


def required_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'Set {name}; no source dataset files are overwritten')
    return Path(value).expanduser().resolve()


@lru_cache(maxsize=1)
def reader():
    from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
    return V3EpisodeReader(required_path('CUSTOM_V3_DATASET'))


def per_arm(states, fps):
    from codegen.grounding.transitions import grip_transitions, debounce_events
    if states.ndim != 2 or states.shape[1] != 34 or not np.isfinite(states).all():
        raise ValueError('Expected finite custom state [T,34]')
    # Same velocity-sign detector, durations scaled from its 30Hz defaults.
    # Edge padding avoids zero-padding inventing transitions on constant-open data.
    window = max(1, round(5*fps/30))
    if window % 2 == 0:
        window += 1
    result = {}
    high = os.environ.get('CUSTOM_V3_OPEN_IS_HIGH', '1') == '1'
    for side, position, grip in [('left', 14, 32), ('right', 23, 33)]:
        pos = states[:, position:position+3].astype(float)
        signal = states[:, grip] if high else -states[:, grip]
        raw = grip_transitions(signal, smooth_frames=window,
                               min_run=max(1, round(4*fps/30)),
                               velocity_epsilon=.01*30/fps, edge_pad=True)
        events, dropped = debounce_events(raw, pos, min_dwell=max(1,round(20*fps/30)), hover_eps=.02)
        result[side] = dict(pos=pos, grip_events=events, dropped=dropped)
    return result


def load_custom_episode(task, episode_index=0):
    if task != 'custom':
        raise ValueError('Custom v3 work unit is custom:<episode_index>')
    from codegen.data.frames import LazyClip
    from codegen.data.readers import Episode
    rd = reader()
    df = rd.read_episode(episode_index, columns=['observation.state', 'task_index'])
    states = np.stack(df['observation.state']).astype(float)
    tasks = set(map(int, df['task_index']))
    if len(tasks) != 1:
        raise ValueError('One task per episode required')
    instruction = rd.tasks[tasks.pop()]
    path = required_path('CUSTOM_V3_WORK') / 'subtasks.json'
    rec = json.loads(path.read_text())[str(episode_index)]
    if rec.get('dry_run', True):
        raise ValueError('Only dry-run boundaries exist; run custom_v3_codegen.py subtasks --label first')
    if rec['instruction'] != instruction:
        raise ValueError('Subtask instruction differs from source task metadata')
    if rec.get('source_state_sha256') != hashlib.sha256(states.tobytes()).hexdigest():
        raise ValueError('Raw episode state differs from subtask sidecar')
    sub_id = np.asarray(rec['sub_id'],dtype=np.int32)
    if len(sub_id) != len(states):
        raise ValueError('Subtask frame count differs from source')
    arms = per_arm(states,rd.fps)
    if rec.get("events") != {s:[list(e) for e in v["grip_events"]] for s,v in arms.items()}:
        raise ValueError("Detector events differ from subtask sidecar; use a new work directory")
    video, first, length = rd.video_frame_window(episode_index, 'observation.images.head')
    return Episode(instruction=instruction, clip=LazyClip(video, first_frame=first,length=length),
                   sub_id=sub_id, text={int(k):v for k,v in rec['text'].items()},
                   per_arm=arms,fps=rd.fps,
                   meta=dict(dataset=str(rd.root), episode_index=episode_index, source='custom_v3',
                             codegen_signature=hashlib.sha256(json.dumps(rec,sort_keys=True).encode()).hexdigest(),
                             detector='fps-scaled-yam-velocity', video_start_frame=first),
                   episode_id=f'episode_{episode_index:03d}')


class CustomV3Spec:
    name='custom_v3'
    display_name='Custom34 v3 bimanual parallel grippers'
    noise_labels=frozenset()
    skills_dir=library_dir('custom_v3')/'skills'

    @property
    def min_block(self):
        return max(1,round(8*reader().fps/30))

    def list_tasks(self):
        return ['custom']

    def preflight(self,task,episode_index=0):
        try:
            load_custom_episode(task,episode_index)
        except (OSError,ValueError,KeyError,RuntimeError,IndexError) as exc:
            return str(exc)
        return None

    def run_dir(self,task,tag,model_short='gemini-3.1-pro',is_default_model=True,episode_id='episode_000'):
        return OUTPUTS/self.name/task/model_short/episode_id/(tag or 'dev')

    def batch_dir(self,tag):
        return OUTPUTS/self.name/f'batch_{tag}'

    def load_episode(self,task,episode_index=0):
        return load_custom_episode(task,episode_index)

    def load_state(self,task,episode_index=0):
        ep=self.load_episode(task,episode_index)
        return dict(T=len(ep.sub_id),per_arm=ep.per_arm,sub_id=ep.sub_id,text=ep.text,instruction=ep.instruction)


SPEC=CustomV3Spec()
