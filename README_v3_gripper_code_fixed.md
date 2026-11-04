# LeRobot v3.0 Air Fryer Coding — targeted fixes after 4348-episode diagnostics

This is a **modified copy** of `build_v3_gripper_code.py`, not a change to the original. It is intended for the four-task air-fryer dataset (34D state, 20D action, LeRobot v3.0 packed parquet), and it does not execute any program on the robot.

## What was observed

- Task 0, 1, 2: 1087 episodes each, exactly one CLOSE and one OPEN per episode.
- Task 2: all 1087 `rotate` estimates were rejected because the prior run set `--knob-max-xyz-drift-m 0.005`, but the sampled episodes have ~0.012–0.016 m XYZ displacement within the held-gripper span.
- Task 3: all 1087 episodes have constant `observation.state[32] == 1`, `observation.state[33] == 1`, as well as constant `action[18] == action[19] == 1`. No gripper event exists in the recorded streams. Lowering thresholds cannot fix a constant signal.

## Targeted changes

1. `--task3-open-push` opt-in, for **Task 3 only**. If the left gripper stays OPEN and no arm gripper transitions are found, use measured left EEF motion to create `robot.move_arm()` labels. Never invent `robot.grasp()`, `robot.release()`, or `robot.place()` for Task 3. If the end effector first travels away from its initial pose and then comes meaningfully back, split into a **task3_open_push** stage and a **task3_retreat** stage. If reversal is not observable, keep a single motion stage and flag review. If no motion is observable, leave it unlabeled. This is a *kinematic heuristic*, not proof the door closed.
2. No change to Tasks 0/1/2 event logic. For Task 2, **experiment** with a less restrictive `--knob-max-xyz-drift-m 0.02` value. The pose-estimated rotation remains subject to path-angle and local-axis-coherence checks. A 20-mm threshold does not prove the knob was truly held fixed and should be visually audited.
3. All event-free, inferred Task 3 episodes still get explicit warnings saying that contact has not been visually verified; do not interpret warning counts alone as failures.

## Run without overwriting the first cache

```bash
pip install numpy torch pyarrow

ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
mkdir -p "$ROOT/coding_v2"

python3 build_v3_gripper_code_fixed.py \
  --dataset "$ROOT" \
  --open-is-high \
  --gripper-full-stroke-mm 70 \
  --task3-open-push \
  --knob-axis auto \
  --knob-max-xyz-drift-m 0.02 \
  --knob-min-angle-deg 8 \
  --retreat-min-motion-m 0.015 \
  --retreat-step-m 0.0015 \
  --retreat-max-waypoints 3 \
  --out "$ROOT/coding_v2/policy_code_idx_custom.pt" \
  --audit "$ROOT/coding_v2/gripper_code_audit.jsonl"
```

Outputs: `coding_v2/policy_code_idx_custom.pt`, `coding_v2/episode_order.json`, `coding_v2/gripper_code_audit.jsonl`.

## Check the new labels

```bash
ROOT=/data/datasets/air_fryer_wam/wam/air_fryer_all_0813_0827_1087_v30
python3 - "$ROOT/coding_v2/gripper_code_audit.jsonl" <<'PY'
import json, sys
from collections import defaultdict, Counter
stats = defaultdict(Counter)
examples = {}
for line in open(sys.argv[1], encoding='utf-8'):
    r = json.loads(line)
    t = r['task_index']; st = stats[t]
    st['episodes'] += 1
    st['frames'] += r['n_frames']
    st['labeled'] += sum(s['end']-s['start'] for s in r['spans'] if s['code'])
    st['rotates'] += sum('robot.rotate(' in s['code'] for s in r['spans'])
    st['grasp'] += sum('robot.grasp(' in s['code'] for s in r['spans'])
    st['task3_retreat'] += sum(s['event']=='task3_retreat' for s in r['spans'])
    st['task3_push'] += sum(s['event']=='task3_open_push' for s in r['spans'])
    if t not in examples: examples[t] = r
for t, st in sorted(stats.items()):
    print('Task', t, dict(st), 'coverage', round(st['labeled']/st['frames'],3))
for t in [2,3]:
    r = examples[t]
    print('example task',t,'episode',r['episode_index'])
    for span in r['spans']:
        print(span['start'],span['end'],span['event'],span['code'][:160])
    if t == 3: print('task3 audit',r.get('task3_open_push_audit'))
PY
```

Expected *structural* checks: the total number of cache indices should still be 285262; Task 3 should never have grasp/release/pick/place calls when operating in open-push mode; `task3_retreat` must appear only when a plausible return movement was detected. Task 2 `robot.rotate` coverage must be counted rather than inferred from a warning number. Review video on 10–20 episodes for Task 2 and 3 before training.

## Calibration limits

- `robot.rotate` assumes rotation about the **gripper approach axis**. `--knob-axis auto` estimates the dominant local wrist rotation axis but does not prove it matches the physical knob axis. Verify rotation sign and coordinate convention. The original source code assumes rot6D is the first **two columns** of the end-effector rotation matrix.
- Task 3's wrist path cannot by itself verify door contact or closure. In particular, "farthest distance from initial pose" is a **heuristic cut**, not a recognized door-contact timestamp.
- `policy_code_idx_custom.pt` must be aligned to the loader's exact episode/frame order before training. A syntactically successful cache is not sufficient.
