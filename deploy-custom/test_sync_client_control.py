"""Execute the actual sync loop with simulated env/client; never import robot SDKs."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE=Path(__file__).with_name('rtc_client_agenticwam.py')


class Logger:
    def __getattr__(self,name):return lambda *args,**kwargs:None


def execute(*,steps=40,stale_grip=False,done_every=False,switch_and_fail=False,pause_same_prompt=False,pause_replay=False,
            auto_recovery=False, repeat_prompt=False, infer_fail_at=None):
    tree=ast.parse(SOURCE.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    selected=[n for n in main.body if isinstance(n,(ast.ClassDef,ast.AsyncFunctionDef))
              and n.name in ['GripperStateStabilizer','run_loop_sync']]
    shared=SimpleNamespace(value='task1',pauses=2,revision=0)
    def prompt():
        if pause_same_prompt and env.primary==5 and shared.pauses:
            shared.pauses-=1
            return ''
        return shared.value
    shared.get=prompt
    shared.snapshot=lambda: (prompt(),shared.revision)
    class Env:
        def __init__(self):self.commands=[];self.primary=0;self.resets=0;self.obs_reads=0
        def observation(self,fresh=False):
            grip=0 if stale_grip and not fresh else 1000
            if switch_and_fail and fresh and self.primary>=40:grip=0
            return dict(ee_pos_quat=np.full(12,10 if shared.value=='task1' else 20),
                        arm_actpos=dict(left=[1000],right=[grip]))
        def reset(self):self.resets+=1;return self.observation(),{}
        def get_obs(self):self.obs_reads+=1;return self.observation()
        def step(self,cmd,get_obs=True):
            self.commands.append(cmd)
            # Primary policy commands have a prediction id in position zero.
            if cmd['ee_pos_quat'][0] in (1,2,3,4):
                self.primary+=1
                if switch_and_fail and self.primary==20:shared.value='task2'
                if repeat_prompt and self.primary==5:shared.revision+=1
            return self.observation(fresh=True),0,done_every,False,{}
    env=Env()
    class Client:
        s_min=20
        prompt='task1'
        def __init__(self):self.calls=0;self.resets=0;self.stopped=False
        async def connect(self):pass
        async def reset_episode(self):self.resets+=1
        async def guided_inference(self,obs,**kwargs):
            self.calls+=1
            if self.calls==infer_fail_at:raise ConnectionError('temporary inference failure')
            a=np.zeros((32,20));a[:,0]=self.calls;a[:,18:20]=1
            if pause_same_prompt and self.calls==1:a[1:,18:20]=0
            return dict(action=a)
        def suppress_rtc_next(self):pass
        async def stop(self):self.stopped=True
    rtc=Client()
    replay_checks=[2]
    def replaying():
        if pause_replay and env.primary==5 and replay_checks[0]:
            replay_checks[0]-=1
            return True
        return False
    async def thread(fn,*args,**kwargs):return fn(*args,**kwargs)
    async def sleep(_):pass
    async def get_obs():return env.get_obs()
    namespace=dict(asyncio=SimpleNamespace(to_thread=thread,sleep=sleep),np=np,
                   time=__import__('time'),logger=Logger(),env=env,rtc=rtc,shared_prompt=shared,
                   replay_ctrl=SimpleNamespace(is_replaying=replaying),
                   get_obs_with_retry=get_obs,trans_utils=SimpleNamespace(rot6d_to_euler_angle=lambda _:np.zeros(3)),
                   auto_gripper_recovery=auto_recovery,gripper_lock_steps=0,
                   cfg=SimpleNamespace(env=SimpleNamespace(num_steps=steps,max_hz=0)))
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),namespace)
    asyncio.run(asyncio.wait_for(namespace['run_loop_sync'](),timeout=2))
    return env,rtc


def test_one_inference_per_executed_chunk():
    env,rtc=execute()
    assert rtc.calls==2 and len(env.commands)==40 and rtc.stopped
    assert [c['ee_pos_quat'][0] for c in env.commands]==[1]*20+[2]*20
    assert env.obs_reads==2


def test_stale_previous_gripper_feedback_does_not_trigger_recovery():
    env,rtc=execute(steps=20,stale_grip=True)
    assert rtc.calls==1 and len(env.commands)==20
    assert all(c['ee_pos_quat'][0]==1 for c in env.commands)


def test_done_counts_consumed_action_and_resets_remote_episode():
    env,rtc=execute(steps=3,done_every=True)
    assert env.primary==3 and rtc.calls==3
    assert env.resets==4 and rtc.resets==2


def test_task_switch_refreshes_recovery_anchor():
    env,rtc=execute(steps=40,switch_and_fail=True,auto_recovery=True)
    assert rtc.calls==2 and rtc.resets==1
    assert env.commands[-1]['gripper_positions']==['open','open']
    assert np.all(env.commands[-1]['ee_pos_quat']==20) # second episode, not lifetime-first pose 10


def test_same_prompt_clear_resume_resets_remote_and_gripper_latch():
    env,rtc=execute(pause_same_prompt=True)
    assert rtc.resets==1 and rtc.calls==3 and len(env.commands)==40
    assert env.commands[4]['gripper_positions']==['close','close']
    assert env.commands[5]['gripper_positions']==['open','open']


def test_sync_checks_replay_between_commands():
    env,rtc=execute(pause_replay=True)
    assert rtc.resets==1 and rtc.calls==3
    assert [c['ee_pos_quat'][0] for c in env.commands[:5]]==[1]*5
    assert env.commands[5]['ee_pos_quat'][0]==2


def test_repeated_same_prompt_restarts_and_aborts_old_chunk():
    env,rtc=execute(repeat_prompt=True)
    assert rtc.resets==1 and rtc.calls==3
    assert [c['ee_pos_quat'][0] for c in env.commands[:5]]==[1]*5
    assert env.commands[5]['ee_pos_quat'][0]==2


def test_no_automatic_recovery_when_feedback_is_closed():
    env,rtc=execute(steps=40,switch_and_fail=True)
    assert len(env.commands)==40 and env.commands[-1]['ee_pos_quat'][0]==2


def test_failed_inference_fetches_fresh_observation_and_sends_no_command():
    env,rtc=execute(infer_fail_at=2)
    assert rtc.calls==3 and env.obs_reads==3 and len(env.commands)==40
    assert [c['ee_pos_quat'][0] for c in env.commands]==[1]*20+[3]*20
