"""GPU-free review of real custom server runtime and uploaded client protocol."""
import asyncio
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys
import threading
from types import SimpleNamespace
from urllib.request import Request,urlopen

import numpy as np
import pytest
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agenticwam.datasets.lerobot.transforms.custom_unified_eef import CustomUnifiedEEFTransform

spec=importlib.util.spec_from_file_location('custom_server',ROOT/'deploy-custom/custom34_server.py')
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
client=runpy.run_path(str(ROOT/'deploy-custom/test_custom34_rtc_adapter.py'))


def raw_state(x=.2):
    s=np.zeros(34,np.float32)
    s[14:17]=[x,.1,.3];s[23:26]=[-.2,.1,.4]
    s[17:23]=s[26:32]=[1,0,0,0,1,0];s[32:34]=1
    return s


def request_body(x=.2,instruction='pick cake'):
    image=client['encode_png_b64'](np.zeros((24,32,3),np.uint8))
    return dict(instruction=instruction,state=raw_state(x).tolist(),images={k:image for k in ['head','left','right']})


def runtime(objective='joint'):
    rt=server.Custom34Runtime.__new__(server.Custom34Runtime)
    rt.transform=CustomUnifiedEEFTransform(['default']);rt.key='default';rt.horizon=32
    rt.lock=threading.Lock();rt.code_log_dir=None;rt.cam_order=('head','left','right')
    rt.rtc_enabled=False;rt.rtc_count=0;rt.calls=0;rt.fast_joint=False
    rt.server_id='test-instance';rt.code_objective=objective;rt._instruction=None;rt.history_modes=[]
    class Policy:
        def __init__(self):
            self.model=SimpleNamespace(_rtc_targets=None,_rtc_weights=None,code_memory_enabled=False)
            self.code_cache={};self.step_count=0;self.episode_count=0;self._last_code=None
        def reset(self):
            self.code_cache={};self.step_count=0;self.episode_count+=1
        def _infer_action_chunk(self,obs,instruction):
            s=np.asarray(obs['joint_action']['vector'],np.float32)
            if objective=='joint' or 'program' not in self.code_cache:
                self.code_cache['program']=f'pose_x={float(s[14]):.3f}'
            self._last_code=self.code_cache['program']
            a=np.tile(np.r_[s[14:32],.25,.75],(32,1)).astype(np.float32)
            a[:,0]+=.05
            return rt.transform._action64(torch.from_numpy(a)[None],torch.from_numpy(s)[None])[0].numpy()
    rt.policy=Policy()
    return rt


def test_actual_runtime_absolute_action_and_camera_mapping():
    rt=runtime()
    a,_,_=rt.infer(request_body())
    assert a.shape==(32,20)
    assert np.allclose(a[:,0],.25) and np.allclose(a[:,18:20],[.25,.75])
    assert np.allclose(a[:,3:9],[1,0,0,0,1,0])
    assert rt.policy.model._rtc_targets is None


def test_nonjoint_code_refresh_and_task_reset():
    rt=runtime('ar')
    rt.infer(request_body(.2));old=rt.policy._last_code
    rt.infer(request_body(.4));assert rt.policy._last_code!=old
    rt.infer(request_body(.4,'open door'));assert rt.policy.episode_count==1
    rt.reset()
    assert rt._instruction is None and rt.policy._last_code is None
    assert rt.policy.model._last_code_canvas is None and not rt.policy.code_cache


def test_bad_rotation_and_logging_failure(tmp_path,monkeypatch):
    rt=runtime();body=request_body();body['state'][17:23]=[0]*6
    with pytest.raises(ValueError,match='rotation6D'):rt.infer(body)
    assert rt.calls==0
    rt.code_log_dir=tmp_path
    monkeypatch.setattr(Path,'write_text',lambda *a,**k:(_ for _ in ()).throw(OSError('disk full')))
    a,_,_=rt.infer(request_body())
    assert a.shape==(32,20) and rt.calls==1


def test_rtc_rejects_degenerate_committed_rotation():
    rt=runtime();rt.rtc_enabled=True
    with pytest.raises(ValueError,match='rotation6D'):
        rt._prepare_rtc(raw_state(),dict(committed=np.zeros((2,20)).tolist(),freeze=1))


def test_console_logging_failure_does_not_discard_action(monkeypatch):
    rt=runtime()
    def fail(*args,**kwargs):raise BrokenPipeError('log pipe closed')
    monkeypatch.setattr('builtins.print',fail)
    action,_,_=rt.infer(request_body())
    assert action.shape==(32,20) and rt.calls==1


def test_rtc_staging_cleared_on_inference_failure():
    rt=runtime()
    def prepare(*args):
        rt.policy.model._rtc_targets='staged';rt.policy.model._rtc_weights='staged'
        return True
    rt._prepare_rtc=prepare
    def fail(*args):raise RuntimeError('model failure')
    rt.policy._infer_action_chunk=fail
    with pytest.raises(RuntimeError,match='model failure'):rt.infer(request_body())
    assert rt.policy.model._rtc_targets is None and rt.policy.model._rtc_weights is None


def test_client_rejects_float_images_unknown_grip_and_invalid_url():
    with pytest.raises(ValueError,match='uint8'):client['encode_png_b64'](np.zeros((24,32,3),np.float32))
    obs=client['fake_obs']();obs['gripper_position']=['open','unknown']
    with pytest.raises(ValueError,match='gripper'):client['convert_obs'](obs)
    with pytest.raises(ValueError):client['normalize_http_base_url']('ftp://example.com')
    assert client['normalize_http_base_url']('https://example.com')=='https://example.com'


def test_client_server_direct_protocol():
    rt=runtime()
    c=client['RTCClient']('http://example.com','pick cake',True,H=32,s_min=20,use_async=False)
    paths=[]
    async def request(method,path,body=None):
        paths.append(path)
        if path=='/health':return dict(status='success',state_dim=34,action_dim=20,model_dim=64,chunk_size=32,rtc_enabled=False,server_id=rt.server_id)
        if path=='/reset':rt.reset();return dict(status='success',server_id=rt.server_id)
        a,ms,_=rt.infer(body);return dict(status='success',action=a.tolist(),latency=ms,server_id=rt.server_id)
    c._request=request
    obs=client['fake_obs']();obs['ee_pos_quat'][:3]=[.2,.1,.3]
    async def scenario():
        await c.connect()
        out=await c.guided_inference(obs)
        assert out['action'].shape==(32,20) and np.allclose(out['action'][:,0],.25)
        await c.reset_episode();assert c._server_prompt is None
        await c.stop()
    asyncio.run(asyncio.wait_for(scenario(),timeout=5))
    assert paths==['/health','/reset','/act','/reset']


def test_real_http_handler_metadata_and_reset(monkeypatch):
    rt=runtime();holder=[];ready=threading.Event()
    class TestServer(server.InferenceServer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);holder.append(self);ready.set()
    monkeypatch.setattr(server,'InferenceServer',TestServer)
    monkeypatch.setattr(server,'Custom34Runtime',lambda _:rt)
    monkeypatch.setattr(sys,'argv',['custom34_server.py','--ckpt','unused','--task','posttrain_custom',
                                  '--dataset','unused','--port','0'])
    thread=threading.Thread(target=server.main,daemon=True);thread.start()
    assert ready.wait(5)
    base=f'http://127.0.0.1:{holder[0].server_address[1]}'
    def req(path,body=None):
        r=Request(base+path,data=json.dumps(body).encode() if body is not None else None,
                  headers={'Content-Type':'application/json'} if body is not None else {})
        with urlopen(r,timeout=5) as response:return json.loads(response.read())
    try:
        health=req('/health')
        assert health['server_id']=='test-instance' and health['code_objective']=='joint'
        a=req('/act',request_body())
        assert a['server_id']==health['server_id'] and np.asarray(a['action']).shape==(32,20)
        reset=req('/reset',{});assert reset['episode_count']==1
        c=client['RTCClient'](base,'pick cake',True,H=32,s_min=20,use_async=False)
        obs=client['fake_obs']();obs['ee_pos_quat'][:3]=[.2,.1,.3]
        async def actual_http_client():
            await c.connect()
            out=await c.guided_inference(obs)
            assert np.allclose(out['action'][:,0],.25) and c._server_id==rt.server_id
            await c.reset_episode()
            await c.stop()
        asyncio.run(asyncio.wait_for(actual_http_client(),timeout=5))
    finally:
        holder[0].shutdown();thread.join(5)
    assert not thread.is_alive()
