#!/usr/bin/env python3
"""No real robot, no CUDA. Mock HTTP-server contract and RTC clock alignment."""
from __future__ import annotations
import ast
import asyncio
import base64
from collections import deque, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from PIL import Image
import numpy as np

class DummyLogger:
    def warning(self,*a):pass
    def error(self,*a):pass
    def info(self,*a):pass
logger=DummyLogger()
class DummyTrans:
    def euler_angle_to_rot6d(self, e):
        return np.array([1,0,0,0,1,0],np.float32)
trans_utils=DummyTrans()

SRC=Path(__file__).with_name('rtc_client_agenticwam.py').read_text()
tree=ast.parse(SRC)
names={'normalize_http_base_url','convert_obs','encode_png_b64','make_zero_observation','RTCClient',
       'validate_rot6d', 'deployment_control_options', 'SharedPrompt'}
keep=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name in names]
exec(compile(ast.Module(body=keep,type_ignores=[]),'adapter','exec'),globals())

class FakeServer(BaseHTTPRequestHandler):
    stats=[]
    delay_s=0.025
    def log_message(self,*a):pass
    def resp(self,code,payload):
        body=json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        assert self.path=='/health'
        self.resp(200,{'status':'success','state_dim':34,'action_dim':20,'model_dim':64,
                       'chunk_size':32,'rtc_enabled':True})
    def do_POST(self):
        n=int(self.headers.get('Content-Length','0'))
        data=json.loads(self.rfile.read(n))
        if self.path=='/reset':
            self.resp(200,{'status':'success','message':'reset','action':None,'latency':1})
            return
        assert self.path=='/act',self.path
        assert set(data)=={'instruction','state','images','request_id'} or set(data)=={'instruction','state','images','request_id','committed','freeze'}
        assert len(data['state'])==34
        assert set(data['images'])=={'head','left','right'}
        for k,v in data['images'].items():
            raw=base64.b64decode(v)
            with Image.open(io.BytesIO(raw)) as im:
                assert im.size==(32,24) and im.mode=='RGB'
        FakeServer.stats.append(data)
        time.sleep(FakeServer.delay_s)
        chunk=np.zeros((32,20),np.float32)
        chunk[:,3:9]=[1,0,0,0,1,0]
        chunk[:,12:18]=[1,0,0,0,1,0]
        chunk[:,0]=np.arange(32)+10
        if 'committed' in data:
            c=np.asarray(data['committed'],np.float32)
            chunk[:len(c)]=c
        self.resp(200,{'status':'success','action':chunk.tolist(),'latency':self.delay_s*1000,'message':''})

def fake_obs():
    rgb=np.zeros((24,32,3),np.uint8)
    rgb[:,:,:]=[17,53,99]
    return dict(joint_positions=np.ones(14,dtype=np.float32).tolist(),
        ee_pos_quat=np.zeros(12,dtype=np.float32).tolist(),
        gripper_position=['open','close'],
        foo_head_rgb=rgb, foo_left_rgb=rgb,foo_right_rgb=rgb)

async def main():
    srv=ThreadingHTTPServer(('127.0.0.1',0),FakeServer)
    th=threading.Thread(target=srv.serve_forever,daemon=True)
    th.start()
    url=f'ws://127.0.0.1:{srv.server_address[1]}/custom'
    try:
        c=RTCClient(server_uri=url,prompt='open the door',infer_type=True,
                    H=32,s_min=4,b=4,action_dim=20,use_rtc=True,use_async=True,control_hz=100)
        assert c.server_uri.startswith('http://')
        await c.connect()
        o=fake_obs()
        initial=await c.guided_inference(o)
        assert initial['action'].shape==(32,20)
        assert FakeServer.stats[0]['state'][0]==1
        assert FakeServer.stats[0]['state'][14]==0
        c.A_cur=initial['action']
        await c.get_action()
        await c.get_action()
        await c.get_action()
        await c.get_action(o) # observation captured at t=3 before row 3 was executed
        # Planner activated at t=4 from snapshot index s=3.
        for _ in range(6):
            await asyncio.sleep(0.005)
            await c.get_action()
        await asyncio.sleep(0.08)
        assert len(FakeServer.stats)>=2,FakeServer.stats
        guided=FakeServer.stats[1]
        assert 'committed' in guided, 'RTC guidance not sent'
        assert guided['freeze']>0
        assert len(guided['committed'])>guided['freeze'], 'soft tail expected'
        assert np.isclose(guided['committed'][0][0],initial['action'][3,0]), (guided['committed'][0][0],initial['action'][3,0])
        print('PASS: HTTP /health /reset /act; raw34; PNG; action [32,20]; RTC hard+soft; capture-time alignment')
        await c.stop()
        # Safety: reject wrong H without ever calling POST /act
        wrong=RTCClient(server_uri=c.server_uri,prompt='test',infer_type=True,H=50,s_min=8,
                         action_dim=20,use_async=False)
        try: await wrong.connect()
        except ValueError as exc:
            assert 'chunk_size' in str(exc)
            print('PASS: incompatible server horizon rejected at handshake')
        else:
            raise AssertionError('mismatched H accepted')
    finally:
        srv.shutdown()
        srv.server_close()

if __name__=='__main__': asyncio.run(main())
