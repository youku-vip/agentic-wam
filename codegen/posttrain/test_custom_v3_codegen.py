"""Offline integration tests; no model endpoint or robot is called."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import pandas as pd
import pytest

REPO=Path(__file__).resolve().parents[2]
ENGINE=REPO/'codegen/pretrain/pipeline'
sys.path.insert(0,str(ENGINE))
sys.path.insert(0,str(REPO/'src'))
os.environ['CODEGEN_EMBODIMENT']='custom_v3'
from codegen.embodiments.custom_v3 import per_arm, reader, SPEC
from codegen.grounding.transitions import grip_transitions


@pytest.fixture
def dataset(tmp_path,monkeypatch):
    root=tmp_path/'data'
    for d in ['meta/episodes/chunk-000','data/chunk-000','videos/observation.images.head/chunk-000']:
        (root/d).mkdir(parents=True)
    info=dict(codebase_version='v3.0',fps=10,total_episodes=2,total_frames=140,
              data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
              video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
              features={'observation.state':{'dtype':'float32','shape':[34]},
                        'observation.images.head':{'dtype':'video'}})
    (root/'meta/info.json').write_text(json.dumps(info))
    pd.DataFrame({'task_index':[0,3],'task':['open air fryer','close air fryer']}).to_parquet(root/'meta/tasks.parquet')
    rows=[];metadata=[]
    for ep in range(2):
        for f in range(70):
            s=np.zeros(34);s[17:23]=s[26:32]=[1,0,0,0,1,0]
            s[14]=f*.004;s[32:34]=1
            if ep==0 and 20<=f<45:s[32]=0
            rows.append(dict(index=len(rows),episode_index=ep,frame_index=f,task_index=0 if ep==0 else 3,
                             **{'observation.state':s.tolist()}))
        metadata.append({'episode_index':ep,'data/chunk_index':0,'data/file_index':0,
                         'dataset_from_index':ep*70,'dataset_to_index':(ep+1)*70,'length':70,
                         'videos/observation.images.head/chunk_index':0,
                         'videos/observation.images.head/file_index':0,
                         'videos/observation.images.head/from_timestamp':ep*7.})
    pd.DataFrame(rows).to_parquet(root/'data/chunk-000/file-000.parquet')
    pd.DataFrame(metadata).to_parquet(root/'meta/episodes/chunk-000/file-000.parquet')
    with av.open(str(root/'videos/observation.images.head/chunk-000/file-000.mp4'),'w') as c:
        stream=c.add_stream('mpeg4',rate=10);stream.width=64;stream.height=48;stream.pix_fmt='yuv420p'
        for i in range(140):
            frame=np.zeros((48,64,3),dtype=np.uint8);frame[:]=[255,0,0] if i<70 else [0,255,0]
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    work=tmp_path/'work'
    monkeypatch.setenv('CUSTOM_V3_DATASET',str(root));monkeypatch.setenv('CUSTOM_V3_WORK',str(work))
    monkeypatch.setenv('CUSTOM_V3_OPEN_IS_HIGH','1')
    reader.cache_clear()
    yield root,work
    reader.cache_clear()


def run_cli(root,work,*extra):
    subprocess.run([sys.executable,str(REPO/'scripts/custom_v3_codegen.py'),'--dataset',str(root),
                    '--work',str(work),*extra],check=True,capture_output=True,text=True)


def test_detector_constant_and_binary():
    st=np.zeros((70,34));st[:,32:34]=1;st[:,14]=np.arange(70)*.004
    assert per_arm(st,10)['left']['grip_events']==[]
    st[20:45,32]=0
    ev=per_arm(st,10)['left']['grip_events']
    assert [k for _,k in ev]==['close','open']
    assert abs(ev[0][0]-20)<=3 and abs(ev[1][0]-45)<=3
    # Default parameters preserve original callers' exact behavior.
    assert grip_transitions(st[:,32])==grip_transitions(st[:,32],smooth_frames=5,min_run=4,velocity_epsilon=.01)


def test_original_api_library_and_prompt():
    from codegen.api.active import API_DOC
    from codegen.api.bimanual import API_DOC as original
    from codegen.actor import prompts, prompts_yam
    from codegen.skill_paths import library_dir
    assert API_DOC==original and prompts.CODER_SYS==prompts_yam.CODER_SYS
    assert library_dir('custom_v3')==library_dir('yam')
    assert any(SPEC.skills_dir.glob('*/skill.py'))
    from codegen.program.effects import _primitive
    assert _primitive('close')=='close_gripper'


def test_boundary_report_and_dryrun_rejection(dataset):
    root,work=dataset
    run_cli(root,work,'subtasks','--episodes','0','1')
    assert not (work/'subtasks.json').exists()
    report=json.loads((work/'boundaries_report.json').read_text())
    assert report['0']['events']['left'] and report['1']['events']['left']==[]
    assert sum(b-a for a,b in report['0']['spans'])==70
    assert (work/'units.txt').read_text()=='custom:0\ncustom:1\n'


def test_original_episode_shared_video(dataset):
    root,work=dataset
    run_cli(root,work,'subtasks','--all')
    report=json.loads((work/'boundaries_report.json').read_text())
    # Synthetic named sidecars for adapter test only; no VLM quality claim.
    for rec in report.values():rec['dry_run']=False
    (work/'subtasks.json').write_text(json.dumps(report))
    ep=SPEC.load_episode('custom',1)
    try:
        assert ep.fps==10 and len(ep.clip)==70 and ep.per_arm['left']['grip_events']==[]
        image=ep.clip[0]
        assert image[:,:,1].mean()>200 and image[:,:,0].mean()<30
        assert ep.meta['video_start_frame']==70
    finally:ep.clip.close()


def test_cache_requires_real_outputs(dataset,tmp_path):
    root,work=dataset
    run_cli(root,work,'subtasks','--all')
    report=json.loads((work/'boundaries_report.json').read_text())
    for rec in report.values():rec['dry_run']=False
    (work/'subtasks.json').write_text(json.dumps(report))
    with pytest.raises(subprocess.CalledProcessError):
        run_cli(root,work,'cache','--all','--tag','missing','--model-short','test',
                '--out-dir',str(tmp_path/'cache'))
    assert not (tmp_path/'cache/policy_code_idx_custom.pt').exists()


def test_cache_gate_and_trial_global_order(dataset,tmp_path,monkeypatch):
    import torch
    from codegen.pipeline import gate
    root,work=dataset
    run_cli(root,work,'subtasks','--all')
    report=json.loads((work/'boundaries_report.json').read_text())
    for rec in report.values():rec['dry_run']=False
    (work/'subtasks.json').write_text(json.dumps(report))
    run=tmp_path/'run';run.mkdir()
    program='def policy_chunk(obs, state, robot, skills):\n    return robot.close_gripper(side="left")\n'
    (run/'program.py').write_text(program)
    (run/'lines.json').write_text(json.dumps(dict(frame_src_block=[[1,2]]*70,grip_pins=[])))
    trace=dict(episode_meta=SPEC.load_episode('custom',0).meta,
               slices=[dict(slice=0,label='grasp',validation=dict(ok=True,repairs=0),
                            grip_reconcile=dict(matched=True,observed_grippers=['close_left'],coder_grippers=['close_left']))])
    (run/'trace.json').write_text(json.dumps(trace))
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    monkeypatch.setattr(SPEC,'run_dir',lambda *args,**kwargs:run)
    spec=importlib.util.spec_from_file_location('custom_cli',REPO/'scripts/custom_v3_codegen.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    out=tmp_path/'cache'
    argv=[str(REPO/'scripts/custom_v3_codegen.py'),'--dataset',str(root),'--work',str(work),
          'cache','--episodes','0','--tag','test','--model-short','test','--out-dir',str(out)]
    monkeypatch.setattr(sys,'argv',argv)
    cli.main()
    data=torch.load(out/'policy_code_idx_custom.pt',weights_only=True)
    assert len(data['code_id_by_idx'])==140
    assert all(data['codes'][int(i)]==program for i in data['code_id_by_idx'][:70])
    assert all(data['codes'][int(i)]=='' for i in data['code_id_by_idx'][70:])
    assert json.loads((out/'episode_order.json').read_text())[1]['offset']==70
    # Even a syntactically valid, reconciled program must fail when provenance is stale.
    trace['episode_meta']['codegen_signature']='stale'
    (run/'trace.json').write_text(json.dumps(trace))
    monkeypatch.setattr(sys,'argv',argv[:-1]+[str(tmp_path/'stale-cache')])
    with pytest.raises(ValueError,match='different data/settings'):cli.main()
    assert not (tmp_path/'stale-cache/policy_code_idx_custom.pt').exists()


def test_original_actor_pipeline_offline(dataset,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from codegen.pipeline import episode as actor, gate
    root,work=dataset
    run_cli(root,work,'subtasks','--episodes','0')
    report=json.loads((work/'boundaries_report.json').read_text())
    rec=report['0'];rec['dry_run']=False
    rec['sub_id']=[1]*70;rec['text']={'1':'open air fryer'};rec['spans']=[[0,70]]
    (work/'subtasks.json').write_text(json.dumps(report))
    run=tmp_path/'actor'
    monkeypatch.setattr(SPEC,'run_dir',lambda *args,**kwargs:run)
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    monkeypatch.setattr(actor,'load_key',lambda: None)
    calls=[]
    def fake_model(*args,**kwargs):
        calls.append(args)
        return "robot.grasp('air fryer door handle', side='left')\nrobot.release(side='left')"
    monkeypatch.setattr(actor,'call_model',fake_model)
    args=SimpleNamespace(task='custom',episode=0,tag='test',coder_model='test',frame_gap=13,
                         max_slices=0,no_cache=True,repair_retries=0,vanilla=False)
    assert actor.run_episode(args)==0 and len(calls)==1
    result=gate.gate_task('custom',run)
    assert result['reconciled'] and result['valid_slices']==1 and result['observed_events']==2
    trace=json.loads((run/'trace.json').read_text())
    assert trace['prompt_events'] and trace['episode_meta']['codegen_signature']
