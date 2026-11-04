import importlib.util
import io
import json
from pathlib import Path
import tokenize

import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('semantic_builder',ROOT/'scripts/rebuild_semantic_code_cache.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def code(*lines):
    return 'def policy_chunk(obs, state, robot, skills):\n'+''.join('    '+s+'\n' for s in lines)


def record(terminal='release'):
    semantic=("return robot.release(side='left')" if terminal=='release' else
              "return robot.place('cake', 'air fryer', side='left')")
    return dict(episode_index=0,task_index=0,active_arm='left',n_frames=10,events=[[2,'close'],[4,'open']],
                other_arm_events=[],spans=[
        dict(start=0,end=3,event='close',event_frame=2,code=code("return robot.grasp('handle', side='left')"),motion={}),
        dict(start=3,end=5,event='open',event_frame=4,code=code(semantic),motion={'warning':'preserve this fact'}),
        dict(start=5,end=6,event='post_open_idle',event_frame=None,code='',motion={}),
        dict(start=6,end=9,event='retreat',event_frame=None,
             code=code("return robot.move_arm(side='left', target_pos=[0,0,1], target_rpy=[0,0,0])"),motion={}),
        dict(start=9,end=10,event='post_retreat_idle',event_frame=None,code='',motion={})])


@pytest.mark.parametrize('terminal',['release','place'])
def test_merge_preserves_skill_and_frames(terminal):
    old=record(terminal)
    new,changes=builder.merge_record(old)
    assert len(changes)==1 and len(new['spans'])==3
    assert new['spans'][1]['start']==3 and new['spans'][1]['end']==9
    assert new['spans'][1]['code']==old['spans'][1]['code']
    assert new['spans'][1]['motion']['warning']=='preserve this fact'
    assert new['spans'][-1]['code']==''
    assert len(old['spans'])==5 # input untouched


def test_event_free_push_retreat_unchanged():
    r=record()
    r['task_index']=3
    r['spans']=[dict(start=0,end=6,event='task3_open_push',code=code("return robot.move_arm(side='left', target_pos=[0,0,1], target_rpy=[0,0,0])"),motion={}),
                dict(start=6,end=10,event='task3_retreat',code=code("return robot.move_arm(side='left', target_pos=[0,0,0], target_rpy=[0,0,0])"),motion={})]
    new,changes=builder.merge_record(r)
    assert new['spans']==r['spans'] and changes==[]


def test_reject_retreat_with_other_action():
    r=record();r['spans'][3]['code']=code("return robot.close_gripper(side='left')")
    with pytest.raises(ValueError,match='not pure'):builder.merge_record(r)


def test_cache_rebuild_roundtrip_and_corrupt_input(tmp_path):
    r=record();codes=[];ids=[]
    for span in r['spans']:
        codes.append(span['code']);ids.extend([len(codes)-1]*(span['end']-span['start']))
    cache=tmp_path/'source.pt';order=tmp_path/'order.json';audit=tmp_path/'audit.jsonl';info=tmp_path/'info.json'
    torch.save(dict(codes=codes,code_id_by_idx=torch.tensor(ids,dtype=torch.int32)),cache)
    order.write_text(json.dumps([dict(episode_index=0,task_index=0,offset=0,length=10)]))
    audit.write_text(json.dumps(r)+'\n');info.write_text(json.dumps(dict(total_frames=10,total_episodes=1)))
    summary=builder.rebuild(cache,order,audit,info,tmp_path/'out')
    assert summary['merged_retreats']==1 and summary['removed_move_arm_calls']==1
    d=torch.load(tmp_path/'out/policy_code_idx_custom.pt',weights_only=True)
    assert len(d['code_id_by_idx'])==10
    assert all(d['codes'][int(i)]==r['spans'][1]['code'] for i in d['code_id_by_idx'][3:9])
    kept=builder.rebuild(cache,order,audit,info,tmp_path/'kept',keep_retreat_episodes=[0])
    assert kept['merged_retreats']==0 and kept['removed_move_arm_calls']==0
    preserved=json.loads((tmp_path/'kept/gripper_code_audit.jsonl').read_text())
    assert preserved['spans']==r['spans']
    ids[0]=999
    torch.save(dict(codes=codes,code_id_by_idx=torch.tensor(ids)),cache)
    with pytest.raises(ValueError,match='Invalid global'):builder.rebuild(cache,order,audit,info,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_comment_removal_preserves_strings_and_arguments():
    original=code('# turn knob',"robot.rotate('knob#1', 83.52, side='left') # measured angle",
                  "return robot.release(side='left')")
    cleaned=builder.without_comments(original)
    assert "'knob#1'" in cleaned
    assert builder.calls(cleaned)==builder.calls(original)
    assert not any(t.type==tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(cleaned).readline))


def test_generated_cache_and_audit_have_no_comments(tmp_path):
    r=record();r['spans'][0]['code']=code('# approach handle',"return robot.grasp('handle#1', side='left')")
    codes=[s['code'] for s in r['spans']]
    ids=[i for i,s in enumerate(r['spans']) for _ in range(s['end']-s['start'])]
    paths=[tmp_path/name for name in ['source.pt','order.json','audit.jsonl','info.json']]
    torch.save(dict(codes=codes,code_id_by_idx=torch.tensor(ids)),paths[0])
    paths[1].write_text(json.dumps([dict(episode_index=0,task_index=0,offset=0,length=10)]))
    paths[2].write_text(json.dumps(r)+'\n');paths[3].write_text(json.dumps(dict(total_frames=10,total_episodes=1)))
    # Exception episodes keep their motion spans, but their code is still comment-free.
    report=builder.rebuild(*paths,tmp_path/'out',keep_retreat_episodes=[0])
    assert report['comment_spans_cleaned']==1
    d=torch.load(tmp_path/'out/policy_code_idx_custom.pt',weights_only=True)
    audit=json.loads((tmp_path/'out/gripper_code_audit.jsonl').read_text())
    for c in d['codes']+[s['code'] for s in audit['spans']]:
        assert not any(t.type==tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(c).readline))
