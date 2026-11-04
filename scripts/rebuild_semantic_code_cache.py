#!/usr/bin/env python3
"""Route B: merge measured retreat into existing release/place skill spans.

Read-only inputs; no VLM, robot execution, coordinate re-estimation or video I/O.
Requires the explicit per-episode audit produced by the custom gripper builder.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch


def calls(code):
    if not code:
        return []
    tree=ast.parse(code)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('Expected one policy_chunk function, not arbitrary executable code')
    fn=tree.body[0]
    if fn.name!='policy_chunk' or fn.decorator_list:
        raise ValueError('Unsupported policy function')
    result=[]
    for node in fn.body:
        if not isinstance(node,(ast.Expr,ast.Return)) or not isinstance(node.value,ast.Call):
            raise ValueError('Only straight-line robot API calls are supported')
        c=node.value
        if not isinstance(c.func,ast.Attribute) or not isinstance(c.func.value,ast.Name) or c.func.value.id!='robot':
            raise ValueError('Unsupported API call; requires manual review')
        # literal_eval reads arguments without evaluating arbitrary expressions.
        args=[ast.literal_eval(v) for v in c.args]
        kwargs={k.arg:ast.literal_eval(k.value) for k in c.keywords}
        if None in kwargs:
            raise ValueError('Expanded kwargs unsupported')
        result.append((c.func.attr,args,kwargs))
    return result


def validate_inputs(data,order,records,info):
    codes,ids=data['codes'],data['code_id_by_idx']
    if not isinstance(codes,list) or not all(isinstance(c,str) for c in codes):
        raise ValueError('Invalid code pool')
    if ids.ndim!=1 or ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('Invalid index tensor')
    if len(ids)!=info['total_frames'] or not len(ids) or int(ids.min())<0 or int(ids.max())>=len(codes):
        raise ValueError('Invalid global cache indices/count')
    if len(order)!=len(records) or len(order)!=info['total_episodes']:
        raise ValueError('Episode count mismatch')
    for code in codes:
        calls(code)
    end=0;seen=set()
    for o,r in zip(order,records):
        ep=r['episode_index']
        if ep in seen or ep!=o['episode_index'] or r['task_index']!=o['task_index']:
            raise ValueError('Episode/task mismatch or duplicate episode')
        seen.add(ep)
        if o['offset']!=end or o['length']!=r['n_frames']:
            raise ValueError('Order offset/length mismatch')
        local=0
        for s in r['spans']:
            a,b=s['start'],s['end']
            if a!=local or not a<b<=r['n_frames']:
                raise ValueError('Span gap/overlap')
            values=ids[end+a:end+b].unique().tolist()
            if len(values)!=1 or codes[values[0]]!=s['code']:
                raise ValueError(f'Episode {ep}: audit/cache mismatch')
            local=b
        if local!=r['n_frames']:
            raise ValueError('Incomplete episode coverage')
        end+=local
    if end!=len(ids):
        raise ValueError('Incomplete global coverage')


def merge_record(record):
    rec=deepcopy(record)
    old=rec['spans'];new=[];changes=[];i=0
    while i<len(old):
        span=deepcopy(old[i])
        seq=calls(span['code'])
        terminal=seq[-1][0] if seq else None
        if terminal not in ('release','place'):
            new.append(span);i+=1;continue
        if sum(name in ('release','place') for name,_,_ in seq)!=1:
            raise ValueError('Ambiguous multiple release/place calls in one span')
        side=seq[-1][2].get('side')
        if side not in ('left','right') or side!=rec['active_arm']:
            raise ValueError('Terminal semantic skill arm mismatch')
        j=i+1
        if j<len(old) and old[j]['event']=='post_open_idle' and not old[j]['code']:
            j+=1
        if j>=len(old) or old[j]['event']!='retreat':
            new.append(span);i+=1;continue
        retreat=old[j]
        movement=calls(retreat['code'])
        if not movement or any(name!='move_arm' or kw.get('side')!=side for name,_,kw in movement):
            raise ValueError('Retreat is not pure same-arm movement; refusing to discard it')
        if any(old[k]['end']!=old[k+1]['start'] for k in range(i,j)):
            raise ValueError('Noncontiguous semantic skill/retreat')
        original_end=span['end']
        span['end']=retreat['end']
        span['source_event']=span['event']
        span['event']=terminal+'_with_retreat'
        facts=span.setdefault('motion',{})
        facts['semantic_retreat_merge']=dict(skill=terminal,original_skill_end=original_end,
             retreat_start=retreat['start'],retreat_end=retreat['end'],
             removed_move_arm_calls=len(movement),retreat_motion=retreat.get('motion',{}),
             note='Measured retreat frames now belong to release/place; no separate move_arm label')
        changes.append(dict(episode_index=rec['episode_index'],task_index=rec['task_index'],skill=terminal,
                            start=span['start'],end=span['end'],retreat_start=retreat['start'],
                            removed_move_arm_calls=len(movement),absorbed_idle_frames=retreat['start']-original_end))
        new.append(span);i=j+1
    rec['spans']=new
    rec['semantic_template_version']='release-place-retreat-v1'
    return rec,changes


def rebuild(cache,order_path,audit_path,info_path,out,keep_retreat_episodes=()):
    paths=[Path(p).expanduser().resolve() for p in (cache,order_path,audit_path,info_path)]
    cache,order_path,audit_path,info_path=paths
    # Safe loader: do not execute pickle globals from an uploaded cache.
    data=torch.load(cache,map_location='cpu',weights_only=True)
    order=json.loads(order_path.read_text(encoding='utf-8'))
    records=[json.loads(l) for l in audit_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    info=json.loads(info_path.read_text(encoding='utf-8'))
    validate_inputs(data,order,records,info)
    keep=set(keep_retreat_episodes)
    if keep-{r['episode_index'] for r in records}:
        raise ValueError('Unknown keep-retreat episode IDs')
    rebuilt=[];changes=[]
    for record in records:
        rec,delta=(deepcopy(record),[]) if record['episode_index'] in keep else merge_record(record)
        rebuilt.append(rec);changes.extend(delta)
    codes=[''];lookup={'':0};ids=[]
    before=Counter();after=Counter();tasks={}
    for old,new in zip(records,rebuilt):
        for s in old['spans']:
            before.update(name for name,_,_ in calls(s['code']))
        for s in new['spans']:
            after.update(name for name,_,_ in calls(s['code']))
            c=s['code']
            if c not in lookup:lookup[c]=len(codes);codes.append(c)
            ids.extend([lookup[c]]*(s['end']-s['start']))
        t=tasks.setdefault(str(new['task_index']),dict(episodes=0,frames=0,labeled_frames=0))
        t['episodes']+=1;t['frames']+=new['n_frames']
        t['labeled_frames']+=sum(s['end']-s['start'] for s in new['spans'] if s['code'])
    result=dict(codes=codes,code_id_by_idx=torch.tensor(ids,dtype=torch.int32))
    validate_inputs(result,order,rebuilt,info)
    # Only move_arm calls may be removed. Rotation estimates and semantic calls must survive.
    if any(before[k]!=after[k] for k in before if k!='move_arm'):
        raise AssertionError('Unexpected change to semantic calls')
    report=dict(version='release-place-retreat-v1',episodes=len(records),frames=len(ids),
                keep_retreat_episodes=sorted(keep),
                merged_retreats=len(changes),removed_move_arm_calls=before['move_arm']-after['move_arm'],
                calls_before=dict(before),calls_after=dict(after),by_task=tasks,changes=changes,
                verified='Full input/output cache-audit-order consistency',
                not_verified=['Video/physical task correctness','Real loader enumeration','Tokenizer limits',
                              'Vertical retreat compatibility of release with real measured trajectory',
                              'Pick-to-lift boundary; existing stage boundaries retained'],
                input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    out=Path(out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be new or empty; original caches never overwritten')
    out.mkdir(parents=True,exist_ok=True)
    torch.save(result,out/'policy_code_idx_custom.pt')
    (out/'episode_order.json').write_text(json.dumps(order,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'gripper_code_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rebuilt),encoding='utf-8')
    (out/'semantic_merge_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('changes','input_sha256')},ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','order','audit','info','out-dir']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--keep-retreat-episodes',type=int,nargs='+',default=[],
                   help='Preserve separate movement when it is not covered by the semantic skill')
    a=p.parse_args()
    rebuild(a.cache,a.order,a.audit,a.info,a.out_dir,a.keep_retreat_episodes)
