#!/usr/bin/env python3
"""Custom v3 bridge to the ORIGINAL VLM actor, YAM API, library and gate.

subtasks defaults to a free boundary report. --label calls the configured VLM.
cache consumes reconciled original-engine outputs; it never fabricates programs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
ENGINE=REPO/'codegen/pretrain/pipeline'
sys.path.insert(0,str(ENGINE))
sys.path.insert(0,str(REPO/'codegen/posttrain'))


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True,help='Sidecar directory outside raw data')
    p.add_argument('--open-is-low',action='store_true',help='Default: high gripper value means open')
    sub=p.add_subparsers(dest='stage',required=True)
    st=sub.add_parser('subtasks')
    st.add_argument('--label',action='store_true',help='PAID: name sensor spans with original VLM prompt')
    st.add_argument('--min-span-s',type=float,default=1.)
    st.add_argument('--max-span-s',type=float,default=8.)
    st.add_argument('--frames-per-span',type=int,default=4)
    ca=sub.add_parser('cache')
    ca.add_argument('--tag',required=True)
    ca.add_argument('--model-short',required=True,help='Exact model directory segment from original engine')
    ca.add_argument('--out-dir',type=Path,required=True)
    for parser in [st,ca]:
        g=parser.add_mutually_exclusive_group(required=True)
        g.add_argument('--episodes',nargs='+',type=int)
        g.add_argument('--all',action='store_true')
    a=p.parse_args()
    os.environ['CODEGEN_EMBODIMENT']='custom_v3'
    os.environ['CUSTOM_V3_DATASET']=str(a.dataset.expanduser().resolve())
    work=a.work.expanduser().resolve()
    os.environ['CUSTOM_V3_WORK']=str(work)
    os.environ['CUSTOM_V3_OPEN_IS_HIGH']='0' if a.open_is_low else '1'
    from codegen.embodiments.custom_v3 import reader,per_arm,SPEC
    rd=reader()
    if rd.episode_ids!=list(range(rd.total_episodes)):
        raise ValueError('Current training loader expects contiguous episode IDs from zero')
    chosen=set(rd.episode_ids if a.all else a.episodes)
    if chosen-set(rd.episode_ids):
        raise ValueError('Requested episodes absent from dataset')
    work.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(dict(root=str(rd.root),info=rd.info,
                         episodes=rd.episode_table.to_json(),open_is_low=a.open_is_low),sort_keys=True).encode()).hexdigest()
    if a.stage=='subtasks':
        if not 0<a.min_span_s<=a.max_span_s or a.frames_per_span<1:
            p.error('Require 0 < min-span-s <= max-span-s and frames-per-span >=1')
        import annotate_subtasks_gripper as original
        # Original segmentation/naming, with actual encoded fps instead of hardcoded 30Hz.
        original.MIN_SPAN_S=a.min_span_s
        original.MAX_SPAN_S=a.max_span_s
        options=dict(dataset_signature=signature,min_span_s=a.min_span_s,max_span_s=a.max_span_s,
                     frames_per_span=a.frames_per_span,detector='fps-scaled-yam-velocity-v1')
        options['adapter_sha256']=hashlib.sha256((ENGINE/'codegen/embodiments/custom_v3.py').read_bytes()).hexdigest()
        sidecar=work/'subtasks.json'
        existing=json.loads(sidecar.read_text()) if sidecar.exists() else {}
        report={}
        if a.label:
            from codegen.actor import client,media
            if not client.VLM_URL or '<' in client.VLM_URL:
                raise ValueError('Configure VLM_URL and CODER_MODEL before --label; no key values in commands/logs')
            key=client.load_key()
        for ep in sorted(chosen):
            if a.label and str(ep) in existing:
                if existing[str(ep)].get('generation_options')!=options:
                    raise ValueError(f'Episode {ep}: existing labels use different settings; choose a new --work')
                print(f'episode {ep}: existing labels retained',flush=True)
                continue
            df=rd.read_episode(ep,columns=['observation.state','task_index'])
            ti=set(map(int,df['task_index']))
            if len(ti)!=1:raise ValueError('Multiple tasks in episode')
            instruction=rd.tasks[ti.pop()]
            states=np.stack(df['observation.state']).astype(np.float64)
            arms=per_arm(states,rd.fps)
            events={s:v['grip_events'] for s,v in arms.items()}
            spans=original.spans_from_events(events,len(df),rd.fps)
            text={};sub_id=np.zeros(len(df),dtype=np.int32)
            clip=None
            if a.label:
                from codegen.data.frames import LazyClip
                video,first,length=rd.video_frame_window(ep,'observation.images.head')
                clip=LazyClip(video,first_frame=first,length=length)
            try:
                for k,(start,end) in enumerate(spans,1):
                    facts=original.span_facts(events,start,end,rd.fps)
                    sub_id[start:end]=k
                    if a.label:
                        picks=np.linspace(start,end-1,a.frames_per_span).astype(int)
                        content=[media._txt(original.PROMPT.format(instruction=instruction,k=k,n=len(spans),
                                     t0=start/rd.fps,t1=end/rd.fps,dur=len(df)/rd.fps,facts=facts))]
                        content += [media._img(media.frame_b64(media._to_pil(clip[int(f)]))) for f in picks]
                        reply=client.call_model([dict(role='user',content=content)],key,max_tokens=1024,temperature=.1)
                        if not reply.strip():raise ValueError('Empty VLM subtask label')
                        text[str(k)]=reply.strip().strip('"').splitlines()[-1].strip().strip('"').lower()
                    else:
                        text[str(k)]=f'[boundary only] {facts}'
            finally:
                if clip is not None:clip.close()
            # Match original behavior: merge adjacent identical VLM names.
            if a.label:
                merged=[];names={}
                for k,(start,end) in enumerate(spans,1):
                    if merged and names[str(len(merged))]==text[str(k)]:
                        merged[-1][1]=end
                    else:
                        merged.append([start,end]);names[str(len(merged))]=text[str(k)]
                spans,text=merged,names
                for k,(start,end) in enumerate(spans,1):sub_id[start:end]=k
            rec=dict(instruction=instruction,sub_id=sub_id.tolist(),text=text,spans=spans,
                     events={s:[list(e) for e in v] for s,v in events.items()},dry_run=not a.label,
                     generation_options=options, source_state_sha256=hashlib.sha256(states.tobytes()).hexdigest(), dropped_events={s:v['dropped'] for s,v in arms.items()})
            report[str(ep)]=rec
            if a.label:
                existing[str(ep)]=rec
                atomic_json(sidecar,existing)
            print(f'episode {ep}: T={len(df)} spans={len(spans)} events={events} '+('VLM named' if a.label else 'DRY RUN'),flush=True)
        atomic_json(work/('naming_report.json' if a.label else 'boundaries_report.json'),report)
        (work/'units.txt').write_text(''.join(f'custom:{i}\n' for i in sorted(chosen)))
        print('Next: review detector boundaries before VLM labeling or batch code generation.')
    else:
        import torch
        from codegen.pipeline.gate import gate_task
        from codegen.pipeline import artifacts
        from build_flat_code_cache import episode_frame_codes
        out=a.out_dir.expanduser().resolve()
        if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be new/empty')
        codes=[''];lookup={'':0};ids=[];order=[];gates=[];nonempty=0
        for ep in rd.episode_ids:
            df=rd.read_episode(ep,columns=['index','task_index'])
            offset=len(ids);n=len(df)
            if not np.array_equal(df['index'].to_numpy(),np.arange(offset,offset+n)):
                raise ValueError(f'Episode {ep}: physical global index differs from training enumeration')
            frame_codes=['']*n
            if ep in chosen:
                rec=json.loads((work/'subtasks.json').read_text())[str(ep)]
                if rec.get('dry_run',True) or rec.get('generation_options',{}).get('dataset_signature')!=signature:
                    raise ValueError('Subtasks are stale or dry-run only')
                run=SPEC.run_dir('custom',a.tag,a.model_short,episode_id=f'episode_{ep:03d}')
                gate=gate_task('custom',run)
                if (not gate.get('reconciled') or not gate.get('compiles') or not gate.get('slices')
                        or gate.get('valid_slices') != gate.get('slices')):
                    raise ValueError(f'Episode {ep}: original reconciliation gate failed: {gate}')
                trace=json.loads((run/artifacts.TRACE).read_text())
                current=SPEC.load_episode('custom',ep)
                if trace.get('episode_meta',{}).get('codegen_signature') != current.meta['codegen_signature']:
                    raise ValueError('Original engine result belongs to different data/settings; regenerate under a new tag')
                gates.append(gate)
                frame_codes=episode_frame_codes(run/artifacts.PROGRAM,artifacts.lines_path(run))
                if len(frame_codes)!=n:raise ValueError('Original line map length mismatch; no padding/trimming allowed')
            tasks=set(map(int,df['task_index']))
            if len(tasks)!=1:raise ValueError('Multiple tasks per episode')
            order.append(dict(trajectory_id=f'ep{ep:06d}',episode_index=ep,task_index=tasks.pop(),offset=offset,length=n))
            for c in frame_codes:
                if not c.strip():c=''
                if c not in lookup:lookup[c]=len(codes);codes.append(c)
                ids.append(lookup[c]);nonempty+=bool(c)
        if len(ids)!=rd.total_frames:raise ValueError('Frame count mismatch')
        out.mkdir(parents=True,exist_ok=True)
        torch.save(dict(codes=codes,code_id_by_idx=torch.tensor(ids,dtype=torch.int32)),out/'policy_code_idx_custom.pt')
        atomic_json(out/'episode_order.json',order)
        atomic_json(out/'gate_report.json',gates)
        atomic_json(out/'generation_manifest.json',dict(mode='full' if a.all else 'trial_masked',
            dataset_signature=signature,frames=len(ids),nonempty_frames=nonempty,selected_episodes=sorted(chosen),
            source='original-engine-custom_v3',tag=a.tag,model_short=a.model_short,
            note='Unselected/ungrounded frames remain empty and masked; programs are never executed'))
        print(f'CACHE DONE: {len(ids)} frames, {nonempty} nonempty; {out}')


if __name__=='__main__':
    main()
