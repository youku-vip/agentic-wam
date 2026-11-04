#!/usr/bin/env python3
"""CPU-only custom v3 relative-statistics verification and raw-loader benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agenticwam.datasets.lerobot.v3.episode_reader import V3EpisodeReader
from agenticwam.datasets.lerobot.transforms.custom_unified_eef import CustomUnifiedEEFTransform,_rot6d_to_R
from agenticwam.datasets.lerobot.shared_stats import PooledStats


class ProjectStats:
    """Match BaseLerobotDataset.get_dataset_stats: episode weights and q envelopes."""
    def __init__(self):
        self.episodes=[]

    def update(self,values):
        x=np.asarray(values,dtype=np.float64)
        if x.ndim==2:x=x[:,None,:]
        if len(x)<2 or not np.isfinite(x).all():
            raise ValueError('Original project variance requires >=2 samples per episode')
        self.episodes.append(dict(min=x.min(0),max=x.max(0),mean=x.mean(0),var=x.var(0,ddof=1),
                                  q01=np.quantile(x,.01,axis=0),q99=np.quantile(x,.99,axis=0)))

    def finish(self):
        if not self.episodes:raise ValueError('No episode statistics')
        e={k:np.stack([r[k] for r in self.episodes]) for k in self.episodes[0]}
        mean=e['mean'].mean(0);global_mean=e['mean'].mean((0,1))
        step=dict(min=e['min'].min(0),max=e['max'].max(0),q01=e['q01'].min(0),q99=e['q99'].max(0),
                  mean=mean,std=np.sqrt((e['var']+(e['mean']-mean)**2).mean(0)))
        glob=dict(min=step['min'].min(0),max=step['max'].max(0),q01=step['q01'].min(0),q99=step['q99'].max(0),
                  mean=global_mean,std=np.sqrt((e['var']+(e['mean']-global_mean)**2).mean((0,1))))
        return {'default':{**{f'stepwise_{k}':v.tolist() for k,v in step.items()},
                           **{f'global_{k}':v.tolist() for k,v in glob.items()}}}


def calculate(root,*,stride=5,padding='repeat',max_samples=20000,episodes=None,existing=None,aggregation='pooled'):
    if aggregation=='project' and (stride!=1 or padding!='repeat'):
        raise ValueError('Project policy requires --stride 1 --padding repeat')
    rd=V3EpisodeReader(root)
    selected=rd.episode_ids if episodes is None else list(dict.fromkeys(episodes))
    tf=CustomUnifiedEEFTransform(['default'])
    state_acc=ProjectStats() if aggregation=='project' else PooledStats(max_samples=max_samples,seed=42)
    action_acc=ProjectStats() if aggregation=='project' else PooledStats(max_samples=max_samples,seed=42)
    frames=windows=0;error=0.;rotation_error=0.;saturated=0;norm_count=0
    norm=None
    if existing:
        from agenticwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json,LinearNormalizer
        norm=LinearNormalizer({'state':[{'key':'default'}],'action':[{'key':'default'}]},True,
                              'q01/q99',None,load_dataset_stats_from_json(existing))
        for field in ['state','action']:
            n=norm.normalizers[field]['default']
            if not torch.isfinite(n.scale).all() or not torch.isfinite(n.offset).all():
                raise ValueError('Existing stats produce nonfinite normalizer parameters')
    for ep in selected:
        df=rd.read_episode(ep,columns=['observation.state','action'])
        st=np.stack(df['observation.state']).astype(np.float32)
        ac=np.stack(df['action']).astype(np.float32)
        if st.shape!=(len(df),34) or ac.shape!=(len(df),20) or not np.isfinite(st).all() or not np.isfinite(ac).all():
            raise ValueError(f'Episode {ep}: invalid custom34/action20 data')
        for d,offsets in [(st,[17,26]),(ac,[3,12])]:
            for offset in offsets:
                x=d[:,offset:offset+3];y=d[:,offset+3:offset+6]
                if (np.linalg.norm(x,axis=1)<1e-6).any() or (np.linalg.norm(np.cross(x,y),axis=1)<1e-6).any():
                    raise ValueError(f'Episode {ep}: degenerate rotation6D')
        state_acc.update(tf._state64(torch.from_numpy(st)).numpy())
        frames+=len(st)
        stop=len(st) if padding=='repeat' else max(0,len(st)-31)
        starts=np.arange(0,stop,stride)
        episode_actions=[]
        for offset in range(0,len(starts),128):
            cur=starts[offset:offset+128]
            indices=np.minimum(cur[:,None]+np.arange(32)[None],len(st)-1)
            raw=torch.from_numpy(ac[indices])
            batch=tf.forward({'state':{'default':torch.from_numpy(st[cur,None].copy())},'action':{'default':raw.clone()}})
            rel=batch['action']['default']
            if aggregation=='project':episode_actions.append(rel.numpy().copy())
            else:action_acc.update(rel.numpy())
            windows+=len(cur)
            decoded=tf.backward_action20({'state':{'default':batch['state']['default']},'action':{'default':rel.clone()}})['action']['default']
            dims=[0,1,2,9,10,11,18,19]
            error=max(error,float((decoded[...,dims]-raw[...,dims]).abs().max()))
            for r in [3,12]:
                rotation_error=max(rotation_error,float((_rot6d_to_R(decoded[...,r:r+6])-_rot6d_to_R(raw[...,r:r+6])).abs().max()))
            if norm is not None:
                n=norm.normalizers['action']['default']
                z=rel*n.scale+n.offset
                active=list(range(9))+[31]+list(range(32,41))+[63]
                z=z[...,active]
                saturated+=int((z.abs()>5).sum());norm_count+=z.numel()
        if aggregation=='project':action_acc.update(np.concatenate(episode_actions,axis=0))
    if error>1e-4 or rotation_error>1e-4:
        raise ValueError(f'Relative action roundtrip failed: xyz/gripper={error}, rotation matrix={rotation_error}')
    stats=dict(state=state_acc.finish(),action=action_acc.finish(),num_episodes=len(selected),num_transition=frames)
    stats['custom_relative_provenance']=dict(root=str(rd.root),horizon=32,stride=stride,padding=padding,
        aggregation=aggregation,quantile_reservoir=max_samples if aggregation=='pooled' else None,
        seed=42,action_windows=windows,episodes=selected,
        transform='CustomUnifiedEEFTransform: base=state[t], inverse Rbase',
        transform_sha256=hashlib.sha256(Path(sys.modules[CustomUnifiedEEFTransform.__module__].__file__).read_bytes()).hexdigest())
    report=dict(episodes=len(selected),frames=frames,action_windows=windows,
                max_xyz_gripper_roundtrip_error=error,max_rotation_matrix_roundtrip_error=rotation_error,
                existing_action_preclip_outside_5_fraction=saturated/norm_count if norm_count else None,
                aggregation=aggregation,
                note=('Original project policy: equal episode weights, unbiased per-episode variance, min episode q01/max episode q99. Float64 aggregation may differ slightly from original float32.' if aggregation=='project' else
                      'All moments cover scanned values. Quantiles use reservoir unless max-samples=0. Different sampling/padding policies can differ without being a defect.'))
    if existing:
        old=json.loads(Path(existing).read_text())
        report['max_absolute_stat_difference']={}
        for field in ['state','action']:
            report['max_absolute_stat_difference'][field]={k:float(np.max(np.abs(np.asarray(v)-np.asarray(old[field]['default'][k]))))
                for k,v in stats[field]['default'].items()}
    return stats,report


def bench(a):
    from torch.utils.data import DataLoader,Subset
    from agenticwam.datasets.lerobot.v3.dataset import V3MultiLeRobotDataset
    rd=V3EpisodeReader(a.root)
    timestamps={'observation.state':[i/rd.fps for i in range(33)],'action':[i/rd.fps for i in range(32)]}
    video_indices=range(0,33,4) if a.sparse_video else range(33)
    for key in rd.video_keys:timestamps[key]=[i/rd.fps for i in video_indices]
    ds=V3MultiLeRobotDataset([str(rd.root)],delta_timestamps=timestamps,video_backend=a.backend)
    count=(a.batches+a.warmup)*a.batch_size
    indices=np.random.default_rng(42).integers(0,len(ds),size=count).tolist()
    kwargs=dict(num_workers=a.workers,batch_size=a.batch_size)
    if a.workers:kwargs.update(prefetch_factor=2,persistent_workers=True)
    loader=DataLoader(Subset(ds,indices),**kwargs)
    it=iter(loader);times=[]
    for i in range(a.batches+a.warmup):
        start=time.perf_counter();sample=next(it);elapsed=time.perf_counter()-start
        if i>=a.warmup:times.append(elapsed)
        del sample
    print(json.dumps(dict(workers=a.workers,batch_size=a.batch_size,batches=a.batches,
        camera_count=len(rd.video_keys),requested_video_frames_per_camera=len(video_indices),
        batch_wait_seconds_p50=float(np.median(times)),batch_wait_seconds_p95=float(np.quantile(times,.95)),
        samples_per_second=a.batches*a.batch_size/sum(times),
        limits='Raw CPU loader only; excludes transforms, text-cache loads, H2D, model and distributed communication. Sparse mode is a benchmark, not a training patch.'),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('stats');s.add_argument('--root',type=Path,required=True)
    s.add_argument('--existing',type=Path);s.add_argument('--out',type=Path,required=True)
    s.add_argument('--stride',type=int,default=5);s.add_argument('--padding',choices=['repeat','valid'],default='repeat')
    s.add_argument('--max-samples',type=int,default=20000);s.add_argument('--episodes',type=int,nargs='+')
    s.add_argument('--aggregation',choices=['pooled','project'],default='pooled',help='project reproduces original episode-weighted normalization policy')
    b=sub.add_parser('loader');b.add_argument('--root',type=Path,required=True)
    b.add_argument('--batch-size',type=int,default=2);b.add_argument('--workers',type=int,default=4)
    b.add_argument('--batches',type=int,default=16);b.add_argument('--warmup',type=int,default=2)
    b.add_argument('--sparse-video',action='store_true');b.add_argument('--backend',choices=['pyav','torchcodec'],default='pyav')
    a=p.parse_args()
    if a.mode=='stats':
        if a.stride<1 or a.max_samples<0:p.error('stride>=1 and max-samples>=0 required')
        report_path=a.out.with_suffix(a.out.suffix+'.report.json')
        if a.out.exists() or report_path.exists():raise ValueError('Choose new output paths; existing stats never overwritten')
        stats,report=calculate(a.root,stride=a.stride,padding=a.padding,max_samples=a.max_samples,episodes=a.episodes,existing=a.existing,aggregation=a.aggregation)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(stats,allow_nan=False),encoding='utf-8')
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(report,indent=2))
    else:
        if min(a.batch_size,a.batches)<1 or min(a.workers,a.warmup)<0:p.error('Invalid loader parameters')
        bench(a)


if __name__=='__main__':main()
