import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from test_custom_v3_codegen import dataset

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('diagnose',ROOT/'scripts/diagnose_custom_data.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def test_relative_stats_local_frame_and_episode_boundaries(tmp_path):
    for d in ['meta/episodes/chunk-000','data/chunk-000']:(tmp_path/d).mkdir(parents=True)
    info=dict(codebase_version='v3.0',fps=10,total_episodes=2,total_frames=75,
              data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
              features={'observation.state':{'dtype':'float32'},'action':{'dtype':'float32'}})
    (tmp_path/'meta/info.json').write_text(json.dumps(info))
    pd.DataFrame(dict(task_index=[0],task=['test'])).to_parquet(tmp_path/'meta/tasks.parquet')
    rows=[];meta=[]
    for ep,n in enumerate([40,35]):
        start=len(rows)
        for f in range(n):
            st=np.zeros(34,np.float32);st[33]=ep;st[14]=100*(ep+1);st[17:23]=[0,1,0,-1,0,0];st[26:32]=[1,0,0,0,1,0]
            ac=np.r_[st[14:23],st[23:32],st[32:34]].astype(np.float32);ac[0]+=1
            rows.append(dict(index=len(rows),frame_index=f,episode_index=ep,**{'observation.state':st,'action':ac}))
        meta.append(dict(episode_index=ep,dataset_from_index=start,dataset_to_index=len(rows),length=n,
                         **{'data/chunk_index':0,'data/file_index':0}))
    pd.DataFrame(rows).to_parquet(tmp_path/'data/chunk-000/file-000.parquet')
    pd.DataFrame(meta).to_parquet(tmp_path/'meta/episodes/chunk-000/file-000.parquet')
    stats,report=tool.calculate(tmp_path,stride=1,padding='repeat',max_samples=16)
    assert stats['num_transition']==75 and report['action_windows']==75
    mean=stats['action']['default']['global_mean']
    assert abs(mean[0])<1e-6 and abs(mean[1]+1)<1e-6
    assert report['max_xyz_gripper_roundtrip_error']<1e-4
    valid,report=tool.calculate(tmp_path,stride=1,padding='valid',max_samples=16)
    assert report['action_windows']==13 # 40-32+1 + 35-32+1, never across packed episodes
    assert abs(valid['action']['default']['global_mean'][1]+1)<1e-6
    previous=tmp_path/'existing.json';previous.write_text(json.dumps(stats))
    _,report=tool.calculate(tmp_path,stride=1,padding='repeat',max_samples=16,existing=previous)
    assert report['existing_action_preclip_outside_5_fraction']==0
    project,report=tool.calculate(tmp_path,stride=1,padding='repeat',aggregation='project')
    assert abs(stats['state']['default']['global_mean'][63]-35/75)<1e-6
    assert project['state']['default']['global_mean'][63]==.5
    assert project['action']['default']['global_mean'][63]==.5


def test_project_policy_matches_original_torch_formula():
    import torch
    rng=np.random.default_rng(42)
    arrays=[rng.normal(size=(n,32,64)).astype(np.float32) for n in [35,70,40]]
    acc=tool.ProjectStats()
    for a in arrays:acc.update(a)
    out=acc.finish()['default']
    tensors=[torch.from_numpy(a) for a in arrays]
    means=torch.stack([a.mean(0) for a in tensors]);vars=torch.stack([a.var(0) for a in tensors])
    stepmean=means.mean(0);globalmean=means.mean((0,1))
    assert np.allclose(out['global_mean'],globalmean.numpy(),atol=1e-6)
    assert np.allclose(out['global_std'],(vars+(means-globalmean)**2).mean((0,1)).sqrt().numpy(),atol=1e-6)
    assert np.allclose(out['stepwise_std'],(vars+(means-stepmean)**2).mean(0).sqrt().numpy(),atol=1e-6)
    assert np.allclose(out['stepwise_q01'],torch.stack([torch.quantile(a,.01,dim=0) for a in tensors]).amin(0),atol=1e-6)
    assert np.allclose(out['stepwise_q99'],torch.stack([torch.quantile(a,.99,dim=0) for a in tensors]).amax(0),atol=1e-6)


def test_loader_benchmark_uses_real_video(dataset,capsys):
    from types import SimpleNamespace
    root,_=dataset
    path=root/'data/chunk-000/file-000.parquet'
    df=pd.read_parquet(path)
    df['action']=[np.r_[s[14:32],s[32:34]].astype(np.float32) for s in df['observation.state']]
    df.to_parquet(path)
    info_path=root/'meta/info.json';info=json.loads(info_path.read_text())
    info['features']['action']={'dtype':'float32','shape':[20]}
    info_path.write_text(json.dumps(info))
    tool.bench(SimpleNamespace(root=root,sparse_video=True,batches=2,warmup=0,
                               batch_size=1,workers=0,backend='pyav'))
    report=json.loads(capsys.readouterr().out)
    assert report['requested_video_frames_per_camera']==9 and report['samples_per_second']>0
