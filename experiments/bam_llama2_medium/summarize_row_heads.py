"""Merge exact layer/head knockouts; preserve per-sequence scores and selection split."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

PATHS=('Q','K','V','O')

def stats(a):
 a=np.asarray(a,float)
 return dict(mean=float(a.mean()),se=float(a.std(ddof=1)/np.sqrt(len(a))),median=float(np.median(a)),positive=int(np.sum(a>0)),n=len(a))

def run(root):
 heads=root/'heads';reference=np.load(heads/'reference.npz');g=np.full((64,24,4,16),np.nan)
 seen=set();files={};metas=[];native_error=0.;control_error=0.
 for worker in sorted(heads.glob('worker*')):
  metadata=json.loads((worker/'metadata.json').read_text());metas.append(metadata)
  scenarios=json.loads((worker/'scenarios.json').read_text())
  sequence_files=sorted(worker.glob('heads_[0-9][0-9][0-9].npz'));assert len(sequence_files)==64,(worker,len(sequence_files))
  for file in sequence_files:
   index=int(file.stem.split('_')[-1])
   with np.load(file) as a:
    assert str(a['sequence_hash'])==str(reference['sequence_hash'][index])
    values=a['loss'].ravel().astype(float);native_error=max(native_error,float(abs(a['baseline'].item()-reference['loss'][index,0])))
    np.testing.assert_allclose(values[:5],reference['loss'][index],atol=1e-6,rtol=0)
    control_error=max(control_error,float(np.max(np.abs(values[:5]-reference['loss'][index]))))
    assert np.isfinite(values).all()
    for scenario,value in zip(scenarios,values):
     if scenario['kind']!='head':continue
     l,p,h=scenario['layer'],PATHS.index(scenario['path']),scenario['head'];key=(index,l,p,h)
     assert key not in seen;seen.add(key);g[key]=value-reference['loss'][index,0]
   files[str(file.relative_to(heads))]=dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
 expected={(i,l,p,h) for i in range(64) for l in range(1,24) for p in range(4) if not(p==2 and l%3==2) for h in range(16)}
 assert seen==expected,(len(seen),len(expected));g[:,0,:,:]=0.
 assert len({m['model'] for m in metas})==1 and len({m['checkpoint'] for m in metas})==1
 for m in metas:assert m['sequence_hashes']==metas[0]['sequence_hashes']
 whole=json.loads((root/'summary.json').read_text());rows={};rankings={}
 for l in range(1,24):
  for p,path in enumerate(PATHS):
   if p==2 and l%3==2:continue
   values=g[:,l,p,:];mean=values.mean(0);first=values[:32].mean(0);second=values[32:].mean(0)
   order=np.argsort(-mean,kind='stable');positive=np.maximum(mean,0);total=float(positive.sum())
   key=f'{l:02d}_{path}';rows[key]=dict(layer=l,path=path,heads={str(h):stats(values[:,h]) for h in range(16)},
    single_head_sum=stats(values.sum(1)),whole_layer_knockout=whole['scenarios'][f'layer_{l:02d}_{path}'],
    sorted_heads=order.tolist(),positive_singleton_score_top1_fraction=float(positive[order[:1]].sum()/total) if total else None,
    positive_singleton_score_top4_fraction=float(positive[order[:4]].sum()/total) if total else None,
    first32_top4=np.argsort(-first,kind='stable')[:4].tolist(),last32_top4=np.argsort(-second,kind='stable')[:4].tolist())
   rank_first=np.argsort(first,kind='stable');rank_second=np.argsort(second,kind='stable')
   rows[key]['half_split_top4_overlap']=len(set(rank_first[-4:])&set(rank_second[-4:]))
   rankings[key]=dict(first32_ascending=rank_first.tolist(),last32_ascending=rank_second.tolist())
 controls=[dict(name='native',kind='control',coalition=0)]+[dict(name=f'all_{p}',kind='control',path=p,coalition=1<<i) for i,p in enumerate(PATHS)]
 groups=list(controls)
 for path in PATHS:
  for count in (4,8):
   for side in ('bottom','top'):
    remove=[]
    for key,r in rankings.items():
     if key.split('_')[1]!=path:continue
     chosen=r['first32_ascending'][:count] if side=='bottom' else r['first32_ascending'][-count:]
     remove += [[int(key.split('_')[0]),path,h] for h in chosen]
    groups.append(dict(name=f'{path}_{side}{count}',kind='group',remove=remove,selection='first32 singleton mean; evaluate last32 separately'))
 for count in (4,8):
  remove=[]
  for key,r in rankings.items():remove += [[int(key.split('_')[0]),key.split('_')[1],h] for h in r['first32_ascending'][:count]]
  groups.append(dict(name=f'all_bottom{count}',kind='group',remove=remove,selection='first32 singleton mean; evaluate last32 separately'))
 assert len(groups)==23
 output=dict(model=metas[0]['model'],checkpoint=metas[0]['checkpoint'],samples=64,heads_per_path_layer=16,
  measured_head_cases=1344,known_zero='layer0',not_applicable='V on F layers',
  concentration_definition='Fraction of summed positive single-head knockout means, not an additive allocation of whole-layer causal effect',
  rows=rows,first32_selection_last32_evaluation=True)
 (heads/'head_summary.json').write_text(json.dumps(output,indent=2))
 (heads/'group_scenarios.json').write_text(json.dumps(groups,indent=2))
 (heads/'verification.json').write_text(json.dumps(dict(samples=64,measured_head_cases=1344,measurements=len(seen),max_native_error=native_error,max_control_error=control_error,files=files),indent=2))
 np.savez_compressed(heads/'paired_head_gaps.npz',gap=g,baseline=reference['loss'][:,0],sequence_hash=reference['sequence_hash'])
 print('HEAD_MATRIX_VERIFIED',output['model'],len(seen),'native_error',native_error,'control_error',control_error)
 return output
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
