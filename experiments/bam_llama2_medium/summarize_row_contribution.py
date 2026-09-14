"""Merge disjoint cohort shards; paired losses, exact path Shapley, layer/unit tables."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PATHS=('Q','K','V','O')


def stats(x):
  x=np.asarray(x,dtype=float)
  return dict(n=len(x),mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,
              median=float(np.median(x)),positive=int(np.sum(x>0)))


def summarize(root,limit=64):
  files=sorted(root.rglob('all_[0-9][0-9][0-9].npz'))
  names=None;rows={};hashes={};base={}
  for file in files:
    index=int(file.stem.split('_')[-1])
    if index>=limit:continue
    scenarios=json.loads((file.parent/'all_scenarios.json').read_text())
    if names is None:names=scenarios
    assert scenarios==names,'Scenario mismatch'
    with np.load(file) as f:
      digest=str(f['sequence_hash']);gap=np.asarray(f['gap']).ravel();b=float(np.asarray(f['baseline']).ravel()[0])
    if index in rows:
      assert digest==hashes[index];np.testing.assert_allclose(gap,rows[index],atol=1e-6,rtol=0)
      continue
    rows[index]=gap;hashes[index]=digest;base[index]=b
  assert rows,'No completed samples'
  assert len(set(hashes.values()))==len(hashes),'Duplicate sequences'
  ids=sorted(rows);gaps=np.stack([rows[i] for i in ids]);lookup={s['name']:i for i,s in enumerate(names)}
  coal=np.stack([gaps[:,lookup[f'coalition_{bits:02d}']] for bits in range(16)],axis=1)
  np.testing.assert_allclose(coal[:,0],0,atol=1e-6,rtol=0)
  phi=np.zeros((len(ids),4))
  for p in range(4):
    for bits in range(16):
      if bits&(1<<p):continue
      size=bits.bit_count();weight=math.factorial(size)*math.factorial(3-size)/math.factorial(4)
      phi[:,p]+=weight*(coal[:,bits|(1<<p)]-coal[:,bits])
  np.testing.assert_allclose(phi.sum(axis=1),coal[:,15],atol=1e-10,rtol=1e-10)
  by_name={s['name']:stats(gaps[:,i]) for i,s in enumerate(names)}
  output=dict(sample_ids=ids,sequence_hashes=hashes,baseline=stats([base[i] for i in ids]),
              all_rows_off=stats(coal[:,15]),scenarios=by_name,
              knockout={p:stats(coal[:,1<<j]) for j,p in enumerate(PATHS)},
              shapley={p:stats(phi[:,j]) for j,p in enumerate(PATHS)},
              total_minus_sum_single_path=stats(coal[:,15]-sum(coal[:,1<<j] for j in range(4))),
              layer_sum={},unit_interactions={})
  for p in PATHS:
    indices=[i for i,s in enumerate(names) if s['kind']=='layer' and s['path']==p]
    output['layer_sum'][p]=stats(gaps[:,indices].sum(axis=1))
    for unit in range(8):
      indices=[i for i,s in enumerate(names) if s['kind']=='layer' and s['path']==p and s['layer']//3==unit]
      output['unit_interactions'][f'{unit}_{p}']=stats(gaps[:,lookup[f'unit_{unit}_{p}']]-gaps[:,indices].sum(axis=1))
  def value(x):
    return f"{x['mean']:+.6f} ± {1.96*x['se']:.6f}" if x['se'] is not None else f"{x['mean']:+.6f}"
  text=[f'# Row contribution: {len(ids)} sequences',
    'Model: BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow; checkpoint13500.',
    'Positive = increased loss; mean ±1.96 paired-sequence SE (descriptive, uncorrected).',
    f"All rows off: **{value(output['all_rows_off'])}**.",'',
    '| Path | Alone knockout | Exact 4-path Shapley | Half amplitude | Sum single-layer knockout (not additive attribution) |',
    '|---|---:|---:|---:|---:|']
  for p in PATHS:text.append('| '+p+' | '+' | '.join(value(v) for v in [output['knockout'][p],output['shapley'][p],by_name[f'half_{p}'],output['layer_sum'][p]])+' |')
  text+=['',f"All-off minus sum path-only knockouts: {value(output['total_minus_sum_single_path'])}",'',
         '| Layer | Q | K | V | O |','|---|---:|---:|---:|---:|']
  for layer in range(24):
    text.append(f'| {layer} '+('F' if layer%3==2 else 'L')+' | '+' | '.join(value(by_name[f'layer_{layer:02d}_{p}']) if f'layer_{layer:02d}_{p}' in by_name else 'N/A' for p in PATHS)+' |')
  text+=['','| LLF unit | Q | K | V | O |','|---|---:|---:|---:|---:|']
  for u in range(8):text.append(f'| {u} | '+' | '.join(value(by_name[f'unit_{u}_{p}']) for p in PATHS)+' |')
  (root/'summary.json').write_text(json.dumps(output,indent=2));(root/'summary.md').write_text('\n'.join(text)+'\n')
  np.savez_compressed(root/'paired_results.npz',sample_ids=ids,gaps=gaps,shapley=phi,baseline=[base[i] for i in ids])
  print('\n'.join(text[:14]))
  return output


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--limit',type=int,default=64);a=p.parse_args();summarize(a.root,a.limit)
