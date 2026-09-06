"""Paired source-position loss decomposition, with exact-null audits."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_row_mediation import stats


def load(root):
  root=Path(root)
  meta=json.loads((root/'summary.json').read_text())
  records=[]
  for path in sorted(root.glob('batch_*.npz')):
    with np.load(path) as x:
      for name in ('null_max_error','past_max_error','source_scope_error',
                   'unused_reference_error','immediate_cut_error'):
        if np.any(x[name]!=0):raise ValueError(f'{path}: {name}')
      for i,h in enumerate(x['sequence_hashes']):
        records.append(dict(hash=str(h),position=int(x['positions'][i]),
            effect=x['position_effects'][i], loss=x['loss'][i],
            z_norm=float(x['source_z_norm'][i])))
      # Cutting the actual source increment before its first consumer should
      # reproduce deleting that read; audit in token space, not only mean loss.
      names=[a['name'] for a in meta['arms']]
      cut=names.index(f'cut_L{meta["source_layer"]}_attention')
      error=float(np.max(abs(x['token_loss'][:,cut]-x['token_loss'][:,1])))
      if error!=0:raise ValueError(f'{path}: source-cut/deletion differs by {error}')
  keys=[(r['hash'],r['position']) for r in records]
  if len(set(keys))!=len(keys):raise ValueError('duplicate source positions')
  return meta,records


def summarize(roots):
  output=dict(runs=[],self_cross=[])
  loaded=[]
  for root in roots:
    meta,records=load(root);loaded.append((meta,records))
    e=np.stack([r['effect'] for r in records])
    rows=[]
    by_name={a['name']:i for i,a in enumerate(meta['arms'])}
    for i,arm in enumerate(meta['arms']):
      row=dict(name=arm['name'],total=stats(e[:,i,:6].sum(-1)),
          origin=stats(e[:,i,0]),future=stats(e[:,i,1:6].sum(-1)),
          bins={b[0]:stats(e[:,i,j]) for j,b in enumerate(meta['bins'])})
      if meta.get('source_mode','point')=='all':
        row.pop('origin');row.pop('future');row.pop('bins')
        row['mean_token_delta']=stats(np.array([r['loss'][i]-r['loss'][0] for r in records]))
      if arm['name'].startswith('joint_') and arm['name']!='joint_all_direct_consumers':
        fields=np.asarray(arm['control'])
        single=[]
        for l,f in zip(*np.nonzero(fields)):
          name=f'deny_L{l}_{meta["controls"][f]}'
          if name in by_name:single.append(by_name[name])
        if len(single)==int(np.count_nonzero(fields)):
          difference=e[:,i,:6].sum(-1)-e[:,single,:6].sum((1,2))
          row['joint_minus_sum_individual']=stats(difference)
      rows.append(row)
    output['runs'].append(dict(root=str(root),metadata=meta,n=len(records),
        z_norm=stats([r['z_norm'] for r in records]),arms=rows))
  for i,(m,r) in enumerate(loaded):
    for j,(m2,r2) in enumerate(loaded[:i]):
      if m['checkpoint']!=m2['checkpoint'] or m['source_layer']!=m2['source_layer']:continue
      if {m['source_component'],m2['source_component']}!={'self','cross'}:continue
      if m.get('source_mode','point')!=m2.get('source_mode','point'):continue
      if m['arms']!=m2['arms'] or m['cohort_sha256']!=m2['cohort_sha256']:
        raise ValueError('unmatched self/cross experiments')
      idx={(x['hash'],x['position']):x for x in r2}
      pairs=[(x,idx[(x['hash'],x['position'])]) for x in r if (x['hash'],x['position']) in idx]
      base_delta=np.array([a['loss'][0]-b['loss'][0] for a,b in pairs])
      if np.any(base_delta!=0):raise ValueError('self/cross baseline mismatch')
      differences=np.stack([a['effect']-b['effect'] for a,b in pairs])
      if m['source_component']=='self':differences=-differences
      output['self_cross'].append(dict(first=j,second=i,n=len(pairs),
          contrast='cross minus self (same positions, unchanged baseline)',
          arms=[dict(name=a['name'],total=stats(differences[:,q,:6].sum(-1)),
              origin=stats(differences[:,q,0]),future=stats(differences[:,q,1:6].sum(-1)))
              for q,a in enumerate(m['arms'])]))
      if m.get('source_mode','point')=='all':
        for q,a in enumerate(output['self_cross'][-1]['arms']):
          a.pop('origin');a.pop('future')
          delta=np.asarray([(x['loss'][q]-x['loss'][0])-(y['loss'][q]-y['loss'][0])
                            for x,y in pairs])
          if m['source_component']=='self':delta=-delta
          a['mean_token_delta']=stats(delta)
  return output


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('roots',nargs='+')
  parser.add_argument('--output',required=True);args=parser.parse_args()
  result=summarize(args.roots)
  Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
  print('CONSUMER_ANALYSIS_READY '+args.output)
