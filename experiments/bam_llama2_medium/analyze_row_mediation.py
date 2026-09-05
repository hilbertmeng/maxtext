"""Aggregate loss-only mediator artifacts and audit paired controls across workers."""
import argparse
import json
from pathlib import Path
import numpy as np


def stats(values):
  x=np.asarray(values,dtype=float)
  return dict(mean=float(x.mean()),ci95=float(1.96*x.std(ddof=1)/len(x)**.5) if len(x)>1 else None,
              median=float(np.median(x)),positive=int((x>0).sum()),n=len(x))


def load(root):
  root=Path(root); meta=json.loads((root/'summary.json').read_text())
  loss=[]; hashes=[]; validation=[]
  for p in sorted(root.glob('batch_*.npz')):
    with np.load(p) as x:
      loss.append(x['loss']); hashes.extend(x['sequence_hashes'].tolist()); validation.append(x['validation_loss'])
  if len(set(hashes))!=len(hashes):raise ValueError('duplicate sequences')
  if len(hashes)>meta['requested_sequences']:raise ValueError('unexpected extra sequences')
  return meta,np.concatenate(loss),hashes,np.concatenate(validation)


def analyze(roots):
  report={'runs':[],'cross_run_controls':[],'matched_self_reference':[]}
  loaded=[]
  for root in roots:
    meta,loss,hashes,validation=load(root); loaded.append((meta,loss,hashes))
    gap=loss[:,1]-loss[:,0]
    entries=[]
    for i,a in enumerate(meta['arms']):
      d=loss[:,i]-loss[:,0]
      local=loss[:,i]-loss[:,1] if a['corrupted'] else d
      entries.append(dict(name=a['name'],delta=stats(d),effect_vs_recipient=stats(local),
          rescue_fraction=float(-local.mean()/gap.mean()) if a['corrupted'] else None,
          block_fraction=float(d.mean()/gap.mean()) if a['donor_corrupted'] else None))
    report['runs'].append(dict(root=str(root),metadata=meta,n=len(loss),
        validation_clean=stats(validation[:,0]),validation_ablated=stats(validation[:,1]),arms=entries))
  for i,(m,x,h) in enumerate(loaded):
    for j,(m2,y,h2) in enumerate(loaded[:i]):
      if m['source_layer']!=m2['source_layer'] or m['checkpoint']!=m2['checkpoint']:continue
      if m.get('source_component','cross')!=m2.get('source_component','cross'):continue
      iy={k:v for v,k in enumerate(h2)}
      pairs=[(a,iy[k]) for a,k in enumerate(h) if k in iy]
      if not pairs:continue
      d=np.stack([x[a,:2]-y[b,:2] for a,b in pairs])
      report['cross_run_controls'].append(dict(first=j,second=i,n=len(pairs),
          maximum_absolute=float(abs(d).max()),mean=d.mean(0).tolist()))
      modes=[m.get('reference_mode','opposite'),m2.get('reference_mode','opposite')]
      if sorted(modes)!=['opposite','self']:continue
      if m['cohort_sha256']!=m2['cohort_sha256']:raise ValueError('cohort mismatch')
      # Pair by sequence and arm identity, never by snapshot length or file order.
      names2={a['name']:q for q,a in enumerate(m2['arms'])}
      corrected=[]
      for q,arm in enumerate(m['arms']):
        if not arm['name'].startswith(('rescue_','block_')):continue
        if arm['name'] not in names2:continue
        q2=names2[arm['name']]
        other=m2['arms'][q2]
        if arm['control']!=other['control'] or arm['corrupted']!=other['corrupted']:
          raise ValueError('intervention mismatch')
        diff=np.asarray([x[a,q]-y[b,q2] for a,b in pairs])
        if modes[0]=='self':diff=-diff
        entry=dict(name=arm['name'],opposite_minus_self=stats(diff))
        if arm['corrupted']:
          deletion=np.asarray([x[a,1]-x[a,0] for a,b in pairs])
          entry['remaining_deletion_cost']=stats(deletion+diff)
        corrected.append(entry)
      report['matched_self_reference'].append(dict(first=j,second=i,
          control_maximum_absolute=float(abs(d).max()),arms=corrected))
  return report


if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('roots',nargs='+');p.add_argument('--output',required=True)
  a=p.parse_args();r=analyze(a.roots)
  Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
  for run in r['runs']:
    print(run['root'],run['n'])
    for arm in run['arms']:
      print(f"{arm['name']:30s} delta={arm['delta']['mean']:+.6f} recipient={arm['effect_vs_recipient']['mean']:+.6f}")
  print('cross-run controls',r['cross_run_controls'])
