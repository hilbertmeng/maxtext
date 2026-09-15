"""Verify column probes and pair with archived row interventions on the same cohort."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
PATHS=('Q','K','V','O')
def stats(x):
 x=np.asarray(x,float);return dict(mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(len(x))),median=float(np.median(x)),positive=int(np.sum(x>0)),n=len(x))
def legacy_rows(root,reference):
 values={}
 for stage in ['all','targeted','depth','vdepth','qktype']:
  fs=sorted(root.glob(f'**/{stage}_[0-9][0-9][0-9].npz'));assert len(fs)==64
  seen=set()
  for file in fs:
   i=int(file.stem[-3:]);assert i not in seen;seen.add(i)
   scenarios=json.loads((file.parent/f'{stage}_scenarios.json').read_text())
   with np.load(file) as a:
    assert str(a['sequence_hash'])==str(reference['sequence_hash'][i]);loss=a['loss'].ravel()
    for j,s in enumerate(scenarios):values.setdefault(f'row/{stage}/{s["name"]}',np.full(64,np.nan))[i]=loss[j]-reference['loss'][i,0]
 assert all(np.isfinite(v).all() for v in values.values())
 return values

def run(root,legacy):
 reference=np.load(root/'reference.npz');files=sorted(root.glob('worker*/column_[0-9][0-9][0-9].npz'));assert len(files)==64
 worker=files[0].parent;meta=json.loads((worker/'metadata.json').read_text());scenarios=json.loads((worker/'scenarios.json').read_text());ids=[s['id'] for s in scenarios];assert len(ids)==376
 old_meta=json.loads((legacy/'heads/worker0/metadata.json').read_text());assert meta['sequence_hashes']==old_meta['sequence_hashes']
 assert meta['cohort_sha256']=='68239ae352be31f968984c18a2a7e3290cdbfb665f350563aad6ff77eea84661'
 legacy_values=legacy_rows(legacy,reference);matrix=np.full((64,len(ids)),np.nan);seen=set();verification={};error=0.
 for file in files:
  i=int(file.stem[-3:]);assert i not in seen;seen.add(i)
  localmeta=json.loads((file.parent/'metadata.json').read_text());assert meta['sequence_hashes']==localmeta['sequence_hashes']
  assert scenarios==json.loads((file.parent/'scenarios.json').read_text())
  with np.load(file) as a:
   assert str(a['sequence_hash'])==str(reference['sequence_hash'][i]);loss=a['loss'].ravel();assert loss.shape==(376,) and np.isfinite(loss).all()
   np.testing.assert_array_equal(loss[:16],reference['loss'][i]);np.testing.assert_array_equal(a['baseline'].item(),reference['loss'][i,0]);matrix[i]=loss-reference['loss'][i,0]
   error=max(error,float(np.max(np.abs(loss[:16]-reference['loss'][i]))))
  verification[str(file.relative_to(root))]=dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
 measured={name:matrix[:,j] for j,name in enumerate(ids)}
 for name in set(legacy_values)&set(measured):np.testing.assert_array_equal(legacy_values[name],measured[name])
 values={**legacy_values,**measured};summary=dict(model=meta['model'],checkpoint=meta['checkpoint'],samples=64,scenarios=len(ids),unique_forwards=meta['unique_forwards'],metrics={name:stats(x) for name,x in values.items()},row_column_pairs={},four_path_shapley={},layer_type_shapley={})
 for name in measured:
  if not name.startswith('both/'):continue
  suffix=name[5:];r=values['row/'+suffix];c=values['column/'+suffix];b=values[name];pr=(r+b-c)/2;pc=(c+b-r)/2
  np.testing.assert_allclose(pr+pc,b,atol=1e-12,rtol=0)
  summary['row_column_pairs'][suffix]=dict(row=stats(r),column=stats(c),both=stats(b),column_minus_row=stats(c-r),interaction=stats(b-r-c),row_shapley=stats(pr),column_shapley=stats(pc))
 for side in ['row','column','both']:
  coalition=np.stack([values[f'{side}/all/coalition_{bits:02d}'] for bits in range(16)],axis=1);shapley=np.zeros((64,4))
  for p in range(4):
   for subset in range(16):
    if subset&(1<<p):continue
    k=subset.bit_count();weight=math.factorial(k)*math.factorial(3-k)/math.factorial(4);shapley[:,p]+=weight*(coalition[:,subset|(1<<p)]-coalition[:,subset])
  np.testing.assert_allclose(shapley.sum(1),coalition[:,15],atol=1e-12,rtol=0);summary['four_path_shapley'][side]={p:stats(shapley[:,i]) for i,p in enumerate(PATHS)}
  for path,L,F,full in [('QK','qktype/QK_L_off','qktype/QK_F_off','all/coalition_03'),('O','targeted/O_local_off','targeted/O_fetch_off','all/coalition_08')]:
   l=values[f'{side}/{L}'];f=values[f'{side}/{F}'];total=values[f'{side}/{full}'];summary['layer_type_shapley'][side+'/'+path]=dict(L=stats((l+total-f)/2),F=stats((f+total-l)/2),L_minus_F=stats(l-f),interaction=stats(total-l-f))
 np.savez_compressed(root/'paired_results.npz',gap=np.stack(list(values.values()),axis=1),names=list(values),sequence_hash=reference['sequence_hash'])
 (root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'verification.json').write_text(json.dumps(dict(raw_files=len(files),max_reference_error=error,cohort_hashes=meta['sequence_hashes'],files=verification),indent=2));print('COLUMN_VERIFIED',meta['model'],len(files),meta['unique_forwards'],'reference_error',error)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('legacy',type=Path);a=p.parse_args();run(a.root,a.legacy)
