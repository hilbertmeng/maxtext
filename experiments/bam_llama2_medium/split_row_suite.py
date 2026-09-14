"""Split one-restore suite outputs into named groups, preserving source provenance."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def run(root):
 files=sorted(root.glob('worker*/suite_[0-9][0-9][0-9].npz'));assert len(files)==64,len(files)
 seen=set();hashes=set();verification={};reference=None
 for f in files:
  index=int(f.stem.split('_')[-1]);assert index not in seen;seen.add(index)
  scenarios=json.loads((f.parent/'suite_scenarios.json').read_text());assert len(scenarios)==157
  if reference is None:reference=scenarios
  assert scenarios==reference
  with np.load(f) as a:arrays={k:np.asarray(a[k]) for k in a.files}
  digest=str(arrays['sequence_hash']);assert digest not in hashes;hashes.add(digest)
  assert np.isfinite(arrays['loss']).all();assert arrays['loss'].shape==(157,1)
  np.testing.assert_array_equal(arrays['loss'][0],arrays['baseline'])
  np.testing.assert_array_equal(arrays['loss'][141],arrays['baseline'])
  np.testing.assert_array_equal(arrays['loss'][150],arrays['baseline'])
  np.testing.assert_array_equal(arrays['loss'][3],arrays['loss'][147])
  np.testing.assert_array_equal(arrays['loss'][3],arrays['loss'][151])
  for stage,start,stop,folder in [('all',0,141,f.parent),('targeted',141,150,root/'targeted'),('depth',150,157,root/'depth')]:
   folder.mkdir(parents=True,exist_ok=True)
   (folder/f'{stage}_scenarios.json').write_text(json.dumps(scenarios[start:stop],indent=2))
   part={k:v[start:stop] if k in ('loss','gap') else v for k,v in arrays.items()}
   np.savez_compressed(folder/f'{stage}_{index:03d}.npz',**part)
  verification[str(f.relative_to(root))]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 assert seen==set(range(64))
 (root/'suite_verification.json').write_text(json.dumps(dict(samples=64,scenarios=157,native_exact_match=True,qk_exact_match=True,files=verification,derived_groups='all/targeted/depth are splits of raw worker*/suite_NNN.npz, not separate experiments'),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
