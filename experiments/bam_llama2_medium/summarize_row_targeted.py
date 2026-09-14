"""Validate follow-up against original paired controls and summarize joint deletion."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from summarize_row_contribution import stats

def run(root):
 target=root/'targeted';scenarios=json.loads((target/'targeted_scenarios.json').read_text());rows=[];files={}
 for i in range(64):
  file=target/f'targeted_{i:03d}.npz'
  with np.load(file) as a,np.load(next(root.glob(f'worker*/all_{i:03d}.npz'))) as b:
   assert str(a['sequence_hash'])==str(b['sequence_hash'])
   np.testing.assert_array_equal(a['baseline'],b['baseline'])
   np.testing.assert_array_equal(a['gap'][6],b['gap'][3])
   rows.append(a['gap'].ravel().astype(float))
  files[file.name]=dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
 gaps=np.array(rows);assert np.isfinite(gaps).all()
 result={s['name']:stats(gaps[:,i]) for i,s in enumerate(scenarios)}
 (target/'summary.json').write_text(json.dumps(result,indent=2))
 (target/'verification.json').write_text(json.dumps(dict(samples=64,native_exact_match=True,qk_exact_match=True,files=files),indent=2))
 np.savez_compressed(target/'paired_results.npz',gap=gaps,names=np.array([s['name'] for s in scenarios]))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(run(a.root),indent=2))
