"""Validate follow-up against original paired controls and summarize joint deletion."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from summarize_row_contribution import stats

def run(root,stage='targeted'):
 target=root/stage;scenarios=json.loads((target/f'{stage}_scenarios.json').read_text());rows=[];files={}
 control_index=next(i for i,s in enumerate(scenarios) if s['name'] in ('QK_off','QK_all_off','V_all_off'))
 for i in range(64):
  file=target/f'{stage}_{i:03d}.npz'
  with np.load(file) as a,np.load(next(root.glob(f'worker*/all_{i:03d}.npz'))) as b:
   assert str(a['sequence_hash'])==str(b['sequence_hash'])
   np.testing.assert_array_equal(a['baseline'],b['baseline'])
   np.testing.assert_array_equal(a['gap'][control_index],b['gap'][4 if stage=='vdepth' else 3])
   rows.append(a['gap'].ravel().astype(float))
  files[file.name]=dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
 gaps=np.array(rows);assert np.isfinite(gaps).all()
 result={s['name']:stats(gaps[:,i]) for i,s in enumerate(scenarios)}
 (target/'summary.json').write_text(json.dumps(result,indent=2))
 (target/'verification.json').write_text(json.dumps(dict(samples=64,native_exact_match=True,whole_path_exact_match=True,files=files),indent=2))
 np.savez_compressed(target/'paired_results.npz',gap=gaps,names=np.array([s['name'] for s in scenarios]))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stage',choices=('targeted','depth','vdepth'),default='targeted');a=p.parse_args();print(json.dumps(run(a.root,a.stage),indent=2))
