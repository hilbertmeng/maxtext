"""Local verification of paired M-route parameter ablations."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
R=Path(__file__).resolve().parent;DATA=Path('/data0/mengqy/bam_diagnostics/mudd_spectral_20261102')
p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(R/'arms.json'));p.add_argument('--output',default=str(R/'verified.json'));a=p.parse_args()
m=json.loads(Path(a.manifest).read_text());F=Path(m.get('local_DATA',str(DATA/'gradient_origin_ablation_v1')))
s=json.loads((F/'complete.json').read_text());assert s['source_commit']==m['commit'] and s['state_step']==53001 and s['sequences']==32 and s['sequence_hashes']==m['sequence_hashes']
with np.load(DATA/'head_spectra/serial_53000_vocabulary_centered.npz') as z:u=z['basis'][:,:128].copy()
reference=s['reference_gradient'];ref_norm2=reference['gradient_norm']**2
with np.load(F/'keep/gradient_vectors.npz') as z:ref=z['original_dark'].astype(np.float64)
assert abs(np.sum(ref*ref)/reference['dark_gradient_energy']-1)<2e-5
checks=[]
for name,arm in s['arms'].items():
 directory=F/name;marker=json.loads((directory/'complete.json').read_text());assert marker==arm and arm['source_commit']==m['commit'] and arm['sequence_hashes']==m['sequence_hashes'] and arm['sequences']==32
 powers=[];darkpowers=[];loss=[];max_hdiff=0
 for i in range(32):
  meta=json.loads((directory/f'sequence_{i:03d}.json').read_text());assert meta['source_commit']==m['commit'] and meta['checkpoint']==s['checkpoint'] and meta['targets']==arm['targets'] and meta['sequence']==i and meta['input_hash']==m['sequence_hashes'][i]
  with np.load(directory/f'sequence_{i:03d}.npz') as z:
   h=z['h'];assert h.dtype==np.float32 and h.shape==(1,2048,2560);assert all(np.isfinite(z[k]).all() for k in z.files)
   with np.load(DATA/'results/serial/53000'/f'sequence_{i:03d}.npz') as old:
    inp=old['inputs'];targets=old['targets'];np.testing.assert_array_equal(z['targets'],targets)
    assert hashlib.sha256(np.concatenate([inp[0],targets[0,-1:]]).astype(np.int32).tobytes()).hexdigest()==meta['input_hash']
    if name=='keep':assert abs(float(z['native_loss'].mean()-old['native_loss'].mean()))<.005
   powers.append(float(np.mean(h.astype(np.float64)**2)));hp=h@u;darkpowers.append(float(np.mean(np.sum(hp.astype(np.float64)**2,-1)/2560)));loss.append(float(z['native_loss'].mean()))
   max_hdiff=max(max_hdiff,meta['head_relative_RMS_difference'])
 np.testing.assert_allclose(np.sqrt(np.mean(powers)),arm['head_RMS'],rtol=2e-5)
 np.testing.assert_allclose(np.sqrt(np.mean(darkpowers)),arm['dark_RMS'],rtol=2e-5)
 assert abs(np.mean(loss)-arm['native_CE'])<1e-6
 with np.load(directory/'gradient_vectors.npz') as z:
  actual=z['actual_dark'].astype(np.float64);fixed=z['fixed_error_dark'].astype(np.float64)
  np.testing.assert_allclose(np.sum(actual*actual),arm['dark_gradient_energy'],rtol=2e-5)
  np.testing.assert_allclose(np.linalg.norm(fixed),arm['fixed_original_error_dark_gradient_norm'],rtol=2e-5)
  np.testing.assert_allclose(np.sum((ref-fixed)*ref)/ref_norm2,arm['fixed_original_error_aligned_reduction'],rtol=2e-4,atol=1e-7)
 checks.append(dict(arm=name,sequences=32,max_head_RMS_difference_from_original=max_hdiff,parameter_targets=arm['targets'],head_and_dark_RMS_recomputed_all_texts=True,gradient_vectors_checked=True))
base=s['arms']['keep'];assert abs(base['global_head_gradient_norm']/reference['gradient_norm']-1)<.02 and abs(base['dark_gradient_fraction']-reference['dark_gradient_fraction'])<.01 and abs(base['dark_input_energy_fraction']-.7696534410623249)<.015
assert len(s['arms']) in [15,16]
out=dict(complete=True,method='Original capture graph, parameter-only current-only M-route resets; all frames/metadata and fixed-error gradient vectors verified',source_commit=m['commit'],checkpoint=s['checkpoint'],arms=len(checks),sequences=32*len(checks),checks=checks,reference_reproduced=True,hidden_vector_differences_reported_not_silently_allocated=True)
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print('ORIGIN_LOCAL_VERIFICATION_PASS',len(checks),'ARMS',flush=True)
