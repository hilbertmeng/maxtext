"""Independently verify frozen-head cases and compare paired changes on texts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();m=json.loads(Path(a.manifest).read_text());root=Path(m['local_DATA']);complete=json.loads((root/'complete.json').read_text());assert complete['cases']==64 and complete['source_commit']==m['commit']
boot=np.random.default_rng(20261102).integers(0,32,(5000,32))
def estimate(v):
 v=np.asarray(v,np.float64);return dict(mean=float(v.mean()),ci95=np.quantile(v[boot].mean(1),[.025,.975]).tolist())
rows={}
for step in m['steps']:
 values=[];max_replay=0;max_means=0
 for i in range(32):
  folder=root/str(step);meta=json.loads((folder/f'sequence_{i:03d}.json').read_text());assert meta['source_commit']==m['commit'] and meta['checkpoint']==m['checkpoints'][str(step)] and meta['state_step']==step and meta['input_hash']==m['cohort_hashes'][i]
  assert meta['projection_dtype']=='bfloat16_materialized' and meta['CE_reduction_dtype']=='float32'
  with np.load(folder/f'sequence_{i:03d}.npz') as z:
   assert z['inputs'].shape==z['targets'].shape==(1,2048)
   tokens=np.concatenate([z['inputs'][0],z['targets'][0,-1:]]).astype(np.int32);assert hashlib.sha256(tokens.tobytes()).hexdigest()==m['cohort_hashes'][i]
   means={}
   for k in z.files:
    assert np.isfinite(z[k]).all()
    if k in ['inputs','targets']:continue
    assert z[k].shape==(2048,) and z[k].dtype==np.float32,(k,z[k].shape,z[k].dtype)
    means[k]=float(z[k].astype(np.float64).mean());max_means=max(max_means,abs(means[k]-meta['means'][k]))
   assert abs(means['archive_native_CE']-m['reference'][str(step)]['old'][i])<1e-6
   max_replay=max(max_replay,abs(means['CE_fp32']-means['archive_native_CE']));values.append(means)
 assert max_means<2e-6 and max_replay<.005
 v={k:np.array([x[k] for x in values]) for k in values[0]};baseline=v['CE_fp32'];gap=np.array(m['reference'][str(step)]['serial'])-np.array(m['reference'][str(step)]['old']);replay=baseline-v['archive_native_CE'];assert abs(replay.mean())<1e-4
 variants={}
 for k in ['CE_pax_bf16','CE_maxtext_train','CE_strict_bf16','CE_bf16_renormalized']:
  delta=v[k]-baseline;variants[k]=dict(loss=estimate(v[k]),delta=estimate(delta),ratio_to_native_FP32_gap=float(delta.mean()/gap.mean()),remaining_gap_after_change=estimate(np.array(m['reference'][str(step)]['serial'])-v[k]))
 rows[str(step)]=dict(old_FP32=estimate(baseline),old_archive_FP32=estimate(v['archive_native_CE']),serial_archive_FP32=estimate(m['reference'][str(step)]['serial']),serial_minus_old_native_FP32_gap=estimate(gap),replay_difference=estimate(replay),variants=variants,bf16_logprob_log_normalizer=estimate(v['bf16_logprob_log_normalizer']),checks=dict(sequences=32,targets_per_sequence=2048,input_hashes_checkpoint_state_dtypes_finite_verified=True,max_mean_recompute_error=max_means,max_per_text_native_replay_difference=max_replay),per_text=v)
for v in rows.values():v['per_text']={k:x.tolist() for k,x in v['per_text'].items()}
out=dict(complete=True,manifest=m,device=complete['device_kind'],jax=complete['jax_version'],rows=rows,scope='Frozen old native H and fixed materialized BF16 projection, same32x2048 texts. Comparison gap is archival old/new native BF16 networks with standardized FP32 CE, not original BF16 training CE. Confidence intervals bootstrap32 texts5000 times, no token independence assumed. This intervention measures immediate reported loss, not training trajectory. Strict BF16 log-probability materialization and FP32 renormalization are diagnostic controls.')
Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
for step,row in rows.items():print(step,'gap',row['serial_minus_old_native_FP32_gap'],'Pax BF16',row['variants']['CE_pax_bf16'],'MaxText train BF16',row['variants']['CE_maxtext_train'],flush=True)
print('SOFTMAX_64_CASES_INDEPENDENTLY_VERIFIED',flush=True)
