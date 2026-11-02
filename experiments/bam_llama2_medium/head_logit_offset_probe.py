"""Diagnose common logit offsets and BF16 output rounding on frozen features."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from ce_gradient_precision_probe import restore_weights


def offset_metrics(h, w, y, runtime):
  h=h.astype(jnp.bfloat16); w=w.astype(jnp.bfloat16)
  high=jnp.matmul(h,w,precision=jax.lax.Precision.HIGHEST,preferred_element_type=jnp.float32)
  low=jnp.matmul(h,w,precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)
  mean=high.mean(-1,keepdims=True)
  centered=high-mean; maxshift=high-high.max(-1,keepdims=True)
  labels=jax.nn.one_hot(y,w.shape[-1],dtype=jnp.float32)
  variants=dict(accumulator_fp32=high,bf16_logits=low,center_fp32_control=centered,
                center_then_bf16=centered.astype(jnp.bfloat16).astype(jnp.float32),
                maxshift_then_bf16=maxshift.astype(jnp.bfloat16).astype(jnp.float32))
  losses={k:runtime._cross_entropy_with_logits_fwd(z,labels,0.)[0][0] for k,z in variants.items()}
  return dict(losses=losses,logit_mean=mean[...,0],logit_min=high.min(-1),logit_max=high.max(-1),
              logit_std=jnp.sqrt(jnp.mean(centered*centered,-1)),
              output_rounding_rms=jnp.sqrt(jnp.mean((high-low)**2,-1)),
              # Compare representations: an in-graph FP32 comparison across a
              # BF16 round-trip can retain excess precision on TPU. A separate
              # host-materialization control confirmed native == rounded high.
              native_vs_rounded_accumulator_mismatch=jnp.mean(
                  jax.lax.bitcast_convert_type(low.astype(jnp.bfloat16),jnp.uint16) !=
                  jax.lax.bitcast_convert_type(high.astype(jnp.bfloat16),jnp.uint16)))


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runtime-dir',type=Path,required=True)
  p.add_argument('--spec',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  a=p.parse_args();sys.path.insert(0,str(a.runtime_dir/'MaxText'))
  import max_utils
  fn=jax.jit(lambda h,w,y:offset_metrics(h,w,y,max_utils))
  a.output.mkdir(parents=True,exist_ok=True)
  for job in json.loads(a.spec.read_text()):
    weights=jnp.asarray(restore_weights(job['checkpoint_items']))
    records=[]
    for i in range(32):
      with np.load(Path(job['inputs'])/f'head-input-{i:03d}.npz') as f:
        h=jnp.asarray(f['h']);y=jnp.asarray(f['targets'])
      for lo in (0,1024):
        records.append(jax.tree.map(lambda x:np.asarray(x).reshape(-1),fn(h[:,lo:lo+1024],weights,y[:,lo:lo+1024])))
    values=jax.tree.map(lambda *xs:np.concatenate(xs),*records)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(values))
    quantiles={k:np.quantile(v,[0,.01,.05,.5,.95,.99,1.]).tolist() for k,v in values.items() if k!='losses'}
    result=dict(job=job,quantile_levels=[0,.01,.05,.5,.95,.99,1.],quantiles=quantiles,
                mean_loss={k:float(v.mean()) for k,v in values['losses'].items()},
                max_fp32_center_control_ce_difference=float(np.max(np.abs(values['losses']['center_fp32_control']-values['losses']['accumulator_fp32']))),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(a.output/f'offset-tokens-{job["step"]}.npz',
        **{k:v for k,v in values.items() if k!='losses'},**{'loss_'+k:v for k,v in values['losses'].items()})
    (a.output/f'offset-{job["step"]}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('OFFSET_CHECKPOINT_COMPLETE',job['step'],json.dumps(result),flush=True)


if __name__=='__main__':
  main()
