"""Matched-checkpoint, frozen-backbone CE precision and backward diagnostics.

No training state is mutated. A/B use exactly the same BF16 logits. C retains
the FP32 accumulator output of a BF16-input projection. Gradients are obtained
through the sealed runtime's actual custom VJP, including dtype-boundary casts.
"""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


MODES = ('native_bf16', 'ce_fp32', 'accumulator_fp32_ce_fp32')


def loss(hidden, weights, targets, mode, runtime):
  """Differentiated, chunked native head; only selected precision changes."""
  hidden = hidden.astype(jnp.bfloat16)
  weights = weights.astype(jnp.bfloat16)
  total = jnp.float32(0)
  for lo in range(0, hidden.shape[1], 1024):
    h = hidden[:, lo:lo+1024]
    if mode == 'accumulator_fp32_ce_fp32':
      logits = jnp.matmul(h, weights, preferred_element_type=jnp.float32,
                          precision=jax.lax.Precision.HIGHEST)
    else:
      logits = jnp.matmul(h, weights, precision=jax.lax.Precision.HIGHEST)
    if mode == 'ce_fp32':
      logits = logits.astype(jnp.float32)
    labels = jax.nn.one_hot(targets[:, lo:lo+1024], weights.shape[-1], dtype=jnp.float32)
    ce = runtime.cross_entropy_with_logits(logits, labels, 0.)[0]
    total += ce.sum()
  return total/targets.size


def logit_derivatives(logits, targets, runtime):
  labels = jax.nn.one_hot(targets, logits.shape[-1], dtype=jnp.float32)
  derivatives = []
  for dtype in (jnp.bfloat16, jnp.float32):
    value, residual = runtime._cross_entropy_with_logits_fwd(logits.astype(dtype), labels, 0.)
    derivatives.append(runtime._cross_entropy_with_logits_bwd(
        residual, (jnp.ones_like(value[0]), jnp.zeros_like(value[1])))[0].astype(jnp.float32))
  return tuple(derivatives)


def token_metrics(hidden, weights, targets, runtime):
  z = jnp.matmul(hidden.astype(jnp.bfloat16), weights.astype(jnp.bfloat16),
                 precision=jax.lax.Precision.HIGHEST)
  a, b = logit_derivatives(z, targets, runtime)
  ay = jnp.take_along_axis(a, targets[..., None], axis=-1)[..., 0]
  by = jnp.take_along_axis(b, targets[..., None], axis=-1)[..., 0]
  # b is the FP32 reference on the same rounded BF16 logits.
  p = 1+by
  return dict(p_y=p, native_target_gradient=ay, fp32_target_gradient=by,
              native_row_sum=a.sum(-1), fp32_row_sum=b.sum(-1),
              error_squared=jnp.sum((a-b)**2, axis=-1),
              reference_squared=jnp.sum(b*b, axis=-1),
              logits_abs_max=jnp.max(jnp.abs(z.astype(jnp.float32)), axis=-1))


def vector_comparison(a, b):
  """Native a versus reference b; stable FP32 reductions."""
  a = a.astype(jnp.float32); b = b.astype(jnp.float32)
  aa = jnp.sum(a*a); bb = jnp.sum(b*b)
  return jnp.stack((jnp.sqrt(aa), jnp.sqrt(bb), jnp.sqrt(aa/bb),
                    jnp.sum(a*b)/jnp.sqrt(aa*bb), jnp.sqrt(jnp.sum((a-b)**2)/bb)))


def summarize_tokens(tokens):
  p = tokens['p_y']; a = tokens['native_target_gradient']; b = tokens['fp32_target_gradient']
  false_zero = (a == 0) & (b != 0)
  bins = []
  for lo, hi in [(0., .5), (.5, .9), (.9, .99), (.99, .996), (.996, 1.00001)]:
    mask = (p >= lo) & (p < hi); n = int(mask.sum())
    bins.append(dict(probability_lower=lo, probability_upper=min(hi, 1.), count=n,
        fraction=n/len(p), false_zero_fraction=float(false_zero[mask].mean()) if n else None,
        target_absolute_error_mean=float(np.abs(a[mask]-b[mask]).mean()) if n else None,
        relative_vector_error=float(np.sqrt(tokens['error_squared'][mask].sum()/
                            tokens['reference_squared'][mask].sum())) if n else None))
  return dict(tokens=len(p), high_confidence_fraction=float((p >= .99).mean()),
              false_zero_fraction=float(false_zero.mean()),
              gradient_relative_l2_error=float(np.sqrt(tokens['error_squared'].sum()/
                                                        tokens['reference_squared'].sum())),
              native_mean_row_sum=float(tokens['native_row_sum'].mean()),
              native_rms_row_sum=float(np.sqrt(np.mean(tokens['native_row_sum']**2))),
              fp32_rms_row_sum=float(np.sqrt(np.mean(tokens['fp32_row_sum']**2))),
              logits_abs_max_quantiles=np.quantile(tokens['logits_abs_max'], [.5,.95,.99]).tolist(),
              bins=bins)


def restore_weights(checkpoint):
  import tensorstore as ts
  metadata = json.loads(subprocess.check_output(['gsutil', 'cat', checkpoint+'/_METADATA']))
  key = ('params', 'params', 'decoder', 'lm_head', 'logits_dense')
  assert key in [ast.literal_eval(k) for k in metadata['tree_metadata']]
  spec = dict(driver='zarr3' if metadata.get('use_zarr3') else 'zarr',
              kvstore=dict(driver='ocdbt', base=checkpoint+'/'), path='.'.join(key))
  w = np.asarray(ts.open(spec, open=True).result().read().result())
  assert w.dtype == np.float32
  return w


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runtime-dir', type=Path, required=True)
  p.add_argument('--spec', type=Path, required=True, help='JSON list: step, inputs, checkpoint_items')
  p.add_argument('--output', type=Path, required=True)
  a = p.parse_args()
  sys.path.insert(0, str(a.runtime_dir/'MaxText'))
  import max_utils
  native = {mode: jax.jit(jax.value_and_grad(
      lambda h, w, y, mode=mode: loss(h, w, y, mode, max_utils), argnums=(0, 1))) for mode in MODES}
  metrics = jax.jit(lambda h,w,y:token_metrics(h,w,y,max_utils))
  compare = jax.jit(vector_comparison)
  a.output.mkdir(parents=True, exist_ok=True)
  expected_hashes = None
  for job in json.loads(a.spec.read_text()):
    begin = time.monotonic()
    inputs = Path(job['inputs']); source = json.loads((inputs/'complete.json').read_text())
    if expected_hashes is None:
      expected_hashes = source['sequence_hashes']
    assert source['sequence_hashes'] == expected_hashes
    assert source['label_step'] == job['step']
    weights_host = restore_weights(job['checkpoint_items'])
    weight_hash = hashlib.sha256(weights_host.tobytes()).hexdigest()
    weights = jnp.asarray(weights_host); del weights_host
    grads = {mode: None for mode in MODES}
    rows = []; token_rows = []
    for lo in range(0, 32, 8):
      hs=[]; ys=[]
      for i in range(lo, lo+8):
        with np.load(inputs/f'head-input-{i:03d}.npz') as f:
          hs.append(f['h']); ys.append(f['targets'])
      h=jnp.asarray(np.concatenate(hs)); y=jnp.asarray(np.concatenate(ys))
      out = {mode: native[mode](h, weights, y) for mode in MODES}
      row=dict(batch_start=lo, losses={m:float(out[m][0]) for m in MODES}, comparisons={})
      for mode in ('native_bf16','accumulator_fp32_ce_fp32'):
        row['comparisons'][mode]={name:np.asarray(compare(out[mode][1][idx], out['ce_fp32'][1][idx])).tolist()
            for name,idx in [('hidden',0),('weights',1)]}
      for mode in MODES:
        gw=out[mode][1][1]
        grads[mode] = gw/4 if grads[mode] is None else grads[mode]+gw/4
      rows.append(row)
      # Slice stats to per-token scalars on-device; do not export vocabulary tensors.
      for i in range(8):
        for offset in (0,1024):
          token_rows.append(jax.tree.map(lambda x:np.asarray(x).reshape(-1),
              metrics(h[i:i+1,offset:offset+1024], weights, y[i:i+1,offset:offset+1024])))
      del out, gw
      print('CE_GRAD_BATCH_OK',job['step'],lo+8,round(time.monotonic()-begin,2),flush=True)
    tokens={k:np.concatenate([x[k] for x in token_rows]) for k in token_rows[0]}
    assert all(np.isfinite(v).all() for v in tokens.values())
    arrays={mode:float(jnp.linalg.norm(g)) for mode,g in grads.items()}
    comparisons={mode:np.asarray(compare(grads[mode],grads['ce_fp32'])).tolist()
                 for mode in ('native_bf16','accumulator_fp32_ce_fp32')}
    result=dict(job=job,source_capture=source,weight_sha256=weight_hash,
                runtime_commit=subprocess.check_output(['git','-C',str(a.runtime_dir),'rev-parse','HEAD'],text=True).strip(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                comparison_fields=['norm','reference_norm','norm_ratio','cosine','relative_l2_error'],
                reference='ce_fp32: identical BF16 logits; only CE computes FP32, gradient cast back through BF16 boundary',
                gradient_scope='Head and post-ln_f input only; batch 8, averaged over 32 texts, not historical batch-1024 training gradients.',
                cohort_head_gradient_norms=arrays,cohort_head_comparison=comparisons,
                mean_loss={m:float(np.mean([r['losses'][m] for r in rows])) for m in MODES},
                per_batch=rows,token_summary=summarize_tokens(tokens),seconds=time.monotonic()-begin)
    np.savez_compressed(a.output/f'tokens-{job["step"]}.npz',**tokens)
    (a.output/f'checkpoint-{job["step"]}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CE_GRAD_CHECKPOINT_COMPLETE',job['step'],json.dumps({
        'loss':result['mean_loss'],'head_norms':arrays,'head_compare':comparisons,
        'tokens':result['token_summary']}),flush=True)
    del weights, grads, h, y, token_rows
    gc.collect()


if __name__ == '__main__':
  main()
