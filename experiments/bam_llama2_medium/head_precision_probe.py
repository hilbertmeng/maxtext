"""Replay a frozen unembedding input with BF16/FP32 projection and training CE.

The backbone is not rerun. Hidden states and targets come from
head_gradient_amplitude_probe.py; checkpoint master head weights remain FP32.
The supplied runtime's custom-VJP forward is the formal training loss algebra.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np


def head_losses(hidden, weights, targets, projection_dtype, softmax_dtype, ce_forward):
  logits = jnp.matmul(hidden.astype(projection_dtype), weights.astype(projection_dtype),
                      precision=jax.lax.Precision.HIGHEST)
  logits = logits.astype(softmax_dtype)
  labels = jax.nn.one_hot(targets, weights.shape[-1], dtype=jnp.float32)
  loss = ce_forward(logits, labels, 0.)[0][0]
  return loss.astype(jnp.float32).reshape(hidden.shape[0], -1).mean(axis=-1)


def paired_summary(values, reference):
  delta = np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
  rng = np.random.default_rng(43000)
  boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
  return dict(mean_delta=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
              sequence_delta_min=float(delta.min()), sequence_delta_max=float(delta.max()))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runtime-dir', type=Path, required=True)
  p.add_argument('--inputs', type=Path, required=True)
  p.add_argument('--checkpoint-items', required=True)
  p.add_argument('--output', type=Path, required=True)
  a = p.parse_args()
  sys.path.insert(0, str(a.runtime_dir/'MaxText'))
  import max_utils
  import tensorstore as ts
  metadata = json.loads(subprocess.check_output(['gsutil', 'cat', a.checkpoint_items+'/_METADATA']))
  key = ('params', 'params', 'decoder', 'lm_head', 'logits_dense')
  assert key in [ast.literal_eval(k) for k in metadata['tree_metadata']]
  spec = dict(driver='zarr3' if metadata.get('use_zarr3', False) else 'zarr',
              kvstore=dict(driver='ocdbt', base=a.checkpoint_items+'/'), path='.'.join(key))
  master = np.asarray(ts.open(spec, open=True).result().read().result())
  assert master.dtype == np.float32
  weights = jnp.asarray(master)
  modes = {'bf16_bf16': (jnp.bfloat16, jnp.bfloat16),
           'bf16_fp32': (jnp.bfloat16, jnp.float32),
           'fp32_bf16': (jnp.float32, jnp.bfloat16),
           'fp32_fp32': (jnp.float32, jnp.float32)}
  compiled = {name: jax.jit(lambda h, w, y, pd=pd, sd=sd:
                           head_losses(h, w, y, pd, sd, max_utils._cross_entropy_with_logits_fwd))
              for name, (pd, sd) in modes.items()}
  # Plain evaluation calls the primal, which uses a different BF16 algebra.
  @jax.jit
  def eval_primal(h, w, y):
    logits = jnp.matmul(h.astype(jnp.bfloat16), w.astype(jnp.bfloat16),
                        precision=jax.lax.Precision.HIGHEST)
    labels = jax.nn.one_hot(y, w.shape[-1], dtype=jnp.float32)
    return max_utils.cross_entropy_with_logits(logits, labels, 0.)[0].mean()
  values = {name: [] for name in modes}
  values['bf16_bf16_eval_primal'] = []
  hashes = []
  inputs = sorted(a.inputs.glob('head-input-*.npz'))
  assert len(inputs) == 32, len(inputs)
  for path in inputs:
    with np.load(path) as f:
      h = jnp.asarray(f['h']); y = jnp.asarray(f['targets'])
      hashes.append(hashlib.sha256(f['targets'].tobytes()).hexdigest())
    for name, fn in compiled.items():
      value = float(np.asarray(fn(h, weights, y))[0])
      assert np.isfinite(value)
      values[name].append(value)
    values['bf16_bf16_eval_primal'].append(float(eval_primal(h, weights, y)))
    print('PRECISION_SEQUENCE_OK', len(hashes), flush=True)
  result = dict(checkpoint_items=a.checkpoint_items,
                runtime_commit=subprocess.check_output(['git', '-C', str(a.runtime_dir), 'rev-parse', 'HEAD'], text=True).strip(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_capture=json.loads((a.inputs/'complete.json').read_text()),
                fixed_hidden='Identical native BF16 ln_f outputs in all arms; no backbone precision change.',
                weights_dtype=str(master.dtype), weight_shape=list(master.shape),
                weight_sha256=hashlib.sha256(master.tobytes()).hexdigest(),
                sequence_target_hashes=hashes, ce='runtime custom-VJP training forward, z_loss=0',
                per_sequence=values, mean_loss={k: float(np.mean(v)) for k, v in values.items()},
                delta_vs_fp32_fp32={k: paired_summary(v, values['fp32_fp32']) for k, v in values.items()})
  a.output.parent.mkdir(parents=True, exist_ok=True)
  a.output.write_text(json.dumps(result, indent=2)+'\n')
  print('PRECISION_COMPLETE', json.dumps(result['mean_loss']), flush=True)


if __name__ == '__main__':
  main()
