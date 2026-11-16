"""Per-layer causal attention-logit ranges of a frozen checkpoint (read-only forward).

Captures each layer's Splash inputs (scaled Q, K) and reports max |logit|, the mean over
rows of the row maximum and the per-head maximum. No gradients, updates or writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-dir', type=Path, required=True)
  parser.add_argument('--runtime-commit', required=True)
  parser.add_argument('--exp', required=True)
  parser.add_argument('--items', required=True, help='checkpoint .../STEP/items')
  parser.add_argument('--cohort', required=True, help='cohort.npz (gs:// or local)')
  parser.add_argument('--label', required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--max-samples', type=int, default=4)
  args = parser.parse_args()
  actual = subprocess.check_output(['git', '-C', str(args.runtime_dir), 'rev-parse', 'HEAD'], text=True).strip()
  assert actual == args.runtime_commit, (actual, args.runtime_commit)
  runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir / 'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  import tensorflow as tf
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils
  import pyconfig
  import train
  from layers import attentions
  from layers.models import Transformer

  args.output.mkdir(parents=True, exist_ok=True)
  (args.output / 'configuration' / 'logits').mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=args.exp, run_name='logits',
      base_output_directory=str(args.output / 'configuration') + '/', enable_checkpointing=False, only_eval=True,
      dataset_type='synthetic', target_train_batch_size=None, per_device_batch_size=1.,
      record_internal_nn_metrics=False, record_training_health_metrics=False, jax_cache_dir='', log_config=False)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  print('RESTORE_BEGIN', args.items, flush=True)
  cp = ocp.PyTreeCheckpointer()
  tree = cp.metadata(args.items).item_metadata.tree['params']
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), tree)
  restored = cp.restore(args.items, item={'params': abstract}, transforms={},
      restore_args={'params': ocp.checkpoint_utils.construct_restore_args(abstract)})
  cp.close()
  params = jax.device_put(unfreeze(nn.unbox(restored['params'])), sharding)
  del restored
  print('RESTORE_OK', sum(x.size for x in jax.tree.leaves(params)), flush=True)
  local = args.output / 'cohort.npz'
  with tf.io.gfile.GFile(args.cohort, 'rb') as handle:
    local.write_bytes(handle.read())
  with np.load(local) as saved:
    batch = {key: saved[key] for key in saved.files}
  t = cfg.max_target_length
  assert batch['inputs'].shape[1] >= t, (batch['inputs'].shape, t)
  order = np.random.default_rng(0).permutation(batch['inputs'].shape[0])[:args.max_samples]

  splash = attentions._bam_splash_attention
  stack = []

  def captured(q, k, v, segments, config, mesh, window=None, seq_minor=False):
    stack[-1].sow('intermediates', 'lr_q', q)
    stack[-1].sow('intermediates', 'lr_k', k)
    return splash(q, k, v, segments, config, mesh, window=window, seq_minor=seq_minor)

  attentions._bam_splash_attention = captured

  def intercept(next_fun, call_args, kwargs, context):
    if isinstance(context.module, attentions.BamAttention) and context.method_name == '__call__':
      stack.append(context.module)
      try:
        return next_fun(*call_args, **kwargs)
      finally:
        stack.pop()
    return next_fun(*call_args, **kwargs)

  def ranges(qk):
    q, k = qk[0][0].astype(jnp.float32), qk[1][0].astype(jnp.float32)      # [T, N, D]
    s = jnp.einsum('tnd,snd->nts', q, k, precision=jax.lax.Precision.HIGHEST)
    allowed = (jnp.arange(t)[None, :] < jnp.arange(t)[:, None])[None]
    a = jnp.where(allowed, jnp.abs(s), 0.)
    row_max = jnp.max(jnp.where(allowed, s, -jnp.inf), axis=-1)[:, 1:]
    return jnp.max(a), jnp.mean(row_max), jnp.max(a, axis=(1, 2))

  def forward(p, data):
    with nn.intercept_methods(intercept):
      loss, aux = train.loss_fn(model, cfg, dict(data), jax.random.PRNGKey(0), p, is_train=True)
    inter = aux['intermediate_outputs']['intermediates']['decoder']
    body = inter['layers']['block']['self_attention']
    abs_max, row_mean, per_head = jax.lax.map(ranges, (body['lr_q'][0], body['lr_k'][0]))
    final = inter.get('final_local_layer', {}).get('block', {}).get('self_attention')
    if final is not None:
      fa, fm, fh = ranges((final['lr_q'][0], final['lr_k'][0]))
      abs_max = jnp.concatenate((abs_max, fa[None]))
      row_mean = jnp.concatenate((row_mean, fm[None]))
      per_head = jnp.concatenate((per_head, fh[None]))
    return loss, abs_max, row_mean, per_head

  results = []
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    compiled = jax.jit(forward)
    for sequence in order:
      data = {k: jnp.asarray(v[sequence:sequence + 1, :t]) for k, v in batch.items()}
      loss, abs_max, row_mean, per_head = jax.device_get(compiled(params, data))
      record = dict(label=args.label, sequence=int(sequence), loss=float(loss),
                    abs_max=np.asarray(abs_max).tolist(), row_max_mean=np.asarray(row_mean).tolist(),
                    per_head_max=np.asarray(per_head).tolist())
      results.append(record)
      top = int(np.argmax(abs_max))
      print('LOGITS', json.dumps(dict(label=args.label, sequence=int(sequence), loss=float(loss),
            max_abs=float(np.max(abs_max)), layer=top, l1=float(abs_max[1]),
            per_layer=[round(float(x), 1) for x in abs_max])), flush=True)
  (args.output / f'{args.label}.json').write_text(json.dumps(dict(
      runtime_commit=actual, runner_sha256=runner_sha, exp=args.exp, items=args.items, cohort=args.cohort,
      results=results), indent=1) + '\n')
  print('LOGITS_COMPLETE', args.label, flush=True)


if __name__ == '__main__':
  main()
