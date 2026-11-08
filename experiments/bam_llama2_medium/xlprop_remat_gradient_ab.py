"""Same-checkpoint, same-sequence parameter gradients under remat/packing variants (read-only).

Tests whether XL V5 gradient spikes depend on the Splash residual remat (saved forward
output/logsumexp + recomputed Q/K/V in the backward) or on the 20-head whole-block packed
read keys. Every arm restores the same parameters and runs the same sequences; only
bam_pallas_packed_read_keys and remat_policy change. Also records per-layer causal
logit ranges of the forward. No parameter update or checkpoint write.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ARMS = (('packed_fss', True, 'full_save_splash'), ('packed_full', True, 'full'),
        ('unpacked_fss', False, 'full_save_splash'), ('unpacked_full', False, 'full'))
COMPONENTS = ('query', 'key', 'W_lq_c8_up', 'W_lk_c8_up', 'W_lv_c8_up', 'W_R_up', 'out')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-dir', type=Path, required=True)
  parser.add_argument('--runtime-commit', required=True)
  parser.add_argument('--exp', default='BamXLPropV5FastKernel')
  parser.add_argument('--checkpoint-root', required=True)
  parser.add_argument('--cohort', required=True, help='cohort.npz (gs:// or local) of the next global batch')
  parser.add_argument('--step', type=int, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--max-samples', type=int, default=4)
  parser.add_argument('--arms', default=','.join(a[0] for a in ARMS))
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
  from flax.traverse_util import flatten_dict
  import max_utils
  import pyconfig
  import train
  from layers import attentions
  from layers.models import Transformer

  assert jax.process_count() == 1
  args.output.mkdir(parents=True, exist_ok=True)

  def config(packed, remat):
    folder = args.output / f'configuration-{int(packed)}-{remat}'
    (folder / 'remat-ab').mkdir(parents=True, exist_ok=True)
    return pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=args.exp, run_name='remat-ab',
        base_output_directory=str(folder) + '/', enable_checkpointing=False, only_eval=True,
        dataset_type='synthetic', target_train_batch_size=None, per_device_batch_size=1.,
        record_internal_nn_metrics=False, record_training_health_metrics=False, jax_cache_dir='',
        log_config=False, bam_pallas_packed_read_keys=packed, remat_policy=remat)

  base = config(True, 'full_save_splash')
  layers = base.num_decoder_layers - 1          # scanned body; the terminal layer is separate
  checkpoint = args.checkpoint_root.rstrip('/') + f'/{args.step}/items'
  print('RESTORE_BEGIN', checkpoint, flush=True)
  cp = ocp.PyTreeCheckpointer()
  tree = cp.metadata(checkpoint).item_metadata.tree['params']
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), tree)
  restored = cp.restore(checkpoint, item={'params': abstract}, transforms={},
      restore_args={'params': ocp.checkpoint_utils.construct_restore_args(abstract)})
  cp.close()
  params = jax.device_put(unfreeze(nn.unbox(restored['params'])), sharding)
  del restored
  print('RESTORE_OK', sum(x.size for x in jax.tree.leaves(params)), flush=True)

  with tf.io.gfile.GFile(args.cohort, 'rb') as handle:
    local = args.output / 'cohort.npz'
    local.write_bytes(handle.read())
  with np.load(local) as saved:
    batch = {key: saved[key] for key in saved.files}
  order = np.random.default_rng(0).permutation(batch['inputs'].shape[0])[:args.max_samples]

  # Capture each layer's Splash inputs (stop-gradient) for logit ranges; values only, no probes.
  splash = attentions._bam_splash_attention
  stack = []

  def captured(q, k, v, segments, cfg, mesh, window=None, seq_minor=False):
    module = stack[-1]
    module.sow('intermediates', 'ab_q', jax.lax.stop_gradient(q))
    module.sow('intermediates', 'ab_k', jax.lax.stop_gradient(k))
    return splash(q, k, v, segments, cfg, mesh, window=window, seq_minor=seq_minor)

  attentions._bam_splash_attention = captured

  def intercept(next_fun, call_args, kwargs, context):
    if isinstance(context.module, attentions.BamAttention) and context.method_name == '__call__':
      stack.append(context.module)
      try:
        return next_fun(*call_args, **kwargs)
      finally:
        stack.pop()
    return next_fun(*call_args, **kwargs)

  def scaled_norm(x):
    x = x.astype(jnp.float32)
    m = jnp.max(jnp.abs(x))
    return m * jnp.sqrt(jnp.sum(jnp.square(x / jnp.maximum(m, 1e-30)))), m

  def per_layer_norms(grads):
    """{component: [27 body layers + terminal]} L2 norms (max-scaled, then combined in FP64 on host)."""
    flat = flatten_dict(grads['params']['decoder'])
    out = {}
    for path, g in flat.items():
      name = path[-2] if path[-1] in ('kernel', 'bias') else path[-1]
      if path[0] == 'layers':
        axis = 1 if g.ndim >= 2 else 0
        assert g.shape[axis] == layers, (path, g.shape)
        moved = jnp.moveaxis(g, axis, 0).reshape(layers, -1)
        out['/'.join(path)] = jax.vmap(scaled_norm)(moved)
      else:
        n, m = scaled_norm(g)
        out['/'.join(path)] = (n[None], m[None])
    return out

  def logit_ranges(q, k):
    """Per layer: max |causal logit| over all heads/rows and the mean over rows of the row max."""
    def one(qk):
      q, k = qk[0][0], qk[1][0]                              # [T, N, D]
      t = q.shape[0]
      s = jnp.einsum('tnd,snd->nts', q.astype(jnp.float32), k.astype(jnp.float32),
                     precision=jax.lax.Precision.HIGHEST)
      allowed = jnp.arange(t)[None, :] < jnp.arange(t)[:, None]
      s = jnp.where(allowed[None], s, -jnp.inf)
      row_max = jnp.max(s, axis=-1)[:, 1:]
      return jnp.stack([jnp.max(jnp.abs(jnp.where(allowed[None], s, 0.))), jnp.mean(row_max), jnp.max(row_max)])
    return jax.lax.map(one, (q, k))

  results = []
  arms = [a for a in ARMS if a[0] in args.arms.split(',')]
  for label, packed, remat in arms:
    cfg = config(packed, remat)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)

    def loss_and_grads(p, data, cfg=cfg, model=model):
      def objective(p):
        with nn.intercept_methods(intercept):
          loss, aux = train.loss_fn(model, cfg, dict(data), jax.random.PRNGKey(0), p, is_train=True)
        body = aux['intermediate_outputs']['intermediates']['decoder']['layers']['block']['self_attention']
        return loss, (body['ab_q'][0], body['ab_k'][0])
      (loss, (q, k)), grads = jax.value_and_grad(objective, has_aux=True)(p)
      total = jnp.stack([jnp.stack(scaled_norm(g)) for g in jax.tree.leaves(grads)])
      return loss, total, per_layer_norms(grads), logit_ranges(q, k)

    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      compiled = jax.jit(loss_and_grads)
      for ordinal, sequence in enumerate(order):
        data = {k: jnp.asarray(v[sequence:sequence + 1]) for k, v in batch.items()}
        started = time.monotonic()
        loss, total, norms, logits = jax.device_get(jax.block_until_ready(compiled(params, data)))
        total = np.asarray(total, np.float64)
        global_norm = float(np.sqrt(np.sum(np.square(total[:, 0]))))
        layers_out = {}
        for path, (n, m) in norms.items():
          leaf = path.split('/')[-2] if path.split('/')[-1] in ('kernel', 'bias') else path.split('/')[-1]
          if path.startswith('layers/') and leaf in COMPONENTS and '/self_attention/' in path:
            layers_out[leaf] = [float(x) for x in np.asarray(n, np.float64)]
        layer_total = np.zeros(layers)
        for path, (n, m) in norms.items():
          if path.startswith('layers/'):
            layer_total += np.square(np.asarray(n, np.float64))
        record = dict(arm=label, packed=packed, remat=remat, sequence=int(sequence), ordinal=ordinal,
            loss=float(loss), global_grad_norm=global_norm, layer_grad_norm=np.sqrt(layer_total).tolist(),
            attention_components=layers_out,
            logits=dict(abs_max=np.asarray(logits[:, 0]).tolist(), row_max_mean=np.asarray(logits[:, 1]).tolist(),
                        row_max_max=np.asarray(logits[:, 2]).tolist()),
            input_sha256=hashlib.sha256(np.asarray(data['inputs']).tobytes()).hexdigest(),
            seconds=time.monotonic() - started)
        results.append(record)
        print('AB_RESULT', json.dumps(dict(arm=label, sequence=int(sequence), loss=record['loss'],
              global_grad_norm=global_norm, l0=float(np.sqrt(layer_total[0])), l1=float(np.sqrt(layer_total[1])),
              l2=float(np.sqrt(layer_total[2])), logit_abs_max_l1=record['logits']['abs_max'][1],
              logit_abs_max_all=max(record['logits']['abs_max']))), flush=True)
        (args.output / 'results.json').write_text(json.dumps(dict(
            runtime_commit=actual, runner_sha256=runner_sha, checkpoint=checkpoint, cohort=args.cohort,
            sequence_order=order.tolist(), results=results), indent=1) + '\n')
  print('REMAT_AB_COMPLETE', flush=True)


if __name__ == '__main__':
  main()
