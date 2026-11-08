"""Multi-host, read-only: same parameters and batch, gradients under several remat/packing arms.

Restores parameters only (load_parameters_path, no optimizer/checkpoint writes) and takes one
real batch from the configured data pipeline. Every arm computes sharded parameter gradients;
pairwise global and per-leaf relative differences ||a-b||/||b|| and cosines are reduced on
device. Run the same command on every worker.

  python bam_remat_grad_consistency.py --exp EXP --items .../STEP/items --dataset-path ... \
      --output-base gs://.../diag/ --arms minimal_save_splash:1,full:1,full:0
  (arm = remat_policy:packed_read_keys)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--exp', required=True)
  parser.add_argument('--items', required=True)
  parser.add_argument('--dataset-path', required=True)
  parser.add_argument('--output-base', required=True)
  parser.add_argument('--arms', default='minimal_save_splash:1,full:1,full:0')
  parser.add_argument('--result', required=True, help='gs:// or local JSON written by process 0')
  args = parser.parse_args()
  runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  sys.path.insert(0, 'MaxText')
  os.environ['LIBTPU_INIT_ARGS'] = os.environ.get('LIBTPU_INIT_ARGS', '') + ' --xla_tpu_spmd_rng_bit_generator_unsafe=true'
  import jax
  import jax.numpy as jnp
  import numpy as np
  import tensorflow as tf
  from flax import linen as nn
  from flax.traverse_util import flatten_dict
  import pyconfig
  import train
  from layers.models import Transformer

  arms = [(r, bool(int(p))) for r, p in (a.split(':') for a in args.arms.split(','))]

  def config(remat, packed, tag):
    return pyconfig.initialize(
        [None, 'MaxText/configs/base.yml'], exp_class=args.exp, run_name=f'gradconsistency-{tag}',
        base_output_directory=args.output_base, dataset_path=args.dataset_path,
        # Required by pyconfig with load_parameters_path; the empty diagnostic run directory has no
        # cursor, so setup restores parameters only and nothing is ever saved.
        enable_checkpointing=True, async_checkpointing=False, checkpoint_period=10**9,
        load_parameters_path=args.items, record_internal_nn_metrics=False, log_config=False,
        remat_policy=remat, bam_pallas_packed_read_keys=packed)

  base_cfg = config(*arms[0], 'base')
  (_, _, _, state_shardings, _, mesh, _, data_iterator, _, state) = train.setup_train_loop(base_cfg)
  params = state.params
  param_shardings = state_shardings.params
  data = next(data_iterator)
  print('SETUP_OK', jax.process_index(), sum(x.size for x in jax.tree.leaves(params)), flush=True)

  def grads_for(remat, packed):
    cfg = config(remat, packed, f'{remat}-{int(packed)}')
    model = Transformer(cfg, mesh, quant=None)

    def objective(p, batch):
      loss, _ = train.loss_fn(model, cfg, batch, jax.random.PRNGKey(0), p, is_train=True)
      return loss

    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      fn = jax.jit(jax.value_and_grad(objective), out_shardings=(None, param_shardings))
      loss, g = fn(params, data)
      jax.block_until_ready(g)
    return float(loss), g

  def compare(a, b):
    def leaf(x, y):
      x, y = x.astype(jnp.float32), y.astype(jnp.float32)
      m = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))), 1e-30)
      x, y = x / m, y / m
      return jnp.stack([jnp.sum(x * x), jnp.sum(y * y), jnp.sum(x * y), jnp.sum((x - y) ** 2), m])
    stats = jax.jit(lambda a, b: jax.tree.map(leaf, a, b))(a, b)
    flat = {'/'.join(map(str, k)): np.asarray(jax.device_get(v), np.float64) for k, v in flatten_dict(stats).items()}
    tot = np.zeros(4)
    per = {}
    for name, (aa, bb, ab, dd, m) in flat.items():
      tot += np.array([aa, bb, ab, dd]) * m * m
      per[name] = dict(rel=float(np.sqrt(dd / max(bb, 1e-300))), cos=float(ab / np.sqrt(max(aa * bb, 1e-300))),
                       norm_a=float(m * np.sqrt(aa)), norm_b=float(m * np.sqrt(bb)))
    aa, bb, ab, dd = tot
    return dict(rel=float(np.sqrt(dd / bb)), cos=float(ab / np.sqrt(aa * bb)),
                norm_a=float(np.sqrt(aa)), norm_b=float(np.sqrt(bb))), per

  results = {}
  grads = {}
  for remat, packed in arms:
    label = f'{remat}:{int(packed)}'
    loss, g = grads_for(remat, packed)
    grads[label] = g
    print('GRAD_OK', label, 'loss', loss, flush=True)
    results.setdefault('loss', {})[label] = loss
  labels = list(grads)
  pairs = {}
  for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
      total, per = compare(grads[labels[i]], grads[labels[j]])
      worst = sorted(per.items(), key=lambda kv: -kv[1]['rel'])[:12]
      attn_l1 = {k: v for k, v in per.items() if ('self_attention' in k)}
      pairs[f'{labels[i]} vs {labels[j]}'] = dict(total=total, worst_leaves=worst,
                                                   attention=attn_l1)
      print('PAIR', labels[i], 'vs', labels[j], json.dumps(total), flush=True)
      print('WORST', json.dumps([(k, round(v['rel'], 5), round(v['cos'], 6)) for k, v in worst[:6]]), flush=True)
  results['pairs'] = pairs
  if jax.process_index() == 0:
    with tf.io.gfile.GFile(args.result, 'w') as handle:
      handle.write(json.dumps(dict(exp=args.exp, items=args.items, arms=args.arms, runner_sha256=runner_sha,
                                   results=results), indent=1))
  print('GRAD_CONSISTENCY_COMPLETE', flush=True)


if __name__ == '__main__':
  main()
