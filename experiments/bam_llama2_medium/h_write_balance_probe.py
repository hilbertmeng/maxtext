"""Read-only write-content and matrix-increment probes on an exact runtime checkout.

Only compact per-sequence statistics leave the compiled forward. Original
Pallas writes remain unchanged. Extra leave-one-writer calls measure each
writer's marginal effect after the actual BF16 matrix-store rounding.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

FIELDS = ('content_ms', 'gated_content_ms', 'delta_ms', 'gate_mean',
          'gate_ms', 'raw_content_ms', 'both_normalized_content_ms',
          'stored_marginal_ms', 'stored_nonzero_fraction')


def factors_and_statistics(groups, divisors, epsilon, matrix_dtype):
  import jax
  import jax.numpy as jnp
  result = []
  for (content, logits, address), tau in zip(groups, divisors):
    raw = content.astype(jnp.float32)
    normalized = raw * jax.lax.rsqrt(jnp.mean(raw**2, axis=-1, keepdims=True) + epsilon)
    c = normalized if tau is None else raw / tau
    g = jax.nn.sigmoid(logits.astype(jnp.float32))
    cg = c * g[..., None]
    a = address.astype(jnp.float32)
    a *= jax.lax.rsqrt(jnp.mean(a*a, axis=-1, keepdims=True) + epsilon)
    # Same factor rounding and FP32 MXU accumulation as v8t _tm_write_kernel.
    delta = jnp.einsum('btnv,btnk->btkv', a.astype(matrix_dtype),
                       cg.astype(matrix_dtype), preferred_element_type=jnp.float32)
    ms = lambda x: jnp.mean(x.astype(jnp.float32)**2, axis=tuple(range(1, x.ndim)))
    result.append(jnp.stack((ms(c), ms(cg), ms(delta),
                             jnp.mean(g, axis=(1, 2)), ms(g), ms(raw),
                             ms(normalized)), axis=-1))
  return jnp.stack(result, axis=1)


@contextlib.contextmanager
def capture_writes():
  import jax.numpy as jnp
  from flax import linen as nn
  from layers import bam_pallas, fusion
  original = bam_pallas.write

  def intercept(next_fun, args, kwargs, ctx):
    module = ctx.module
    if ctx.method_name != '__call__' or not isinstance(module, fusion.SubDecoderLayer):
      return next_fun(*args, **kwargs)

    def measured(matrix, groups, **kw):
      out = original(matrix, groups, **kw)
      if module.is_initializing() or len(groups) != 2:
        return out
      divisors = kw.get('content_divisors', (None, None))
      stats = factors_and_statistics(groups, divisors, kw['epsilon'], matrix.dtype)
      margins = []
      for writer in range(2):
        other = 1 - writer
        leave_kw = dict(kw, content_divisors=(divisors[other],))
        leave = original(matrix, (groups[other],), **leave_kw)
        marginal = out.astype(jnp.float32) - leave.astype(jnp.float32)
        axes = tuple(range(1, marginal.ndim))
        margins.append(jnp.stack((jnp.mean(marginal**2, axis=axes),
                                  jnp.mean(marginal != 0, axis=axes)), axis=-1))
      stats = jnp.concatenate((stats, jnp.stack(margins, axis=1)), axis=-1)
      module.sow('intermediates', 'write_balance_probe', stats)
      return out

    before = bam_pallas.write
    bam_pallas.write = measured
    try:
      return next_fun(*args, **kwargs)
    finally:
      bam_pallas.write = before

  with nn.intercept_methods(intercept):
    yield


def extract_stats(collection):
  import jax.numpy as jnp
  from flax.traverse_util import flatten_dict
  found = []
  for path, value in flatten_dict(collection).items():
    if path[-1] != 'write_balance_probe':
      continue
    while isinstance(value, (tuple, list)):
      value = value[0]
    if 'layers' in path:
      found.append((0, value))
    else:
      match = next((re.fullmatch(r'layers_(\d+)', x) for x in path
                    if re.fullmatch(r'layers_(\d+)', x)), None)
      if match is not None:
        found.append((int(match[1]), value[None]))
  if not found:
    raise ValueError('No write_balance_probe values captured')
  return jnp.concatenate([v for _, v in sorted(found)], axis=0)


def restore_params(path):
  import jax
  import orbax.checkpoint as ocp
  cp = ocp.PyTreeCheckpointer()
  metadata = cp.metadata(path)
  tree = metadata.item_metadata.tree['params']
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sharding), tree)
  restored = cp.restore(path, item={'params': abstract}, transforms={},
                        restore_args={'params': ocp.checkpoint_utils.construct_restore_args(abstract)})
  cp.close()
  return restored['params']


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runtime-dir', type=Path, required=True)
  p.add_argument('--exp', required=True)
  p.add_argument('--checkpoint-root', required=True)
  p.add_argument('--steps', required=True, help='comma-separated, 0 creates seeded initialization')
  p.add_argument('--cohort', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--length', type=int, default=2048)
  p.add_argument('--sequences', type=int, default=32)
  p.add_argument('--vmem-mib', type=int, default=14)
  p.add_argument('--tile', type=int, default=64,
                 help='Forward-only Pallas token tile, bounded for v6e VMEM')
  args = p.parse_args()
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir / 'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  import max_utils
  import pyconfig
  from layers.models import Transformer
  assert jax.process_count() == 1
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'],
      exp_class=args.exp, run_name='write-balance-probe',
      base_output_directory='/tmp/h-write-balance-output/',
      enable_checkpointing=False, only_eval=True, dataset_type='synthetic',
      target_train_batch_size=None, per_device_batch_size=1.,
      max_target_length=args.length, max_prefill_predict_length=args.length,
      record_internal_nn_metrics=False, record_training_health_metrics=False,
      jax_cache_dir='', log_config=False)
  # Historical runtimes carry these optional kernel knobs only on exp classes,
  # so their CLI parser rejects them even though BamAttention consumes them.
  cfg.get_keys().update(bam_pallas_read_tile=args.tile, bam_pallas_write_tile=args.tile,
                        bam_pallas_vmem_mib=args.vmem_mib)
  assert cfg.bam_pallas_core and cfg.bam_pallas_body == 'v8t'
  assert not cfg.bam_sqrt_n_scale
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  cohort = np.load(args.cohort)['tokens'][:args.sequences, :args.length + 1]
  assert cohort.shape == (args.sequences, args.length + 1)
  args.output.mkdir(parents=True, exist_ok=True)
  pos = jnp.arange(args.length, dtype=jnp.int32)[None]
  seg = jnp.ones((1, args.length), jnp.int32)

  def forward(params, tokens):
    with capture_writes():
      result, collections = model.apply(params, tokens[:, :-1], pos,
          decoder_segment_ids=seg, decoder_target_mask=seg,
          decoder_target_tokens=tokens[:, 1:], enable_dropout=False,
          mutable=['intermediates'])
    return extract_stats(collections['intermediates'])

  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    compiled = jax.jit(forward)
    for step in map(int, args.steps.split(',')):
      start = time.monotonic()
      checkpoint = None if step == 0 else args.checkpoint_root.rstrip('/') + f'/{step}/items'
      print('RESTORE_BEGIN', step, checkpoint, flush=True)
      if step:
        params = restore_params(checkpoint)
      else:
        # Reuse the runtime's exact training initializer and RNG dictionaries.
        params = nn.unbox(jax.jit(lambda rng: max_utils.init_initial_state(
            model, None, cfg, False, rng).params)(jax.random.PRNGKey(cfg.init_weights_seed)))
      jax.block_until_ready(params)
      print('RESTORE_OK', step, round(time.monotonic()-start, 2), flush=True)
      rows = []
      for index, tokens in enumerate(cohort):
        value = np.asarray(compiled(params, jnp.asarray(tokens[None])))
        assert value.shape == (cfg.num_decoder_layers-1, 1, 2, len(FIELDS)), value.shape
        rows.append(value[:, 0])
        if index in (0, 7, 15, 31):
          print('FORWARD_OK', step, index+1, round(time.monotonic()-start, 2), flush=True)
      array = np.stack(rows)
      assert np.isfinite(array).all()
      path = args.output / f'step-{step}-T{args.length}.npz'
      np.savez_compressed(path, statistics=array)
      mean = array.mean(axis=0)
      ratio = lambda field: np.sqrt(mean[:, 1, FIELDS.index(field)] /
                                    np.maximum(mean[:, 0, FIELDS.index(field)], 1e-30)).tolist()
      meta = dict(exp=args.exp, checkpoint=checkpoint, step=step,
          planned_steps=int(cfg.steps), native_layers=cfg.num_decoder_layers,
          write_layers=list(range(cfg.num_decoder_layers-1)), fields=FIELDS,
          sequences=args.sequences, length=args.length, init_seed=cfg.init_weights_seed,
          sequence_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in cohort],
          ratios={f: ratio(f) for f in ('content_ms','gated_content_ms','delta_ms','stored_marginal_ms')},
          means=mean.tolist(), seconds=time.monotonic()-start,
          semantics='Ratios of aggregate RMS across same sequences/tokens/heads. Stored marginal: full write minus actual BF16 leave-one-writer result; margins are conditional and not additive.',
          overrides=dict(only_eval=True, batch=1, length=args.length,
                         vmem_mib=args.vmem_mib, forward_tile=args.tile),
          raw_file=str(path), devices=[str(x) for x in jax.devices()])
      (path.with_suffix('.json')).write_text(json.dumps(meta, indent=2)+'\n')
      print('STEP_DONE', step, round(meta['seconds'], 2), flush=True)
      del params, rows
      gc.collect()


if __name__ == '__main__':
  main()
