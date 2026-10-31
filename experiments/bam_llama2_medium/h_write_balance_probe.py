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
READ_ARMS = ('q', 'k', 'v', 'o')
READ_FIELDS = ('dynamic_gate_mean', 'dynamic_gate_ms',
               'static_multiplier_mean', 'static_multiplier_ms', 'read_ms')


def read_gate_statistics(logits, static_multipliers, outputs, qk_cols):
  """Actual kernel gate inputs and outputs; per sequence/arm, no parameter proxies.

  Static multipliers are 1.1*sigmoid(Wx+b)/initial_opening, not raw gate
  openings. Q/K RMS includes only the M-derived prefix before full QKNorm.
  """
  import jax
  import jax.numpy as jnp
  result = []
  for i, (logit, multiplier, output) in enumerate(zip(logits, static_multipliers, outputs)):
    dynamic = jax.nn.sigmoid(logit.astype(jnp.float32))
    static = multiplier.astype(jnp.float32)
    value = output[..., :qk_cols] if i < 2 else output
    mean = lambda x: jnp.mean(x, axis=tuple(range(1, x.ndim)))
    result.append(jnp.stack((mean(dynamic), mean(dynamic**2), mean(static),
                             mean(static**2), mean(value.astype(jnp.float32)**2)), axis=-1))
  return jnp.stack(result, axis=1)


@contextlib.contextmanager
def capture_reads(enabled=True):
  """Capture the fused read's real gates without changing its returned values."""
  if not enabled:
    yield
    return
  import jax.numpy as jnp
  from flax import linen as nn
  from layers import attentions, bam_pallas

  def intercept(next_fun, args, kwargs, ctx):
    module = ctx.module
    if ctx.method_name != '__call__' or not isinstance(module, attentions.BamAttention):
      return next_fun(*args, **kwargs)
    original = bam_pallas.read

    def measured(matrix, weight, q_key, q_logit, k_key, k_logit, vo_key,
                 o_logit, v_logit, q_standard, k_standard, **kw):
      out = original(matrix, weight, q_key, q_logit, k_key, k_logit, vo_key,
                     o_logit, v_logit, q_standard, k_standard, **kw)
      if not module.is_initializing():
        multipliers = kw.get('static_gates')
        if multipliers is None:
          multipliers = tuple(jnp.ones_like(q_logit) for _ in READ_ARMS)
        stats = read_gate_statistics((q_logit, k_logit, v_logit, o_logit),
                                     multipliers, out[:4], kw['qk_cols'])
        module.sow('intermediates', 'read_gate_probe', stats)
      return out

    bam_pallas.read = measured
    try:
      return next_fun(*args, **kwargs)
    finally:
      bam_pallas.read = original

  with nn.intercept_methods(intercept):
    yield


def extract_read_stats(collection, final_layer):
  import jax.numpy as jnp
  from flax.traverse_util import flatten_dict
  found = []
  for path, value in flatten_dict(collection).items():
    if path[-1] != 'read_gate_probe':
      continue
    while isinstance(value, (tuple, list)):
      value = value[0]
    if 'layers' in path:
      found.append((0, value))
    elif 'final_local_layer' in path:
      found.append((final_layer, value[None]))
    else:
      match = next((re.fullmatch(r'layers_(\d+)', part) for part in path
                    if re.fullmatch(r'layers_(\d+)', part)), None)
      if match:
        found.append((int(match[1]), value[None]))
  if not found:
    raise ValueError('No read_gate_probe values captured')
  return jnp.concatenate([value for _, value in sorted(found)], axis=0)


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
        leave_kw = dict(kw)
        # Older all-normalized runtimes predate the content_divisors API.
        if 'content_divisors' in kw:
          leave_kw['content_divisors'] = (divisors[other],)
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
  p.add_argument('--vmem-mib', type=int, default=40,
                 help='v6e scoped VMEM budget; N32/V64 requires more than14MiB. Medium can use14.')
  p.add_argument('--tile', type=int, default=128,
                 help='Pallas token tile; token-minor TPU blocks require 128 alignment')
  p.add_argument('--config-only', action='store_true',
                 help='Verify exact historical configuration without weights/inference')
  p.add_argument('--read-gates', action='store_true',
                 help='Also capture all eight read gates and actual four-arm M read RMS')
  args = p.parse_args()
  if args.tile <= 0 or args.tile % 128 or args.length % args.tile:
    p.error('--tile must be a positive multiple of128 dividing sequence length')
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir / 'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils
  import pyconfig
  from layers.models import Transformer
  assert jax.process_count() == 1
  config_output = args.output / 'configuration'
  (config_output / 'write-balance-probe').mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'],
      exp_class=args.exp, run_name='write-balance-probe',
      base_output_directory=str(config_output) + '/',
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
  if args.config_only:
    print('HISTORICAL_CONFIG_OK', args.exp, cfg.num_decoder_layers, cfg.emb_dim,
          cfg.bam_k, cfg.bam_v, cfg.mlp_dim, cfg.bam_memory_mlp_dim, flush=True)
    return
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  cohort = np.load(args.cohort)['tokens'][:args.sequences, :args.length + 1]
  assert cohort.shape == (args.sequences, args.length + 1)
  args.output.mkdir(parents=True, exist_ok=True)
  pos = jnp.arange(args.length, dtype=jnp.int32)[None]
  seg = jnp.ones((1, args.length), jnp.int32)

  forward_traces = 0

  def forward(params, tokens):
    nonlocal forward_traces
    forward_traces += 1
    print('FORWARD_TRACE', forward_traces, flush=True)
    with capture_writes(), capture_reads(args.read_gates):
      result, collections = model.apply(params, tokens[:, :-1], pos,
          decoder_segment_ids=seg, decoder_target_mask=seg,
          decoder_target_tokens=tokens[:, 1:], enable_dropout=False,
          mutable=['intermediates'])
    writes = extract_stats(collections['intermediates'])
    if args.read_gates:
      return writes, extract_read_stats(collections['intermediates'], cfg.num_decoder_layers-1)
    return writes

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
      # Initialization and Orbax restoration must enter JIT with the same
      # container and sharding representation; parameter values stay unchanged.
      params = jax.device_put(unfreeze(nn.unbox(params)),
                              jax.sharding.SingleDeviceSharding(jax.devices()[0]))
      jax.block_until_ready(params)
      leaves, treedef = jax.tree.flatten(params)
      signature = hashlib.sha256(repr((str(treedef), [
          (x.shape, str(x.dtype), x.weak_type, str(x.sharding), x.committed)
          for x in leaves])).encode()).hexdigest()
      weak_type_paths = [jax.tree_util.keystr(k) for k, v in
                         jax.tree_util.tree_flatten_with_path(params)[0] if v.weak_type]
      print('PARAM_SIGNATURE', step, signature, flush=True)
      print('WEAK_TYPE_PATHS', step, weak_type_paths, flush=True)
      del leaves
      print('RESTORE_OK', step, round(time.monotonic()-start, 2), flush=True)
      rows, read_rows = [], []
      for index, tokens in enumerate(cohort):
        value = compiled(params, jnp.asarray(tokens[None]))
        if args.read_gates:
          value, read_value = value
          read_value = np.asarray(read_value)
          assert read_value.shape == (cfg.num_decoder_layers, 1, 4, len(READ_FIELDS)), read_value.shape
          read_rows.append(read_value[:, 0])
        value = np.asarray(value)
        assert value.shape == (cfg.num_decoder_layers-1, 1, 2, len(FIELDS)), value.shape
        rows.append(value[:, 0])
        if index in (0, 7, 15, 31):
          print('FORWARD_OK', step, index+1, round(time.monotonic()-start, 2), flush=True)
      array = np.stack(rows)
      assert np.isfinite(array).all()
      path = args.output / f'step-{step}-T{args.length}.npz'
      extra = dict(read_statistics=np.stack(read_rows)) if args.read_gates else {}
      if extra:
        assert np.isfinite(extra['read_statistics']).all()
      np.savez_compressed(path, statistics=array, **extra)
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
          param_signature=signature, weak_type_paths=weak_type_paths,
          forward_traces=forward_traces,
          raw_file=str(path), devices=[str(x) for x in jax.devices()])
      if args.read_gates:
        meta.update(read_arms=READ_ARMS, read_fields=READ_FIELDS,
                    static_gate_init=float(cfg.bam_static_read_gate_init),
                    read_key_scale=float(cfg.bam_read_key_scale),
                    read_semantics='Dynamic gate=sigmoid(logit); static multiplier=1.1*sigmoid(Wx+b)/initial_opening. Both depend on token. M-derived Q/K prefix RMS is before full QKNorm; V/O RMS is the actual fused output including common scale and dynamic-O gamma.')
      (path.with_suffix('.json')).write_text(json.dumps(meta, indent=2)+'\n')
      print('STEP_DONE', step, round(meta['seconds'], 2), flush=True)
      del params, rows
      gc.collect()


if __name__ == '__main__':
  main()
