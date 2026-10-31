"""Paired inference loss interventions on dynamic Q/K read gates, exact old runtimes."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def intervention_matrix(layers, high_start):
  """0=learned, 1=initial gate, 2=exact zero; intervene only on upper layers."""
  import numpy as np
  if not 0 <= high_start < layers:
    raise ValueError('Invalid first upper-layer index')
  names, controls = ['learned'], [np.zeros((layers, 2), np.int32)]
  for arm, columns in [('Q', [0]), ('K', [1]), ('QK', [0, 1])]:
    for operation, code in [('initial', 1), ('off', 2)]:
      value = np.zeros((layers, 2), np.int32)
      value[np.ix_(list(range(high_start, layers)), columns)] = code
      names.append(f'upper_{arm}_{operation}')
      controls.append(value)
  return names, np.stack(controls)


def replace_logit(logit, operation, initial_gate):
  import jax.numpy as jnp
  initial = jnp.asarray(math.log(initial_gate / (1-initial_gate)), logit.dtype)
  # Finite -1000 gives exactly zero FP32 sigmoid, without introducing infinities.
  return jnp.where(operation == 0, logit,
                   jnp.where(operation == 1, initial, jnp.asarray(-1000., logit.dtype)))


@contextlib.contextmanager
def intervene_read_gates(controls, initial_gate=.05, record=False):
  """Change only real fused Q/K logits, using dynamic scan indices incl. tail."""
  import jax.numpy as jnp
  from flax import linen as nn
  from layers import attentions, bam_pallas

  def intercept(next_fun, args, kwargs, ctx):
    module = ctx.module
    if ctx.method_name != '__call__' or not isinstance(module, attentions.BamAttention):
      return next_fun(*args, **kwargs)
    layer = kwargs.get('layer_index')
    if layer is None:
      layer = module.layer_inx
    original = bam_pallas.read

    def modified(matrix, weight, q_key, q_logit, k_key, k_logit, vo_key,
                 o_logit, v_logit, q_standard, k_standard, **kw):
      if not module.is_initializing():
        q_logit = replace_logit(q_logit, controls[layer, 0], initial_gate)
        k_logit = replace_logit(k_logit, controls[layer, 1], initial_gate)
      out = original(matrix, weight, q_key, q_logit, k_key, k_logit, vo_key,
                     o_logit, v_logit, q_standard, k_standard, **kw)
      if record and not module.is_initializing():
        from h_write_balance_probe import read_gate_statistics
        multipliers = kw.get('static_gates')
        if multipliers is None:
          multipliers = tuple(jnp.ones_like(q_logit) for _ in range(4))
        module.sow('intermediates', 'read_gate_probe', read_gate_statistics(
            (q_logit, k_logit, v_logit, o_logit), multipliers, out[:4], kw['qk_cols']))
      return out

    bam_pallas.read = modified
    try:
      return next_fun(*args, **kwargs)
    finally:
      bam_pallas.read = original

  with nn.intercept_methods(intercept):
    yield


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-dir', type=Path, required=True)
  parser.add_argument('--exp', required=True)
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--cohort', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--high-start', type=int, default=24)
  parser.add_argument('--sequences', type=int, default=32)
  parser.add_argument('--length', type=int, default=2048)
  parser.add_argument('--vmem-mib', type=int, default=40)
  args = parser.parse_args()
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir/'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils
  import pyconfig
  from layers.models import Transformer
  from h_write_balance_probe import restore_params
  assert jax.process_count() == 1
  (args.output/'configuration'/'read-gate-ablation').mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'],
      exp_class=args.exp, run_name='read-gate-ablation',
      base_output_directory=str(args.output/'configuration')+'/',
      enable_checkpointing=False, only_eval=True, dataset_type='synthetic',
      target_train_batch_size=None, per_device_batch_size=1.,
      max_target_length=args.length, max_prefill_predict_length=args.length,
      record_internal_nn_metrics=False, record_training_health_metrics=False,
      jax_cache_dir='', log_config=False)
  cfg.get_keys().update(bam_pallas_read_tile=128, bam_pallas_write_tile=128,
                        bam_pallas_vmem_mib=args.vmem_mib)
  assert cfg.bam_pallas_core and cfg.bam_pallas_body == 'v8t'
  assert cfg.bam_read_gate_activation == 'sigmoid'
  initial = float(cfg.bam_read_gate_init)
  names, controls = intervention_matrix(cfg.num_decoder_layers, args.high_start)
  cohort = np.load(args.cohort)['tokens'][:args.sequences, :args.length+1]
  assert cohort.shape == (args.sequences, args.length+1)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  positions = jnp.arange(args.length, dtype=jnp.int32)[None]
  segments = jnp.ones((1, args.length), jnp.int32)
  traces = 0

  def forward(params, tokens, control):
    nonlocal traces
    traces += 1
    print('FORWARD_TRACE', traces, flush=True)
    with intervene_read_gates(control, initial):
      xent, _, _ = model.apply(params, tokens[:, :-1], positions,
          decoder_segment_ids=segments, decoder_target_mask=segments,
          decoder_target_tokens=tokens[:, 1:], enable_dropout=False)
    # Use the exact model's token cross entropy, not a reconstructed output head.
    return jnp.mean(xent.astype(jnp.float32))

  def original_forward(params, tokens):
    xent, _, _ = model.apply(params, tokens[:, :-1], positions,
        decoder_segment_ids=segments, decoder_target_mask=segments,
        decoder_target_tokens=tokens[:, 1:], enable_dropout=False)
    return jnp.mean(xent.astype(jnp.float32))

  start = time.monotonic()
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    print('RESTORE_BEGIN', args.checkpoint, flush=True)
    params = jax.device_put(unfreeze(nn.unbox(restore_params(args.checkpoint))),
                           jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    jax.block_until_ready(params)
    print('RESTORE_OK', round(time.monotonic()-start, 2), flush=True)
    compiled = jax.jit(forward)
    losses = np.empty((len(names), args.sequences), np.float64)
    # Same executable and weights for all interventions; only runtime control differs.
    for variant, name in enumerate(names):
      before = time.monotonic()
      for index, tokens in enumerate(cohort):
        losses[variant, index] = float(compiled(params, jnp.asarray(tokens[None]),
                                               jnp.asarray(controls[variant])))
      assert np.isfinite(losses[variant]).all()
      print('VARIANT_DONE', name, float(losses[variant].mean()),
            'seconds', round(time.monotonic()-before, 2), flush=True)
    # Detect mutable state / accidental cross-variant contamination.
    repeated = float(compiled(params, jnp.asarray(cohort[0:1]), jnp.asarray(controls[0])))
    assert repeated == losses[0, 0], (repeated, losses[0, 0])
    reference = jax.jit(original_forward)
    no_op_deltas = [float(reference(params, jnp.asarray(cohort[i:i+1])))-losses[0,i]
                    for i in range(2)]
    assert max(map(abs,no_op_deltas)) < 1e-5, no_op_deltas
  np.savez_compressed(args.output/'paired-losses.npz', losses=losses, controls=controls)
  meta = dict(exp=args.exp, checkpoint=args.checkpoint, high_start=args.high_start,
      upper_layers=list(range(args.high_start, cfg.num_decoder_layers)),
      names=names, mean_loss=losses.mean(1).tolist(),
      mean_delta=(losses-losses[0]).mean(1).tolist(),
      sequence_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in cohort],
      cohort_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
      initial_gate=initial, baseline_repeat_exact=True, forward_traces=traces,
      original_vs_noop_deltas=no_op_deltas,
      sequences=args.sequences, length=args.length, seconds=time.monotonic()-start,
      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      runtime_commit=__import__('subprocess').check_output(
          ['git','-C',str(args.runtime_dir),'rev-parse','HEAD'],text=True).strip(),
      overrides=dict(batch=1, read_write_tile=128, vmem_mib=args.vmem_mib,
                     only_eval=True, enable_dropout=False),
      semantics='Paired per-sequence token CE. Only dynamic Q/K gates intervened; static addresses/gates, V/O, QKNorm, weights and writes unchanged as functions. Downstream activations respond normally.')
  (args.output/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
  print('ABLATION_COMPLETE', json.dumps(meta['mean_delta']), flush=True)


if __name__ == '__main__':
  main()
