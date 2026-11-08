"""Read-only scanned XL checkpoint cotangents and same-tensor Splash/JAX VJP checks.

Imports model code from the specified immutable runtime checkout. No optimizer
updates or checkpoint writes; only an explicit embedding-address-bias derivative
is requested alongside zero-valued additive probes. Their gradients expose exact
cotangents without TPU host callbacks; model parameters and scan stay unchanged.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def native_attention(q, k, v):
  """FP32 strict-causal reference, including the production zero-valued sink."""
  import jax
  import jax.numpy as jnp
  q, k, v = (jnp.transpose(x.astype(jnp.float32), (0, 2, 1, 3)) for x in (q, k, v))
  t = q.shape[2]
  tile = min(128, t)
  assert t % tile == 0
  sources = jnp.arange(t)

  def body(_, index):
    query = jax.lax.dynamic_slice_in_dim(q, index * tile, tile, axis=2)
    logits = jnp.einsum('bnhd,bnsd->bnhs', query, k, precision=jax.lax.Precision.HIGHEST)
    allowed = sources[None, :] < (index * tile + jnp.arange(tile))[:, None]
    masked = jnp.where(allowed, logits, -jnp.inf)
    maximum = jnp.maximum(jnp.max(masked, axis=-1, keepdims=True), -1e9)
    weights = jnp.exp(masked - maximum)
    denom = weights.sum(axis=-1, keepdims=True) + jnp.exp(-1e9 - maximum)
    result = jnp.einsum('bnhs,bnsd->bnhd', weights / denom, v,
                        precision=jax.lax.Precision.HIGHEST)
    return None, result

  _, chunks = jax.lax.scan(jax.checkpoint(body), None, jnp.arange(t // tile))
  result = jnp.transpose(chunks, (1, 2, 0, 3, 4)).reshape(q.shape[:2] + (t, v.shape[-1]))
  return jnp.transpose(result, (0, 2, 1, 3)).astype(q.dtype)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--runtime-dir', type=Path, required=True)
  parser.add_argument('--runtime-commit', required=True)
  parser.add_argument('--exp', default='BamXLPropV5FastKernel')
  parser.add_argument('--checkpoint-root', required=True)
  parser.add_argument('--step', type=int, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--target-layer', type=int, default=1)
  parser.add_argument('--max-samples', type=int, default=32)
  parser.add_argument('--empty-row-check-only', action='store_true',
                      help='Check the exact zero-gradient invariant for an empty-history query')
  args = parser.parse_args()
  actual = subprocess.check_output(['git', '-C', str(args.runtime_dir), 'rev-parse', 'HEAD'], text=True).strip()
  assert actual == args.runtime_commit, (actual, args.runtime_commit)
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
  from input_pipeline._pile_data_processing import PileDatasets
  from layers import attentions, fusion
  from layers.models import Transformer

  assert jax.process_count() == 1
  args.output.mkdir(parents=True, exist_ok=True)
  (args.output / 'configuration/gradient-localization').mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=args.exp,
      run_name='gradient-localization', base_output_directory=str(args.output / 'configuration') + '/',
      enable_checkpointing=False, only_eval=True, dataset_type='synthetic',
      target_train_batch_size=None, per_device_batch_size=1.,
      record_internal_nn_metrics=False, record_training_health_metrics=False,
      jax_cache_dir='', log_config=False)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  assert not cfg.mix_attn, 'Reference currently requires the formal all-one attention segments'
  if args.empty_row_check_only:
    from layers.models import get_remat_policy
    t, n = cfg.max_target_length, cfg.num_query_heads
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    q, k = [jax.random.normal(key, (1, t, n, cfg.bam_partial_rope_nope_dim + cfg.bam_standard_qk_dim),
                             dtype=jnp.bfloat16) for key in keys[:2]]
    q = q / jnp.sqrt(jnp.float32(cfg.head_dim)).astype(q.dtype)
    v = jax.random.normal(keys[2], (1, t, n, cfg.head_dim), dtype=jnp.bfloat16)
    dy = jnp.zeros_like(v).at[:, 0].set(1)
    segments = jnp.ones((1, t), jnp.int32)
    results = []
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      for fused, remat, normalized in [(True, 'full_save_splash', False),
                                      (True, 'full', False),
                                      (False, 'full_save_splash', False),
                                      (True, 'full_save_splash', True)]:
        cfg.get_keys().update(sa_use_fused_bwd_kernel=fused, remat_policy=remat)
        def objective(q, k, v):
          def attention(q, k, v):
            if normalized:
              raw_q = q.astype(jnp.float32) * jnp.sqrt(jnp.float32(cfg.head_dim))
              q = (raw_q * jax.lax.rsqrt(jnp.mean(raw_q ** 2, axis=-1, keepdims=True)
                   + cfg.normalization_layer_epsilon) / jnp.sqrt(jnp.float32(cfg.head_dim))).astype(q.dtype)
              raw_k = k.astype(jnp.float32)
              k = (raw_k * jax.lax.rsqrt(jnp.mean(raw_k ** 2, axis=-1, keepdims=True)
                   + cfg.normalization_layer_epsilon)).astype(k.dtype)
            return attentions._bam_splash_attention(q, k, v, segments, cfg, mesh, window=t,
                seq_minor=attentions._bam_splash_seq_minor(cfg, False))
          y = jax.checkpoint(attention, policy=get_remat_policy(cfg), prevent_cse=False)(q, k, v)
          return jnp.sum(y.astype(jnp.float32) * dy), jnp.max(jnp.abs(y[:, 0].astype(jnp.float32)))
        def check(q, k, v):
          (loss, first_max), gs = jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True)(q, k, v)
          return jnp.stack([loss, first_max] + [jnp.max(jnp.abs(g.astype(jnp.float32))) for g in gs])
        started = time.monotonic()
        values = list(map(float, np.asarray(jax.jit(check)(q, k, v))))
        result = dict(fused=fused, remat=remat, qknorm=normalized, loss=values[0],
                      first_output_absmax=values[1], dq_dk_dv_absmax=values[2:],
                      seconds=time.monotonic() - started)
        results.append(result)
        print('EMPTY_ROW_CHECK', json.dumps(result), flush=True)
    (args.output / 'empty-row-check.json').write_text(json.dumps(results, indent=2) + '\n')
    return
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
  jax.block_until_ready(params)
  bias = params['params']['decoder']['embedding_bam_write']['W_emb_v_up']['bias']
  print('RESTORE_OK', sum(x.size for x in jax.tree.leaves(params)), flush=True)

  cursor_uri = args.checkpoint_root.rstrip('/') + f'/{args.step}/skip_file_and_step.json'
  with tf.io.gfile.GFile(cursor_uri) as handle:
    cursor = json.load(handle)
  assert cursor['checkpoint_step'] == args.step
  local_cursor = dict(cursor, file_in_data=0, num_infeed_hosts=1)
  # Retain the process-independent global stream/skip count; one diagnostic host
  # receives all128 sequences, then forwards only one sequence at a time.
  cohort_path = args.output / 'cohort.npz'
  if cohort_path.exists():
    with np.load(cohort_path) as saved:
      batch = {key: saved[key] for key in saved.files}
  else:
    iterator = PileDatasets(mesh=mesh, path=cursor['cur_files'],
        num_infeed_hosts=1, batch_size=cursor['global_batch_size'],
        seq_len=cfg.max_target_length, seed=cursor['seed'], task_features=cfg.task_features,
        shuffle_buffer_size=cfg.shuffle_buffer_size, pad_id=cfg.pad_id,
        iter_file_nums=cursor['iter_file_nums'], meta_dict=local_cursor,
        only_eval=False, mix_attn=cfg.mix_attn)
    batch = jax.device_get(next(iterator))
    np.savez_compressed(cohort_path, **batch)
  order = np.random.default_rng(0).permutation(batch['inputs'].shape[0])[:args.max_samples]
  metadata = dict(exp=args.exp, runtime_commit=actual, checkpoint=checkpoint,
      cursor=cursor, sequence_order=order.tolist(), length=cfg.max_target_length,
      diagnostic_batch=1, formal_global_batch=cursor['global_batch_size'],
      scope='Frozen checkpoint; single-sequence mean CE, not a training/global-batch replay.',
      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      backend=jax.default_backend(), scan_layers=cfg.scan_layers)
  (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
  print('COHORT_READY', len(order), flush=True)

  current_probes = None
  layer_stack, attention_stack = [], []
  splash = attentions._bam_splash_attention

  def captured_splash(q, k, v, segments, config, mesh, window=None, seq_minor=False):
    layer = layer_stack[-1]
    module = attention_stack[-1]
    selected = jnp.asarray(layer) == args.target_layer
    q, k, v = (x + jnp.where(selected, current_probes[name], jnp.zeros_like(current_probes[name]))
               for x, name in zip((q, k, v), ('q', 'k', 'v')))
    for name, x in zip(('q', 'k', 'v'), (q, k, v)):
      module.sow('intermediates', 'gradient_probe_' + name, x)
    result = splash(q, k, v, segments, config, mesh, window=window, seq_minor=seq_minor)
    module.sow('intermediates', 'gradient_probe_context', result)
    return result + jnp.where(selected, current_probes['context'], jnp.zeros_like(current_probes['context']))

  attentions._bam_splash_attention = captured_splash

  def intercept(next_fun, call_args, kwargs, context):
    if isinstance(context.module, attentions.BamAttention) and context.method_name == '__call__':
      attention_stack.append(context.module)
      try:
        return next_fun(*call_args, **kwargs)
      finally:
        attention_stack.pop()
    if isinstance(context.module, fusion.SubDecoderLayer) and context.method_name == '__call__':
      layer = kwargs.get('layer_index', context.module.layer_inx)
      layer = context.module.layer_inx if layer is None else layer
      local_args, local_kwargs = list(call_args), dict(kwargs)
      local_args[0] = local_args[0] + current_probes['residual'][layer]
      if local_kwargs.get('M_in') is not None:
        local_kwargs['M_in'] = local_kwargs['M_in'] + current_probes['M'][layer]
        context.module.sow('intermediates', 'gradient_probe_M_rms', stable_rms(local_kwargs['M_in']))
      context.module.sow('intermediates', 'gradient_probe_residual_rms', stable_rms(local_args[0]))
      layer_stack.append(layer)
      try:
        return next_fun(*local_args, **local_kwargs)
      finally:
        layer_stack.pop()
    return next_fun(*call_args, **kwargs)

  def objective(p, probes, data):
    nonlocal current_probes
    current_probes = probes
    effective = unfreeze(p)
    effective['params']['decoder']['embedding_bam_write']['W_emb_v_up']['bias'] = bias + probes['bias']
    try:
      with nn.intercept_methods(intercept):
        loss, aux = train.loss_fn(model, cfg, dict(data), jax.random.PRNGKey(0), effective, is_train=True)
    finally:
      current_probes = None
    decoder = aux['intermediate_outputs']['intermediates']['decoder']
    body = decoder['layers']['block']
    captured = {name: body['self_attention']['gradient_probe_' + name][0][args.target_layer]
                for name in ('q', 'k', 'v', 'context')}
    for name in ('M', 'residual'):
      key = 'gradient_probe_' + name + '_rms'
      captured[name + '_rms'] = jnp.concatenate((body[key][0],
          jnp.reshape(decoder['final_local_layer']['block'][key][0], (1,))))
    return loss, captured

  def stable_rms(x):
    maximum = jnp.max(jnp.abs(x.astype(jnp.float32)))
    return maximum * jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32) / jnp.maximum(maximum, 1e-30))))

  def operator_stats(q, k, v, dy, observed_q, observed_k, observed_v, *, fused):
    cfg.get_keys()['sa_use_fused_bwd_kernel'] = fused
    segments = jnp.ones(q.shape[:2], jnp.int32)
    call = lambda q, k, v: splash(q, k, v, segments, cfg, mesh, window=q.shape[1],
                                 seq_minor=attentions._bam_splash_seq_minor(cfg, False))
    y, pullback = jax.vjp(call, q, k, v)
    grads = pullback(dy)
    reference = lambda q, k, v: native_attention(q, k, v).astype(jnp.bfloat16)
    ref, reference_pullback = jax.vjp(reference, q, k, v)
    refs = reference_pullback(dy)
    values = [stable_rms(y - ref), stable_rms(ref)]
    for g, expected, observed in zip(grads, refs, (observed_q, observed_k, observed_v)):
      values.extend((stable_rms(g), stable_rms(expected), stable_rms(observed),
                     stable_rms(g.astype(jnp.float32) - expected.astype(jnp.float32)),
                     stable_rms(g.astype(jnp.float32) - observed.astype(jnp.float32))))
    return jnp.stack(values)

  probes = dict(
      # The formal carried matrix is value-major: [B,T,V,K].
      M=jnp.zeros((cfg.num_decoder_layers, 1, cfg.max_target_length, cfg.bam_v, cfg.bam_k), cfg.dtype),
      residual=jnp.zeros((cfg.num_decoder_layers, 1, cfg.max_target_length, cfg.emb_dim), cfg.dtype),
      bias=jnp.zeros_like(bias))
  for name, width in [('q', cfg.bam_partial_rope_nope_dim + cfg.bam_standard_qk_dim),
                      ('k', cfg.bam_partial_rope_nope_dim + cfg.bam_standard_qk_dim),
                      ('v', cfg.head_dim), ('context', cfg.head_dim)]:
    probes[name] = jnp.zeros((1, cfg.max_target_length, cfg.num_query_heads, width), cfg.dtype)

  def measure(p, probes, data):
    (loss, captured), gradients = jax.value_and_grad(objective, argnums=1, has_aux=True)(p, probes, data)
    norms = {name: jax.vmap(stable_rms)(gradients[name]) for name in ('M', 'residual')}
    selected_gradients = {name: gradients[name] for name in ('q', 'k', 'v', 'context', 'bias')}
    return loss, captured, selected_gradients, norms

  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    compiled = jax.jit(measure)
    checks = {fused: jax.jit(partial(operator_stats, fused=fused)) for fused in (True, False)}
    for ordinal, sequence in enumerate(order):
      folder = args.output / f'sequence-{int(sequence):03d}'
      folder.mkdir(exist_ok=True)
      data = {k: jnp.asarray(v[sequence:sequence + 1]) for k, v in batch.items()}
      started = time.monotonic()
      loss, captured, gradients, norms = jax.device_get(
          jax.block_until_ready(compiled(params, probes, data)))
      grad_bias = float(np.linalg.norm(gradients['bias'].astype(np.float64)))
      print('BACKWARD_OK', ordinal, int(sequence), float(loss), grad_bias, flush=True)
      for direction, values in [('forward', captured), ('backward', gradients)]:
        for name in ('q', 'k', 'v', 'context'):
          np.save(folder / f'{direction}-{name}.npy', np.asarray(values[name], np.float32))
      layer_rows = [dict(layer=i, M_forward_rms=float(captured['M_rms'][i]),
          residual_forward_rms=float(captured['residual_rms'][i]),
          M_cotangent_rms=float(norms['M'][i]), residual_cotangent_rms=float(norms['residual'][i]))
          for i in range(cfg.num_decoder_layers)]
      (folder / 'layer-cotangents.json').write_text(json.dumps(layer_rows, indent=2) + '\n')
      arrays = [jnp.asarray(captured[name], jnp.bfloat16) for name in ('q', 'k', 'v')]
      arrays += [jnp.asarray(gradients[name], jnp.bfloat16) for name in ('context', 'q', 'k', 'v')]
      results = {}
      for fused, check in checks.items():
        stats = np.asarray(check(*arrays))
        checks_for_arm = {}
        for i, label in enumerate(('q', 'k', 'v')):
          a, b, observed, error, observed_error = map(float, stats[2 + i * 5:7 + i * 5])
          checks_for_arm[label] = dict(splash_rms=a, reference_rms=b, observed_rms=observed,
              splash_reference_error_rms=error, splash_observed_error_rms=observed_error,
              relative_error=error / max(b, 1e-30), observed_relative_error=observed_error / max(b, 1e-30))
        results[str(fused)] = dict(fused_backward=fused, forward_error_rms=float(stats[0]),
                                  forward_reference_rms=float(stats[1]), gradients=checks_for_arm)
      record = dict(sequence=int(sequence), ordinal=ordinal, loss=float(loss),
          embedding_address_bias_grad_norm=grad_bias,
          input_sha256=hashlib.sha256(np.asarray(data['inputs']).tobytes()).hexdigest(),
          seconds=time.monotonic() - started, operators=results)
      (folder / 'comparison.json').write_text(json.dumps(record, indent=2) + '\n')
      print('OPERATOR_COMPARISON', json.dumps(record), flush=True)
      catastrophic = any(max(g['relative_error'], g['observed_relative_error']) > 10
                         for arm in results.values() for g in arm['gradients'].values())
      if catastrophic:
        print('CATASTROPHIC_OPERATOR_MISMATCH_CAPTURED', flush=True)
        break
      for path in folder.glob('*.npy'):
        path.unlink()
  print('GRADIENT_LOCALIZATION_COMPLETE', flush=True)


if __name__ == '__main__':
  main()
