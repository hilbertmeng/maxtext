"""Read-only checkpoint head-gradient amplitude/error/coherence decomposition.

Runtime sources remain at --runtime-commit. Only the vocabulary head receives a
backward-equivalent calculation; backbone parameters are restored on the TPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def head_gradient(h, weights, targets):
  """FP32 conditional gradient, with the native BF16-effective head weights."""
  import jax
  import jax.numpy as jnp
  h = h.reshape(-1, h.shape[-1]).astype(jnp.float32)
  weights = weights.astype(jnp.bfloat16).astype(jnp.float32)
  targets = targets.reshape(-1)
  logits = jnp.matmul(h, weights, precision=jax.lax.Precision.HIGHEST)
  logp = jax.nn.log_softmax(logits, axis=-1)
  error = jnp.exp(logp) - jax.nn.one_hot(targets, weights.shape[-1])
  grad = jnp.matmul(h.T, error, precision=jax.lax.Precision.HIGHEST) / targets.size
  loss = -jnp.take_along_axis(logp, targets[:, None], axis=-1).mean()
  metrics = jnp.stack((loss, jnp.mean(h*h), jnp.mean(jnp.sum(error*error, axis=-1)),
                       jnp.sum(grad*grad)))
  return grad, metrics


def factors(gradient_norm, hidden_rms, error_token_l2_rms):
  """Exact amplitude normalization; residual factor retains directional geometry."""
  return dict(gradient_norm=float(gradient_norm), hidden_rms=float(hidden_rms),
              error_token_l2_rms=float(error_token_l2_rms),
              normalized_geometry_factor=float(gradient_norm/(hidden_rms*error_token_l2_rms)))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ['runtime-dir', 'cohort', 'output']:
    parser.add_argument('--'+name, type=Path, required=True)
  for name in ['runtime-commit', 'exp', 'checkpoint-root', 'steps']:
    parser.add_argument('--'+name, required=True)
  args = parser.parse_args()
  actual = subprocess.check_output(['git', '-C', str(args.runtime_dir), 'rev-parse', 'HEAD'], text=True).strip()
  assert actual == args.runtime_commit, (actual, args.runtime_commit)
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir/'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils
  import pyconfig
  from layers.models import Transformer

  assert jax.process_count() == 1
  args.output.mkdir(parents=True, exist_ok=True)
  (args.output/'configuration/head-amplitude').mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=args.exp,
      run_name='head-amplitude', base_output_directory=str(args.output/'configuration')+'/',
      enable_checkpointing=False, only_eval=True, dataset_type='synthetic',
      target_train_batch_size=None, per_device_batch_size=1., max_target_length=2048,
      max_prefill_predict_length=2048, record_internal_nn_metrics=False,
      record_training_health_metrics=False, jax_cache_dir='', log_config=False)
  cfg.get_keys().update(bam_pallas_core=False, bam_pallas_write=False,
      bam_splash_attention=False, remat_policy='none', matmul_precision='highest')
  for flag in ['bam_pallas_packed_read_keys', 'bam_pallas_fused_qk_norm']:
    if flag in cfg.get_keys():
      cfg.get_keys()[flag] = False
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  tokens = np.load(args.cohort)['tokens'][:32, :2049]
  assert tokens.shape == (32, 2049)
  hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in tokens]
  positions = jnp.arange(2048, dtype=jnp.int32)[None]
  mask = jnp.ones((1, 2048), jnp.int32)
  trace_count = 0

  def forward(params, batch):
    nonlocal trace_count
    trace_count += 1
    print('FORWARD_TRACE', trace_count, flush=True)
    captured = {}
    def intercept(next_fun, call_args, kwargs, context):
      module = context.module
      if (context.method_name == '__call__' and module.name == 'decoder_norm'
          and 'lm_head' in module.scope.path):
        result = next_fun(*call_args, **kwargs)
        captured['h'] = result
        captured['raw_rms'] = jnp.sqrt(jnp.mean(call_args[0].astype(jnp.float32)**2))
        return result
      return next_fun(*call_args, **kwargs)
    with nn.intercept_methods(intercept):
      ce, _, _ = model.apply(params, batch[:, :-1], positions,
          decoder_segment_ids=mask, decoder_target_mask=mask,
          decoder_target_tokens=batch[:, 1:], enable_dropout=False)
    return captured['h'], jnp.mean(ce.astype(jnp.float32)), captured['raw_rms']

  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules), jax.default_matmul_precision('highest'):
    capture = jax.jit(forward)
    conditional = jax.jit(head_gradient)
    for step in map(int, args.steps.split(',')):
      folder = args.output/str(step)
      folder.mkdir(exist_ok=True)
      start = time.monotonic()
      checkpoint = args.checkpoint_root.rstrip('/')+f'/{step}/items'
      print('RESTORE_BEGIN', step, checkpoint, flush=True)
      cp = ocp.PyTreeCheckpointer()
      metadata = cp.metadata(checkpoint)
      tree = metadata.item_metadata.tree['params']
      sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
      abstract = jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), tree)
      restored = cp.restore(checkpoint, item={'params':abstract}, transforms={},
          restore_args={'params':ocp.checkpoint_utils.construct_restore_args(abstract)})
      cp.close()
      params = jax.device_put(unfreeze(nn.unbox(restored['params'])), sharding)
      del restored
      jax.block_until_ready(params)
      weights = params['params']['decoder']['lm_head']['logits_dense']
      print('RESTORE_OK', step, round(time.monotonic()-start, 2), flush=True)
      summed = None
      rows = []
      for i, batch in enumerate(tokens):
        hidden, native_ce, raw_rms = capture(params, jnp.asarray(batch[None]))
        grad, stats = conditional(hidden, weights, jnp.asarray(batch[None,1:]))
        summed = grad/32 if summed is None else summed+grad/32
        stats = np.asarray(stats)
        assert np.isfinite(stats).all()
        row = dict(sequence=i, input_sha256=hashes[i], conditional_ce=float(stats[0]),
            native_ce=float(native_ce), head_input_mean_square=float(stats[1]),
            error_token_l2_mean_square=float(stats[2]), gradient_squared_norm=float(stats[3]),
            pre_final_norm_rms=float(raw_rms))
        rows.append(row)
        np.savez_compressed(folder/f'head-input-{i:03d}.npz',
                            h=np.asarray(hidden,dtype=np.float32), targets=batch[None,1:])
        (folder/f'sequence-{i:03d}.json').write_text(json.dumps(row,indent=2)+'\n')
        if i in [0,7,15,31]:
          print('SEQUENCE_OK',step,i+1,round(time.monotonic()-start,2),flush=True)
      gradient_norm = float(jnp.linalg.norm(summed))
      hidden_rms = float(np.sqrt(np.mean([r['head_input_mean_square'] for r in rows])))
      error_rms = float(np.sqrt(np.mean([r['error_token_l2_mean_square'] for r in rows])))
      result = dict(exp=args.exp, checkpoint=checkpoint, label_step=step,
          runtime_commit=actual, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          cohort_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(), sequence_hashes=hashes,
          sequence_count=32, length=2048, backend='Native JAX BF16 forward; Pallas/Splash OFF; highest matmul precision.',
          gradient_definition='FP32 conditional head gradient on native BF16 head input and BF16-effective W; gradient of 32-text mean.',
          factors=factors(gradient_norm,hidden_rms,error_rms),
          per_text_gradient_rms=float(np.sqrt(np.mean([r['gradient_squared_norm'] for r in rows]))),
          native_ce_mean=float(np.mean([r['native_ce'] for r in rows])),
          max_fp32_vs_native_ce_difference=max(abs(r['conditional_ce']-r['native_ce']) for r in rows),
          forward_traces=trace_count,seconds=time.monotonic()-start,per_sequence=rows,
          parameter_count=sum(a.size for a in jax.tree.leaves(params)))
      (folder/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
      print('CHECKPOINT_COMPLETE',step,json.dumps(result['factors']),flush=True)
      del params, weights, grad, summed
      gc.collect()


if __name__ == '__main__':
  main()
