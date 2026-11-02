"""Read-only, per-layer/head activation gains of linear BAM write adapters.

Execute against --runtime-dir at its training commit. Only compact statistics
and subsampled scalar ratios leave the JIT; captured tensors do not alter output.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

MOMENTS = ('before_ms', 'after_ms', 'update_ms', 'before_after_dot',
           'ratio_mean', 'cosine_mean', 'expanded_fraction')


def vector_statistics(before, after, stride):
  import jax.numpy as jnp
  x, y = before.astype(jnp.float32), after.astype(jnp.float32)
  xx, yy = jnp.mean(x*x, -1), jnp.mean(y*y, -1)
  xy = jnp.mean(x*y, -1)
  gain = jnp.sqrt(yy / jnp.maximum(xx, 1e-30))
  cosine = xy / jnp.sqrt(jnp.maximum(xx*yy, 1e-30))
  avg = lambda z: jnp.mean(z, axis=1)  # [batch, head], all tokens
  moments = jnp.stack((avg(xx), avg(yy), avg(jnp.mean((y-x)**2, -1)),
                       avg(xy), avg(gain), avg(cosine), avg(gain > 1)), -1)
  samples = jnp.stack((gain[:, ::stride], cosine[:, ::stride]), -1)
  return moments, samples


@contextlib.contextmanager
def capture_adapters(stride):
  from flax import linen as nn
  from layers.bam_content_adapter import HeadContentAdapter

  def intercept(next_fun, args, kwargs, ctx):
    value = next_fun(*args, **kwargs)
    module = ctx.module
    if (ctx.method_name == '__call__' and isinstance(module, HeadContentAdapter)
        and module.name == 'content_write_adapter' and not module.is_initializing()):
      moments, samples = vector_statistics(args[0], value, stride)
      module.sow('intermediates', 'adapter_amplitude_moments', moments)
      module.sow('intermediates', 'adapter_amplitude_samples', samples)
    return value

  with nn.intercept_methods(intercept):
    yield


def extract(collection, name):
  from flax.traverse_util import flatten_dict
  found = [(path, value) for path, value in flatten_dict(collection).items() if path[-1] == name]
  assert len(found) == 1, [(p, getattr(v, 'shape', None)) for p, v in found]
  path, value = found[0]
  assert 'layers' in path, path
  while isinstance(value, (tuple, list)):
    assert len(value) == 1
    value = value[0]
  return value


def matrix_analysis(params, scan_axis):
  import numpy as np
  from flax.traverse_util import flatten_dict
  found = [v for k, v in flatten_dict(params).items()
           if k[-2:] == ('content_write_adapter', 'delta')]
  assert len(found) == 1
  delta = np.moveaxis(np.asarray(found[0], np.float32), scan_axis, 0)
  assert delta.shape == (17, 16, 75, 75), delta.shape
  matrix = delta + np.eye(75, dtype=np.float32)
  singular = np.linalg.svd(matrix, compute_uv=False)
  isotropic = np.sqrt(np.mean(np.sum(matrix**2, axis=-1), axis=-1))
  # Best scalar multiplier in Frobenius norm and its non-scalar remainder.
  scalar = np.trace(matrix, axis1=-2, axis2=-1) / 75
  residual = matrix - scalar[..., None, None] * np.eye(75, dtype=np.float32)
  anisotropy = np.sqrt(np.sum(residual**2, axis=(-1, -2)) / np.sum(matrix**2, axis=(-1, -2)))
  return dict(singular_values=singular, isotropic_gain=isotropic,
              best_scalar=scalar, non_scalar_fraction=anisotropy,
              update_over_identity=np.sqrt(np.sum(delta**2, axis=(-1,-2))/75))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  for key in ('runtime-dir', 'cohort', 'output'):
    p.add_argument('--'+key, type=Path, required=True)
  for key in ('exp', 'checkpoint'):
    p.add_argument('--'+key, required=True)
  p.add_argument('--length', type=int, default=4096)
  p.add_argument('--sequences', type=int, default=32)
  p.add_argument('--stride', type=int, default=16)
  args = p.parse_args()
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir/'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils, pyconfig
  from layers.models import Transformer
  from h_write_balance_probe import restore_params
  assert jax.process_count() == 1
  (args.output/'configuration'/'adapter-amplitude').mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'],
      exp_class=args.exp, run_name='adapter-amplitude',
      base_output_directory=str(args.output/'configuration')+'/',
      enable_checkpointing=False, only_eval=True, dataset_type='synthetic',
      target_train_batch_size=None, per_device_batch_size=1.,
      max_target_length=args.length, max_prefill_predict_length=args.length,
      record_internal_nn_metrics=False, record_training_health_metrics=False,
      jax_cache_dir='', log_config=False)
  cfg.get_keys().update(bam_pallas_read_tile=128, bam_pallas_write_tile=128,
                        bam_pallas_vmem_mib=40)
  assert cfg.bam_content_adapter_write and cfg.bam_content_adapter_rank == 0
  assert not getattr(cfg, 'bam_content_adapter_write_before_rms', False)
  assert cfg.scan_layers and cfg.num_decoder_layers == 18
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  tokens = np.load(args.cohort)['tokens'][:args.sequences, :args.length+1]
  padded_target = tokens.shape[1] == args.length
  if padded_target:
    # Existing calibration cohort preserves all input tokens but no next-token suffix.
    # The repeated final target is used only for the no-op instrumentation check.
    tokens = np.concatenate((tokens, tokens[:, -1:]), axis=1)
  assert tokens.shape == (args.sequences, args.length+1)
  pos = jnp.arange(args.length, dtype=jnp.int32)[None]
  seg = jnp.ones((1, args.length), jnp.int32)
  def apply(params, t, mutable=False):
    return model.apply(params, t[:, :-1], pos, decoder_segment_ids=seg,
        decoder_target_mask=seg, decoder_target_tokens=t[:, 1:], enable_dropout=False,
        **({'mutable': ['intermediates']} if mutable else {}))
  def forward(params, t):
    with capture_adapters(args.stride):
      result, collections = apply(params, t, True)
    return (jnp.mean(result[0].astype(jnp.float32)),
            extract(collections['intermediates'], 'adapter_amplitude_moments'),
            extract(collections['intermediates'], 'adapter_amplitude_samples'))
  start = time.monotonic()
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    print('RESTORE_BEGIN', args.checkpoint, flush=True)
    params = jax.device_put(unfreeze(nn.unbox(restore_params(args.checkpoint))),
                           jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    jax.block_until_ready(params)
    print('RESTORE_OK', time.monotonic()-start, flush=True)
    matrices = matrix_analysis(params, cfg.param_scan_axis)
    compiled = jax.jit(forward)
    reference = jax.jit(lambda p, t: jnp.mean(apply(p, t)[0].astype(jnp.float32)))
    rows, samples, losses, noops = [], [], [], []
    for i, t in enumerate(tokens):
      loss, row, sample = compiled(params, jnp.asarray(t[None]))
      row, sample = np.asarray(row), np.asarray(sample)
      assert row.shape == (17, 1, 16, len(MOMENTS)), row.shape
      assert sample.shape == (17, 1, args.length//args.stride, 16, 2), sample.shape
      assert np.isfinite(row).all() and np.isfinite(sample).all()
      rows.append(row[:, 0]); samples.append(sample[:, 0]); losses.append(float(loss))
      if i < 2:
        noops.append(float(reference(params, jnp.asarray(t[None]))) - losses[-1])
        assert abs(noops[-1]) < 1e-6, noops
      if i in (0, 7, 15, 31):
        print('FORWARD_OK', i+1, time.monotonic()-start, flush=True)
    moments, sampled = np.stack(rows), np.stack(samples)
    np.savez_compressed(args.output/'statistics.npz', moments=moments,
                        samples=sampled, losses=losses, **matrices)
    avg = moments.mean(0)
    head_gain = np.sqrt(avg[..., 1] / avg[..., 0])
    layer_gain = np.sqrt(avg[..., 1].mean(1) / avg[..., 0].mean(1))
    per_layer = []
    for l in range(17):
      per_layer.append(dict(layer=l, aggregate_gain=float(layer_gain[l]),
          head_gain=head_gain[l].tolist(),
          token_head_gain_p10_p50_p90=np.quantile(sampled[:, l, :, :, 0], [.1,.5,.9]).tolist(),
          cosine=float(avg[l, :, 5].mean()), expanded_fraction=float(avg[l, :, 6].mean()),
          isotropic_gain=matrices['isotropic_gain'][l].tolist(),
          best_scalar=matrices['best_scalar'][l].tolist(),
          non_scalar_fraction=matrices['non_scalar_fraction'][l].tolist(),
          singular_min_max=[float(matrices['singular_values'][l].min()), float(matrices['singular_values'][l].max())]))
    result = dict(exp=args.exp, checkpoint=args.checkpoint,
        runtime_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cohort_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        sequence_hashes=[hashlib.sha256(x[:args.length].tobytes()).hexdigest() for x in tokens],
        final_target_repeated_for_noop_only=padded_target,
        sequences=args.sequences, length=args.length, stride=args.stride,
        moments=MOMENTS, no_op_loss_deltas=noops, per_layer=per_layer,
        seconds=time.monotonic()-start, scope='Actual BF16 pre/post-adapter content on unchanged full forward. Aggregate gain=sqrt(sum||after||^2/sum||before||^2); head/token ratios and cosine measured separately. Spectral gains are FP32 parameter analysis, not actual activation gains. L17 has no writer/adapter.',
        overrides=dict(only_eval=True, batch=1, length=args.length, vmem_mib=40, tile=128))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DIAG_DONE', time.monotonic()-start, flush=True)

if __name__ == '__main__':
  main()
