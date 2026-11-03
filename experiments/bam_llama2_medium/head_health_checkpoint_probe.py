"""Compact paired-checkpoint logit health on a frozen, hashed Pile cohort.

Restore parameters on the TPU, retain each arm's exact runtime sources, and
return only sampled logit moments. The ignored model CE/head is eliminated by
JIT; no full vocabulary logits or hidden-state captures leave the device.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('runtime-dir', 'cohort', 'output', 'metrics-source'):
    parser.add_argument('--' + name, type=Path, required=True)
  for name in ('runtime-commit', 'exp', 'checkpoint-root', 'steps'):
    parser.add_argument('--' + name, required=True)
  parser.add_argument('--sample-size', type=int, default=65536)
  parser.add_argument('--gcs-output', help='Publish each compact result as soon as its checkpoint finishes.')
  args = parser.parse_args()
  if args.sample_size <= 0 or args.sample_size > 65536 or args.sample_size % 32:
    parser.error('--sample-size must be a positive multiple of 32, at most 65536')
  actual = subprocess.check_output(
      ['git', '-C', str(args.runtime_dir), 'rev-parse', 'HEAD'], text=True).strip()
  assert actual == args.runtime_commit
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir / 'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils
  import pyconfig
  from layers.models import Transformer

  spec = importlib.util.spec_from_file_location('probe_health_metrics', args.metrics_source)
  metrics = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(metrics)
  assert jax.process_count() == 1
  args.output.mkdir(parents=True, exist_ok=True)
  (args.output / 'configuration' / 'head-health').mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=args.exp,
      run_name='head-health', base_output_directory=str(args.output/'configuration')+'/',
      enable_checkpointing=False, only_eval=True, dataset_type='synthetic',
      target_train_batch_size=None, per_device_batch_size=1., max_target_length=2048,
      max_prefill_predict_length=2048, record_internal_nn_metrics=False,
      record_training_health_metrics=False, jax_cache_dir='', log_config=False)
  cfg.get_keys().update(bam_pallas_core=False, bam_pallas_write=False,
      bam_splash_attention=False, remat_policy='none', matmul_precision='highest')
  for flag in ('bam_pallas_packed_read_keys', 'bam_pallas_fused_qk_norm',
               'record_logits_health_metrics'):
    if flag in cfg.get_keys():
      cfg.get_keys()[flag] = False
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  tokens = np.load(args.cohort)['tokens'][:32, :2049]
  assert tokens.shape == (32, 2049)
  selected = metrics.health_sample_positions(32, 2048, args.sample_size)
  # Equal stratified coverage/shape permits one forward compilation per arm.
  # The default includes every token, matching the earlier frozen-head probes.
  positions_by_sequence = np.array([
      [p % 2048 for p in selected if p // 2048 == i] for i in range(32)], np.int32)
  assert positions_by_sequence.shape == (32, args.sample_size // 32)
  positions = jnp.arange(2048, dtype=jnp.int32)[None]
  mask = jnp.ones((1, 2048), jnp.int32)
  trace_count = 0

  def forward(params, batch, sample_positions):
    nonlocal trace_count
    trace_count += 1
    captured = {}

    def intercept(next_fun, call_args, kwargs, context):
      result = next_fun(*call_args, **kwargs)
      if (context.method_name == '__call__' and context.module.name == 'decoder_norm'
          and 'lm_head' in context.module.scope.path):
        captured['h'] = result
      return result

    with nn.intercept_methods(intercept):
      model.apply(params, batch[:, :-1], positions, decoder_segment_ids=mask,
          decoder_target_mask=mask, decoder_target_tokens=batch[:, 1:], enable_dropout=False)
    hidden = captured['h']
    head = params['params']['decoder']['lm_head']['logits_dense'].astype(jnp.bfloat16)
    high = jnp.matmul(hidden[0, sample_positions].astype(jnp.bfloat16), head,
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    def moments(logits):
      mean = jnp.mean(logits, axis=-1)
      std = jnp.sqrt(jnp.mean((logits - mean[:, None]) ** 2, axis=-1))
      return jnp.stack((mean, std, jnp.ones_like(mean)), axis=-1)

    return dict(accumulator=moments(high),
                bf16_output=moments(high.astype(jnp.bfloat16).astype(jnp.float32)),
                hidden_mean_square=jnp.mean(hidden.astype(jnp.float32) ** 2))

  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    probe = jax.jit(forward)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    for step in map(int, args.steps.split(',')):
      start = time.monotonic()
      checkpoint = args.checkpoint_root.rstrip('/') + f'/{step}/items'
      print('RESTORE_BEGIN', step, checkpoint, flush=True)
      cp = ocp.PyTreeCheckpointer()
      tree = cp.metadata(checkpoint).item_metadata.tree['params']
      abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(
          x.shape, x.dtype, sharding=sharding), tree)
      restored = cp.restore(checkpoint, item={'params': abstract}, transforms={},
          restore_args={'params': ocp.checkpoint_utils.construct_restore_args(abstract)})
      cp.close()
      params = jax.device_put(unfreeze(nn.unbox(restored['params'])), sharding)
      del restored
      jax.block_until_ready(params)
      restore_seconds = time.monotonic() - start
      print('RESTORE_OK', step, restore_seconds, flush=True)
      records = []
      for i in range(32):
        record = jax.tree.map(np.asarray, probe(params, jnp.asarray(tokens[i:i+1]),
                                               jnp.asarray(positions_by_sequence[i])))
        assert all(np.isfinite(v).all() for v in record.values())
        records.append(record)
        if i in (0, 7, 15, 31):
          print('SEQUENCE_OK', step, i + 1, round(time.monotonic() - start, 2), flush=True)
      summaries = {}
      for name in ('accumulator', 'bf16_output'):
        samples = np.concatenate([r[name] for r in records])
        values = np.asarray(metrics.summarize_health(jnp.asarray(samples)))
        summaries[name] = dict(common_offset_abs_p95=float(values[0]),
                               centered_std_median=float(values[1]))
      result = dict(exp=args.exp, checkpoint=checkpoint, runtime_commit=actual,
          runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          metrics_source_sha256=hashlib.sha256(args.metrics_source.read_bytes()).hexdigest(),
          cohort_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
          sequence_hashes=[hashlib.sha256(t.tobytes()).hexdigest() for t in tokens],
          sequence_count=32, length=2048, sample_size=len(selected),
          selected_flat_positions=selected, metrics=summaries,
          post_final_norm_rms=float(np.sqrt(np.mean([r['hidden_mean_square'] for r in records]))),
          backend='Native JAX BF16 backbone, Pallas/Splash OFF; shared FP32 accumulator metric and BF16-output control.',
          restore_seconds=restore_seconds, seconds=time.monotonic()-start,
          forward_traces=trace_count, parameter_count=sum(a.size for a in jax.tree.leaves(params)))
      sample_path = args.output / f'samples-{step}.npz'
      summary_path = args.output / f'health-{step}.json'
      np.savez_compressed(sample_path,
          **{name: np.concatenate([r[name] for r in records]) for name in ('accumulator', 'bf16_output')})
      summary_path.write_text(json.dumps(result, indent=2)+'\n')
      if args.gcs_output:
        subprocess.run(['gsutil', '-q', 'cp', str(summary_path), str(sample_path),
                        args.gcs_output.rstrip('/') + '/'], check=True)
      print('CHECKPOINT_COMPLETE', step, json.dumps(summaries), flush=True)
      del params
      gc.collect()


if __name__ == '__main__':
  main()
