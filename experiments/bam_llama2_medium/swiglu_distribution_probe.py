"""Read-only ordinary-MLP u/v distributions across immutable XL V5 checkpoints.

Keep the historical scan/Pallas/Splash forward. Intercept the two actual dense
outputs and return compact statistics only. SiTU comparisons below hold those
inputs fixed; they do not change the model's output or measure a retrained loss.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

THRESHOLDS = (.25, .5, 1., 2., 4., 7., 8., 10., 16., 25.)
CANDIDATES = tuple((g, v) for g in (2., 4., 8.) for v in (2., 4., 8., 25.))
MOMENTS = ('count', 'mean', 'mean_square', 'min', 'max', 'negative_fraction', 'nonfinite_fraction')
SOFT_SUMS = ('delta_h2', 'new_h2', 'new_du2', 'new_dv2', 'new_gate2', 'new_up2',
             'old_h2', 'old_du2', 'old_dv2', 'old_gate2', 'old_up2')


def histogram_edges():
  import numpy as np
  positive = np.geomspace(1e-5, 128., 257).astype(np.float32)
  return np.concatenate(([-np.inf], -positive[::-1], [0.], positive, [np.inf])).astype(np.float32)


def activation_statistics(u, v, valid, phase, sample_count=16384):
  import jax
  import jax.numpy as jnp
  u, v = u.astype(jnp.float32), v.astype(jnp.float32)
  valid = jnp.broadcast_to(valid[..., None], u.shape)
  count = jnp.sum(valid)
  mean = lambda x: jnp.sum(jnp.where(valid, x, 0)) / count
  rows, histograms, units = [], [], []
  edges = jnp.asarray(histogram_edges())
  flat_indexes = ((jnp.arange(sample_count, dtype=jnp.uint32) * 104729
                  + phase.astype(jnp.uint32) * 15485863) % u.size).astype(jnp.int32)
  sample_valid = valid.reshape(-1)[flat_indexes]
  for i, a in enumerate((u, v)):
    rows.append(jnp.stack((count, mean(a), mean(a*a),
        jnp.min(jnp.where(valid, a, jnp.inf)), jnp.max(jnp.where(valid, a, -jnp.inf)),
        mean(a < 0), mean(~jnp.isfinite(a)))))
    sampled = a.reshape(-1)[flat_indexes]
    bins = jnp.searchsorted(edges, sampled, side='right') - 1
    histograms.append(jnp.bincount(bins, weights=sample_valid.astype(jnp.int32), length=len(edges)-1))
    magnitude = a if i == 0 else jnp.abs(a)
    maximum = jnp.max(jnp.where(valid, magnitude, -jnp.inf), axis=(0, 1))
    top, indexes = jax.lax.top_k(maximum, 16)
    rms = jnp.sqrt(jnp.sum(jnp.where(valid, a*a, 0), axis=(0, 1)) / jnp.sum(valid, axis=(0, 1)))
    units.append(jnp.stack((indexes, top, rms[indexes]), axis=-1))
  tails = jnp.stack([jnp.stack([mean(a > c) for c in THRESHOLDS])
                    for a in (u, jnp.abs(u), jnp.abs(v))])
  s = jax.nn.sigmoid(u)
  old_gate, old_up = u*s, v
  old_h = old_gate*v
  old_du, old_dv = (s+u*s*(1-s))*v, old_gate
  squares = lambda x: jnp.sum(jnp.where(valid, x*x, 0))
  old = [squares(x) for x in (old_h, old_du, old_dv, old_gate, old_up)]
  gate_cache, up_cache = {}, {}
  for g, b in CANDIDATES:
    if g not in gate_cache:
      t = jnp.tanh(u/g)
      gate_cache[g] = (g*t*s, (1-t*t)*s+g*t*s*(1-s))
    if b not in up_cache:
      t = jnp.tanh(v/b)
      up_cache[b] = (b*t, 1-t*t)
  soft = []
  for g, b in CANDIDATES:
    gate, dg = gate_cache[g]
    up, dv = up_cache[b]
    hidden = gate*up
    soft.append(jnp.stack([squares(x) for x in
        (hidden-old_h, hidden, dg*up, gate*dv, gate, up)] + old))
  return dict(moments=jnp.stack(rows), histogram=jnp.stack(histograms),
              tails=tails, soft_sums=jnp.stack(soft), top_units=jnp.stack(units))


@contextlib.contextmanager
def capture(valid, phase, sample_count):
  from flax import linen as nn
  from layers import linears
  stack = []

  def intercept(next_fun, args, kwargs, ctx):
    module = ctx.module
    ordinary = (ctx.method_name == '__call__' and isinstance(module, linears.MlpBlock)
                and module.name == 'mlp')
    if ordinary:
      bank = {}
      stack.append((module.scope.path, bank))
      try:
        out = next_fun(*args, **kwargs)
      finally:
        stack.pop()
      if not module.is_initializing():
        assert set(bank) == {'wi_0', 'wi_1'}, (module.scope.path, bank.keys())
        for name, value in activation_statistics(bank['wi_0'], bank['wi_1'], valid, phase, sample_count).items():
          module.sow('intermediates', 'swiglu_uv_' + name, value)
      return out
    out = next_fun(*args, **kwargs)
    if (stack and ctx.method_name == '__call__' and isinstance(module, linears.DenseGeneral)
        and module.scope.path[:-1] == stack[-1][0] and module.name in ('wi_0', 'wi_1')):
      stack[-1][1][module.name] = out
    return out

  with nn.intercept_methods(intercept):
    yield


def extract(collection, name, layers):
  import jax.numpy as jnp
  from flax.traverse_util import flatten_dict
  found = []
  for path, value in flatten_dict(collection).items():
    if path[-1] != 'swiglu_uv_' + name:
      continue
    while isinstance(value, (tuple, list)):
      assert len(value) == 1
      value = value[0]
    if 'layers' in path:
      assert value.shape[0] == layers-1, (path, value.shape)
      found.append((0, value))
    elif 'final_local_layer' in path:
      found.append((layers-1, value[None]))
    else:
      raise ValueError(path)
  found.sort(key=lambda item: item[0])
  assert [x[0] for x in found] == [0, layers-1], [x[0] for x in found]
  return jnp.concatenate([x[1] for x in found], axis=0)


def restore_params(path):
  import jax
  import orbax.checkpoint as ocp
  cp = ocp.PyTreeCheckpointer()
  sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
  tree = cp.metadata(path).item_metadata.tree['params']
  abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sharding), tree)
  restored = cp.restore(path, item={'params': abstract}, transforms={},
                        restore_args={'params': ocp.checkpoint_utils.construct_restore_args(abstract)})
  cp.close()
  return restored['params']


def summarize(arrays):
  import numpy as np
  m = arrays['moments']  # sequence, layer, branch, moment
  counts = m[..., 0]
  weight = lambda x: np.sum(x * counts, axis=0) / np.sum(counts, axis=0)
  summary = dict(mean=weight(m[..., 1]), rms=np.sqrt(weight(m[..., 2])),
                 minimum=m[..., 3].min(0), maximum=m[..., 4].max(0),
                 negative_fraction=weight(m[..., 5]), nonfinite_fraction=weight(m[..., 6]),
                 histogram=arrays['histogram'].sum(0), tails=arrays['tails'].mean(0))
  soft = arrays['soft_sums'].sum(0)
  ratio = lambda n, d: np.sqrt(soft[..., n] / np.maximum(soft[..., d], 1e-30))
  summary['soft'] = dict(hidden_relative_change=ratio(0, 6), hidden_rms_ratio=ratio(1, 6),
      du_multiplier_rms_ratio=ratio(2, 7), dv_multiplier_rms_ratio=ratio(3, 8),
      gate_rms_ratio=ratio(4, 9), up_rms_ratio=ratio(5, 10))
  # Histogram quantiles are estimates; raw extrema/tail fractions use all entries.
  hist, edges = summary['histogram'], histogram_edges()
  cumulative = np.cumsum(hist, axis=-1)
  quantiles = {}
  for q in (.001, .01, .5, .9, .99, .999, .9999):
    idx = np.argmax(cumulative >= q*cumulative[..., -1, None], axis=-1)
    quantiles[str(q)] = np.stack((edges[idx], edges[idx+1]), -1)
  summary['signed_quantile_intervals'] = quantiles
  def convert(x):
    if isinstance(x, dict):return {k:convert(v) for k,v in x.items()}
    return np.asarray(x).tolist()
  return convert(summary)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--runtime-dir', type=Path, required=True)
  p.add_argument('--runtime-commit', required=True)
  p.add_argument('--checkpoints', type=Path, required=True, help='JSON list of {step,path} (items path)')
  p.add_argument('--cohort', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--sequences', type=int, default=32)
  p.add_argument('--sample-count', type=int, default=16384)
  p.add_argument('--upload-root', help='Upload each complete phase immediately (worker -> GCS)')
  args = p.parse_args()
  assert subprocess.check_output(['git','-C',str(args.runtime_dir),'rev-parse','HEAD'],text=True).strip() == args.runtime_commit
  os.chdir(args.runtime_dir)
  sys.path.insert(0, str(args.runtime_dir/'MaxText'))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from flax.core import unfreeze
  import max_utils, pyconfig
  from layers.models import Transformer
  assert jax.process_count() == 1
  (args.output/'configuration'/'uv-distribution').mkdir(parents=True, exist_ok=True)
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class='BamXLPropV5FastKernelFullRemat',
      run_name='uv-distribution', base_output_directory=str(args.output/'configuration')+'/',
      enable_checkpointing=False, only_eval=True, dataset_type='synthetic',
      target_train_batch_size=None, per_device_batch_size=1.,
      record_internal_nn_metrics=False, record_training_health_metrics=False,
      jax_cache_dir='', log_config=False)
  assert tuple(cfg.mlp_activations) == ('silu','linear') and cfg.mgate_dim < 2 and not cfg.fused_mlp
  # Keep formal N20/V40 read shapes and scan. Use the previously validated XL single-layer v6e tile/budget.
  cfg.get_keys().update(bam_pallas_read_tile=128, bam_pallas_write_tile=128, bam_pallas_vmem_mib=63)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quant=None)
  names = ('moments','histogram','tails','soft_sums','top_units')
  data = np.load(args.cohort)
  batch = {k:data[k][:args.sequences] for k in data.files if k != 'source_indices'}
  n = len(batch['inputs'])
  assert n == args.sequences and batch['inputs'].shape[1] == cfg.max_target_length

  def apply(params, d, phase, measured):
    with capture(d['inputs_segmentation'] != 0, phase, args.sample_count) if measured else contextlib.nullcontext():
      result, collections = model.apply(params, d['inputs'], d['inputs_position'],
          decoder_segment_ids=d['inputs_segmentation'], decoder_target_mask=d['targets_segmentation'],
          decoder_target_tokens=d['targets'], enable_dropout=False,
          rngs={'dropout':jax.random.PRNGKey(0),'params':jax.random.PRNGKey(0)}, mutable=['intermediates'])
    mask = d['targets_segmentation'] != 0
    loss = jnp.sum(result[0]*mask) / jnp.sum(mask)
    if not measured:return loss
    return loss, {k:extract(collections['intermediates'],k,cfg.num_decoder_layers) for k in names}

  measured = jax.jit(lambda p,d,phase:apply(p,d,phase,True))
  plain = jax.jit(lambda p,d:apply(p,d,jnp.int32(0),False))
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    for checkpoint in json.loads(args.checkpoints.read_text()):
      step = checkpoint['step']
      output = args.output / str(step)
      if (output/'summary.json').exists():
        print('PHASE_ALREADY_COMPLETE', step, flush=True)
        continue
      output.mkdir(parents=True, exist_ok=True)
      started = time.monotonic()
      print('RESTORE_BEGIN', step, checkpoint['path'], flush=True)
      params = jax.device_put(unfreeze(nn.unbox(restore_params(checkpoint['path']))),
                             jax.sharding.SingleDeviceSharding(jax.devices()[0]))
      jax.block_until_ready(params)
      print('RESTORE_OK', step, time.monotonic()-started, flush=True)
      rows, losses, fidelity = {k:[] for k in names}, [], []
      for i in range(n):
        d = {k:jnp.asarray(v[i:i+1]) for k,v in batch.items()}
        loss, stats = jax.device_get(measured(params,d,jnp.int32(i)))
        assert stats['moments'].shape == (28,2,len(MOMENTS)), stats['moments'].shape
        assert np.max(stats['moments'][...,6]) == 0, (step,i,'nonfinite activation')
        for k in names:rows[k].append(stats[k])
        losses.append(float(loss))
        if i < 2:
          ref = float(plain(params,d));delta = float(loss)-ref
          fidelity.append(dict(sequence=i,plain_ce=ref,measured_ce=float(loss),delta=delta))
          assert abs(delta) < .003, ('instrumentation changes CE',step,i,delta)
          print('NOOP_CHECK',step,i,delta,flush=True)
        if i in (0,7,15,n-1):print('FORWARD_OK',step,i+1,time.monotonic()-started,flush=True)
      arrays = {k:np.stack(v) for k,v in rows.items()}
      np.savez_compressed(output/'statistics.npz',**arrays,losses=losses,edges=histogram_edges())
      result = dict(exp='BamXLPropV5FastKernelFullRemat', checkpoint=checkpoint,
          model_runtime=args.runtime_commit, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          cohort_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
          sequence_hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in batch['inputs']],
          sequences=n,length=cfg.max_target_length,sample_count_per_sequence_layer_branch=args.sample_count,
          histogram_is_sampled=True, extrema_moments_and_tails_use_all_valid_entries=True,
          moments=MOMENTS,tail_branches=('positive_u','abs_u','abs_v'),thresholds=THRESHOLDS,
          candidates=CANDIDATES,soft_sum_fields=SOFT_SUMS,no_op_controls=fidelity,
          precision='Formal BF16 forward, FP32 reductions; one-device probe, not formal-mesh spike reproduction',
          config=dict(scan_layers=cfg.scan_layers,read_tile=128,write_tile=128,vmem_mib=63),
          calibration=summarize({k:v[:n//2] for k,v in arrays.items()}),
          holdout=summarize({k:v[n//2:] for k,v in arrays.items()}),
          all_sequences=summarize(arrays),mean_ce=float(np.mean(losses)),seconds=time.monotonic()-started)
      (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
      if args.upload_root:subprocess.run(['gsutil','-m','rsync','-r',str(output),args.upload_root.rstrip('/')+'/'+str(step)+'/'],check=True)
      print('PHASE_COMPLETE',step,result['seconds'],flush=True)
      del params


if __name__ == '__main__':
  main()
