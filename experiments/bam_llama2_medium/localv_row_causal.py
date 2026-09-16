"""Shared LocalV row: dose and diagonal/off-diagonal MHA transport ablations.

Model source is the exact training runtime. Linen interception is diagnostic-only;
LocalO's consumer of the shared answer is never altered. All token positions are
included. No claim about from-scratch retraining follows from frozen-network loss.
"""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'MaxText'))
from absl import app
from flax import linen as nn
from flax.linen import partitioning
import jax
import jax.numpy as jnp
import numpy as np
import exp
import max_utils
import pyconfig
import train
from layers import attentions

BASE = 'BamXLSharedBasisLocalVRowSharedColRank4CFp32'
TRAIN_COMMIT = '97be64f241ea5ed5596098348e501ab67fcdf4ff'
ROOT = 'gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl-localv-row-causal-20260916'
KEYS = ('inputs', 'targets', 'inputs_position', 'inputs_segmentation', 'targets_segmentation')


class LocalVRowCausalProbe(getattr(exp, BASE)):
  only_eval = True
  per_device_batch_size = 1
  eval_per_device_batch_size = 1
  load_parameters_path = ROOT + '/checkpoints/5250/items'
  record_internal_nn_metrics = False
  record_training_health_metrics = False
  bam_record_fetched_read_health_metrics = False
  bam_record_fetched_read_amplitude_metrics = False
  bam_record_fetch_route_metrics = False
  bam_record_local_routing_metrics = False
  bam_record_local_qk_amplitude_metrics = False


exp.LocalVRowCausalProbe = LocalVRowCausalProbe


def transport_edit(y, y_without_row, alpha, value, value_without_row, q0, s0, keep):
  """Split finite-precision row effect into same-position and other-position AV.

  Total is native AV minus no-row AV, using the SAME alpha. Diagonal uses the
  actual rounded V change. Cross is the remainder (including AV rounding).
  Native and whole-deletion endpoints are selected exactly in their output dtype.
  """
  width = y.shape[1]
  src_index = jnp.arange(q0, q0 + width) - s0
  diag_alpha = jnp.take_along_axis(
      alpha, jnp.broadcast_to(src_index[None, None, :, None],
                              alpha.shape[:3] + (1,)), axis=-1)[..., 0]
  dv = value.astype(jnp.float32) - value_without_row.astype(jnp.float32)
  self_part = jnp.transpose(diag_alpha, (0, 2, 1))[..., None] * dv[:, src_index]
  total = y.astype(jnp.float32) - y_without_row.astype(jnp.float32)
  cross_part = total - self_part
  changed = (y.astype(jnp.float32) + (keep[0]-1)*self_part
             + (keep[1]-1)*cross_part).astype(y.dtype)
  changed = jnp.where(jnp.all(keep == 0), y_without_row, changed)
  return jnp.where(jnp.all(keep == 1), y, changed)


@contextlib.contextmanager
def interventions(scales, route=False):
  """scales[layer, Vdose/Vself/Vcross/Qdose/Kdose]; row consumers only."""
  frames = []
  original_attention = attentions._attention_op

  def attention(query, key, value, valid, **kwargs):
    y, alpha = original_attention(query, key, value, valid, **kwargs)
    frame = frames[-1]
    if route and 'no_row_value' in frame:
      q0, s0 = frame['block']
      no_row = frame['no_row_value'][:, s0:s0 + value.shape[1]]
      y_no_row = jnp.einsum('bnqs,bsnd->bqnd', alpha, no_row)
      y = transport_edit(y, y_no_row, alpha, value, no_row, q0, s0,
                         scales[frame['layer'], 1:3])
    return y, alpha

  def intercept(next_fun, args, kwargs, ctx):
    if not isinstance(ctx.module, attentions.BamAttention):
      return next_fun(*args, **kwargs)
    method = ctx.method_name
    if method == '__call__':
      assert kwargs.get('layer_index') is not None
      frames.append(dict(layer=kwargs['layer_index'], gate_depth=0))
      try:
        return next_fun(*args, **kwargs)
      finally:
        frames.pop()
    if not frames:
      return next_fun(*args, **kwargs)
    frame = frames[-1]
    if method == '_read_local':
      answer = next_fun(*args, **kwargs)
      name = args[0] if args else kwargs['name']
      if name in ('q', 'k'):
        k = ctx.module.bam_k
        scale = scales[frame['layer'], 3 if name == 'q' else 4]
        return jnp.concatenate((answer[..., :k], answer[..., k:] * scale.astype(answer.dtype)), -1)
      return answer
    if method == 'kv_projection' and kwargs.get('proj_name') == 'value':
      answer = next_fun(*args, **kwargs)
      frame['std_value'] = answer
      return answer
    if method == '_gate_local_output':
      frame['gate_depth'] += 1
      try:
        return next_fun(*args, **kwargs)
      finally:
        frame['gate_depth'] -= 1
    if (method == '_expand_full_read' and ctx.module._row_shared_v
        and frame['gate_depth'] == 0):
      col, row = args[0]
      assert row.shape[-1] == 8 and col.shape[-1] == 64
      if route:
        without = next_fun((col, jnp.zeros_like(row)))
        frame['no_row_value'] = frame['std_value'] + without
        return next_fun(*args, **kwargs)
      return next_fun((col, row * scales[frame['layer'], 0].astype(row.dtype)))
    if method == '_attention_block':
      frame['block'] = (kwargs['q0'], kwargs['s0'])
    return next_fun(*args, **kwargs)

  if route:
    attentions._attention_op = attention
  try:
    with nn.intercept_methods(intercept):
      yield
  finally:
    attentions._attention_op = original_attention


def forward(model, params, batch, rng, scales=None, route=False):
  cm = interventions(scales, route) if scales is not None else contextlib.nullcontext()
  with cm:
    output, _ = model.apply(
        params, batch['inputs'], batch['inputs_position'],
        decoder_segment_ids=batch['inputs_segmentation'],
        decoder_target_mask=batch['targets_segmentation'], decoder_target_tokens=batch['targets'],
        enable_dropout=False, rngs={'params': rng, 'dropout': rng}, mutable=['intermediates'])
  mask = batch['targets_segmentation'] != 0
  return jnp.sum(output[0] * mask, -1) / jnp.maximum(mask.sum(-1), 1)


def scenarios(stage):
  result = []
  layers = [l for l in range(24) if l % 3 != 2]
  def add(name, selected, dose=1., self_keep=1., cross_keep=1.):
    a = np.ones((24, 5), np.float32)
    a[selected, :3] = (dose, self_keep, cross_keep)
    result.append(dict(name=name, layers=selected, scales=a.tolist()))
  add('native', [])
  if stage == 'dose':
    for group, selected in [('all_L', layers)] + [(f'L{l:02d}', [l]) for l in layers]:
      for dose in (0., .5, 1.5):
        add(f'{group}_dose{dose:g}', selected, dose=dose)
  elif stage == 'route':
    for group, selected in [('all_L', layers)] + [(f'L{l:02d}', [l]) for l in layers]:
      add(f'{group}_self_off', selected, self_keep=0.)
      add(f'{group}_cross_off', selected, cross_keep=0.)
      add(f'{group}_both_off', selected, self_keep=0., cross_keep=0.)
  elif stage == 'qk':
    for group, selected in [('all', list(range(24)))] + [(f'L{l:02d}', [l]) for l in range(24)]:
      for path, columns in [('Q', [3]), ('K', [4]), ('QK', [3,4])]:
        for dose in ((0., .5, 1.5) if group == 'all' else (0.,)):
          a = np.ones((24,5),np.float32)
          a[np.ix_(selected,columns)] = dose
          result.append(dict(name=f'{group}_{path}_dose{dose:g}',layers=selected,scales=a.tolist()))
  elif stage == 'focus':
    for name, v_layers, q_layers, k_layers in (
        ('V_keep_only_L1', [l for l in layers if l != 1], [], []),
        ('V_L1_off', [1], [], []),
        ('V_all_off', layers, [], []),
        ('QK_all_off', [], list(range(24)), list(range(24))),
        ('QKV_all_off', layers, list(range(24)), list(range(24))),
        ('QK_off_V_keep_L1', [l for l in layers if l != 1], list(range(24)), list(range(24))),
    ):
      a = np.ones((24,5),np.float32)
      a[v_layers,0] = 0; a[q_layers,3] = 0; a[k_layers,4] = 0
      result.append(dict(name=name,scales=a.tolist()))
  else:
    raise ValueError(stage)
  return result


def run(config):
  out = Path(os.environ.get('VROW_OUTPUT', '/tmp/xl-localv-row-causal'))
  out.mkdir(parents=True, exist_ok=True)
  cohort_path = Path(os.environ.get('VROW_COHORT', '/tmp/pile_eval_cohort.npz'))
  with np.load(cohort_path) as source:
    cohort = {k: np.asarray(source[k]) for k in KEYS}
    short_hashes = list(source['sequence_hashes'])
  hashes = [{k: hashlib.sha256(cohort[k][i].tobytes()).hexdigest() for k in KEYS}
            for i in range(len(cohort['inputs']))]
  assert len(hashes) == 128 and [h['inputs'][:16] for h in hashes] == short_hashes
  stage = os.environ.get('VROW_STAGE', 'dose')
  start, stop = int(os.environ.get('VROW_START', 0)), int(os.environ.get('VROW_STOP', 32))
  assert 0 <= start < stop <= 128
  cases = scenarios(stage)
  metadata = dict(model=BASE, checkpoint=config.load_parameters_path, checkpoint_step=5250,
      training_commit=TRAIN_COMMIT, diagnostic_commit=subprocess.check_output(
          ['git', 'rev-parse', 'HEAD'], text=True).strip(),
      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
      sequence_hashes=hashes, stage=stage, start=start, stop=stop,
      scope='LocalV shared row consumer and Q/K row outputs; all valid positions; downstream unrestricted',
      route_semantics='self/cross are current-layer AV source==destination/!=destination, not final loss position',
      rounding='Total finite-precision AV difference; self from rounded V difference; cross includes AV rounding')
  (out / f'{stage}_metadata.json').write_text(json.dumps(metadata, indent=2))
  (out / f'{stage}_scenarios.json').write_text(json.dumps(cases, indent=2))
  rng, writer, manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  cursor = SimpleNamespace(meta_dict={'checkpoint_step': None})
  state, _, _, _ = max_utils.setup_training_state(model, cursor, tx, config, rng, mesh, manager)
  ordinary = jax.jit(lambda p,b: forward(model,p,b,rng))
  dose_fn = jax.jit(lambda p,b,s: forward(model,p,b,rng,s,False))
  route_fn = jax.jit(lambda p,b,s: forward(model,p,b,rng,s,True))
  selected_fn = route_fn if stage == 'route' else dose_fn
  first = True
  for i in range(start, stop):
    file = out / f'{stage}_{i:03d}.npz'
    if file.exists():
      with np.load(file) as old:
        assert str(old['sequence_hash']) == hashes[i]['inputs']
      continue
    begun = time.perf_counter()
    batch = {k:jnp.asarray(cohort[k][i:i+1]) for k in KEYS}
    with mesh, partitioning.axis_rules(config.logical_axis_rules):
      baseline = np.asarray(selected_fn(state.params, batch, jnp.ones((24,5),jnp.float32)))
      if first:
        reference = np.asarray(ordinary(state.params, batch))
        np.testing.assert_allclose(baseline, reference, atol=1e-6, rtol=0)
        inactive = np.ones((24,5), np.float32)
        inactive[0] = 0; inactive[2::3, :3] = 0
        np.testing.assert_allclose(np.asarray(selected_fn(state.params,batch,jnp.asarray(inactive))),
                                   baseline,atol=1e-6,rtol=0)
        if stage == 'route':
          for layers in ([1], [10], [l for l in range(24) if l%3!=2]):
            direct = np.ones((24,5),np.float32);direct[layers,0]=0
            routed = np.ones((24,5),np.float32);routed[layers,1:3]=0
            x = np.asarray(dose_fn(state.params,batch,jnp.asarray(direct)))
            y = np.asarray(route_fn(state.params,batch,jnp.asarray(routed)))
            np.testing.assert_allclose(x,y,atol=1e-6,rtol=0)
        print(f'FIRST_STEP NOOP_OK stage={stage} sample={i} loss={baseline}',flush=True)
        first = False
      losses = np.stack([np.asarray(selected_fn(state.params,batch,jnp.asarray(c['scales'],jnp.float32)))
                         for c in cases])
    assert np.isfinite(losses).all()
    pending = out / f'.pending_{stage}_{i:03d}.npz'
    np.savez_compressed(pending, loss=losses, baseline=baseline, gap=losses-baseline,
                        sequence_hash=hashes[i]['inputs'], tokens=int((cohort['targets_segmentation'][i]!=0).sum()))
    pending.replace(file)
    print(f'SAMPLE_DONE stage={stage} index={i} scenarios={len(cases)} seconds={time.perf_counter()-begun:.2f}',flush=True)
  if writer:
    writer.flush()
  print(f'DONE stage={stage} samples={start}:{stop}',flush=True)


if __name__ == '__main__':
  app.run(lambda argv: run(pyconfig.initialize(argv)))
