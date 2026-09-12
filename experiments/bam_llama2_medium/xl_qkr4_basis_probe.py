"""XL QK rank-4 raw basis geometry; no production edits.

Capture projection plus bias, before normalization/gates/head mixing. Analyze
every valid token and retain per-sequence statistics, not activation tensors.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'MaxText'))
from absl import app
from flax import linen as nn
from flax.linen import partitioning
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import exp
import max_utils
import pyconfig
import train
from layers import attentions, normalizations

BASE = 'BamXLIndependentLLFLocalQKRank4CFp32AlignedRow'
KEYS = ('inputs', 'targets', 'inputs_position', 'inputs_segmentation', 'targets_segmentation')
DECOMPOSE = os.environ.get('QKV_DECOMPOSE_BIAS', '0') == '1'


class LocalQKVKeyProbe(getattr(exp, BASE)):
  only_eval = True
  enable_checkpointing = True
  per_device_batch_size = 1
  eval_per_device_batch_size = 1
  tensorboard_dir = '/tmp/local-qkv-key-tb'
  load_parameters_path = ('gs://newproject-1-llm_projects_europe-west4/log/diagnostics/'
                          'xl-qkr4-basis-16000/checkpoints/16000/items')


exp.LocalQKVKeyProbe = LocalQKVKeyProbe


def apply_capture(model, params, batch, rng):
  owners = []
  original = attentions.factorized_head_bam_read

  def read(M, x, W_R, W_head_mix, **kw):
    result = original(M, x, W_R, W_head_mix, **kw)
    if not owners:
      return result
    module, arm, dynamic = owners[-1]
    if arm not in ('q', 'k'):
      return result
    rank = kw.get('rank', 1)
    assert rank == 4 and kw['rank_routing'] == 'effective_key'
    # W_R is the production closure after pre-RMS bias, before any transforms.
    key = W_R(x)
    assert key.shape[-2:] == (rank, M.shape[-2] + M.shape[-1]), key.shape
    row, col = jnp.split(key, [M.shape[-2]], axis=-1)
    for side, basis in (('row', row), ('col', col)):
      module.sow('intermediates', f'probe_{arm}_{side}_raw', basis.astype(jnp.float32))
    if DECOMPOSE:
      spec = module._local_arms[arm]
      bias = jnp.asarray(getattr(module, f'{spec.prefix}_bias'), dynamic.dtype)
      if not spec.pre_rms_bias:
        bias = jnp.zeros_like(bias)
      for stage, value in (('dynamic', dynamic), ('bias', jnp.broadcast_to(bias, dynamic.shape))):
        sides = jnp.split(value, [M.shape[-2]], axis=-1)
        for side, basis in zip(('row', 'col'), sides):
          module.sow('intermediates', f'probe_{arm}_{side}_{stage}', basis.astype(jnp.float32))
    return result

  def intercept(next_fun, args, kwargs, ctx):
    if ctx.method_name != '_read_local':
      return next_fun(*args, **kwargs)
    arm = args[1] if isinstance(args[0], nn.Module) else args[0]
    local_inputs = kwargs.get('local_inputs', args[-1])
    owners.append([ctx.module, arm, local_inputs[arm][0]])
    try:
      return next_fun(*args, **kwargs)
    finally:
      owners.pop()

  attentions.factorized_head_bam_read = read
  try:
    with nn.intercept_methods(intercept):
      output, collections = model.apply(params, batch['inputs'], batch['inputs_position'],
          decoder_segment_ids=batch['inputs_segmentation'],
          decoder_target_mask=batch['targets_segmentation'], decoder_target_tokens=batch['targets'],
          enable_dropout=False, rngs={'params': rng, 'dropout': rng}, mutable=['intermediates'])
  finally:
    attentions.factorized_head_bam_read = original
  raw = {}
  for path, value in flatten_dict(collections['intermediates']).items():
    if not path[-1].startswith('probe_'):
      continue
    offset = next(int(m[1]) for p in path if (m := re.fullmatch(r'(?:local|fetch)_(\d+)', p)))
    while isinstance(value, (tuple, list)) and len(value) == 1:
      value = value[0]
    assert value.shape[0] == 8, (path, value.shape)
    for block in range(8):
      raw[f'L{3*block+offset:02d}_{path[-1][6:]}'] = value[block]
  assert len(raw) == 24*2*2*(3 if DECOMPOSE else 1), (len(raw), list(raw))
  mask = batch['targets_segmentation'] != 0
  loss = jnp.sum(output[0] * mask, -1) / jnp.maximum(mask.sum(-1), 1)
  return loss, raw


def unit(x):
  norm = np.linalg.norm(x, axis=-1, keepdims=True)
  return np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-12), norm[..., 0] > 1e-12


def stats(raw, mask):
  """All valid tokens; dimensions remain native common M coordinates per side."""
  out = {}
  for l in range(24):
    arms = ['q', 'k']
    for side in ('row', 'col'):
      prefix = f'L{l:02d}_{side}'
      for arm in arms:
        energy = []
        for b in range(len(mask)):
          x = raw[f'L{l:02d}_{arm}_{side}_raw'][b, mask[b]].astype(np.float64)
          power = np.linalg.svd(x, compute_uv=False)**2
          total = power.sum(-1, keepdims=True)
          energy.append(np.mean(np.cumsum(power, -1) / np.maximum(total, 1e-30), 0))
        out[f'{prefix}_{arm}_rank_energy'] = np.stack(energy)
      for stage in ('raw',):
        vectors = np.concatenate([raw[f'L{l:02d}_{a}_{side}_{stage}'] for a in arms], -2)
        # Local rank is not a globally fixed cross-token projection.
        for centered in (False, True):
          records = []
          for b in range(len(mask)):
            x = vectors[b, mask[b]].astype(np.float64)
            if centered:
              x = x - x.mean(0, keepdims=True)
            u, valid = unit(x)
            cos = u @ u.swapaxes(-1, -2)
            valid_pair = valid[..., :, None] & valid[..., None, :]
            cos = np.where(valid_pair, cos, np.nan)
            singular = np.linalg.svd(u, compute_uv=False)**2
            retained = np.cumsum(singular, -1) / np.maximum(singular.sum(-1, keepdims=True), 1e-30)
            actual_singular = np.linalg.svd(x, compute_uv=False)**2
            actual_retained = np.cumsum(actual_singular, -1) / np.maximum(actual_singular.sum(-1, keepdims=True), 1e-30)
            records.append((np.nanmean(cos, 0), np.nanmean(abs(cos), 0),
                            np.nanmean(cos**2, 0), np.mean(valid_pair, 0), retained.mean(0),
                            np.nanquantile(cos, [.05,.25,.5,.75,.95], axis=0),
                            np.nanmean(np.where(valid_pair, cos < -.8, np.nan), 0),
                            np.nanmean(np.where(valid_pair, cos > .8, np.nan), 0),
                            actual_retained.mean(0), np.mean(np.linalg.norm(x, axis=-1), 0)))
          for index, name in enumerate(('cos', 'abs_cos', 'cos2', 'valid_fraction', 'rank_energy',
                                        'cos_quantiles', 'cos_lt_neg08', 'cos_gt_pos08',
                                        'amplitude_weighted_rank_energy', 'basis_norm')):
            out[f'{prefix}_{stage}_{"centered" if centered else "uncentered"}_{name}'] = np.stack([r[index] for r in records])
      # Direction recovery in another arm's per-token span. Pseudoinverse handles
      # repeated/zero bases; orthogonal coefficients need not equal trained mix.
      bases = {a: unit(raw[f'L{l:02d}_{a}_{side}_raw'].astype(np.float64))[0] for a in arms}
      for a in arms:
        for other in arms:
          if a == other:
            continue
          energy = []
          for b in range(len(mask)):
            source, target = bases[a][b, mask[b]], bases[other][b, mask[b]]
            gram = target @ target.swapaxes(-1,-2)
            cross = source @ target.swapaxes(-1,-2)
            recovered = np.sum((cross @ np.linalg.pinv(gram, rcond=1e-6)) * cross, -1)
            energy.append(recovered.mean(0))
          out[f'{prefix}_{a}_in_{other}_span'] = np.stack(energy)
  return out


def run(config):
  cohort_path = Path(os.environ.get('QKV_COHORT', '/tmp/pile_eval_cohort.npz'))
  output = Path(os.environ.get('QKV_OUTPUT', '/tmp/local-qkv-keys'))
  output.mkdir(parents=True, exist_ok=True)
  with np.load(cohort_path) as data:
    cohort = {k: np.asarray(data[k]) for k in (*KEYS, 'sequence_hashes')}
  hashes = [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in cohort['inputs']]
  assert hashes == list(cohort['sequence_hashes']) and len(hashes) == 128
  rng, writer, manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  cursor = SimpleNamespace(meta_dict={'checkpoint_step': None})
  state, _, _, _ = max_utils.setup_training_state(model, cursor, tx, config, rng, mesh, manager)
  capture = jax.jit(lambda p, b: apply_capture(model, p, b, rng))
  def plain(p, b):
    output = model.apply(p, b['inputs'], b['inputs_position'],
        decoder_segment_ids=b['inputs_segmentation'],
        decoder_target_mask=b['targets_segmentation'], decoder_target_tokens=b['targets'],
        enable_dropout=False, rngs={'params': rng, 'dropout': rng})
    mask = b['targets_segmentation'] != 0
    return (output[0] * mask).sum(-1) / jnp.maximum(mask.sum(-1), 1)
  plain = jax.jit(plain)
  metadata = dict(base_class=BASE, checkpoint=config.load_parameters_path, checkpoint_step=16000,
      decompose_bias=DECOMPOSE,
      training_commit='b264b490c1081875ef439f5e59ae57f64d809c6c',
      diagnostic_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(), sequence_hashes=hashes,
      basis_order_local=['q0','q1','q2','q3','k0','k1','k2','k3'],
      basis_order_fetch=['q0','q1','q2','q3','k0','k1','k2','k3'],
      scope='runtime key correlation; not causal loss attribution or retraining benefit')
  (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
  for start in range(128):
    path = output/f'batch_{start:03d}.npz'
    if path.exists():
      continue
    batch = {k: jnp.asarray(cohort[k][start:start+1]) for k in KEYS}
    with mesh, partitioning.axis_rules(config.logical_axis_rules):
      losses, raw = jax.device_get(capture(state.params, batch))
    assert np.isfinite(losses).all()
    assert all(np.isfinite(v).all() for v in raw.values())
    if start == 0:
      with mesh, partitioning.axis_rules(config.logical_axis_rules):
        reference = np.asarray(plain(state.params, batch))
      np.testing.assert_allclose(losses, reference, rtol=0, atol=1e-5)
      print(f'CAPTURE_VALIDATED max_loss_error={abs(losses-reference).max()}', flush=True)
    print(f'FIRST_STEP batch={start} mean_loss={losses.mean():.7f}', flush=True)
    results = stats(raw, cohort['targets_segmentation'][start:start+1] != 0)
    if DECOMPOSE:
      mask = cohort['targets_segmentation'][start:start+1] != 0
      for stage in ('dynamic', 'bias'):
        stage_raw = {k[:-len(stage)]+'raw': v for k, v in raw.items() if k.endswith('_'+stage)}
        results.update({stage+'__'+k: v for k,v in stats(stage_raw, mask).items()})
      for layer in range(24):
        for arm in ('q', 'k'):
          for side in ('row', 'col'):
            prefix = f'L{layer:02d}_{arm}_{side}'
            dynamic = raw[prefix+'_dynamic'].astype(np.float64)
            bias = raw[prefix+'_bias'].astype(np.float64)
            total = raw[prefix+'_raw'].astype(np.float64)
            dn, bn, tn = (np.linalg.norm(v, axis=-1) for v in (dynamic,bias,total))
            cosine = np.sum(dynamic*bias,-1)/np.maximum(dn*bn,1e-30)
            cosine = np.where(dn*bn>1e-20,cosine,np.nan)
            for name,value in (('dynamic_norm',dn),('bias_norm',bn),('total_norm',tn),
                               ('dynamic_bias_cos',cosine),('bias_over_total',bn/np.maximum(tn,1e-30))):
              results[prefix+'_'+name] = np.stack([np.nanmean(value[b,mask[b]],axis=0) for b in range(len(mask))])
    pending = output/f'.pending_{start:03d}.npz'
    np.savez_compressed(pending, loss=losses, **results)
    pending.replace(path)
    print(f'BATCH_DONE {start}', flush=True)
  if writer:
    writer.flush()
  print('DONE', flush=True)


if __name__ == '__main__':
  app.run(lambda argv: run(pyconfig.initialize(argv)))
