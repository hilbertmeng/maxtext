"""Historical independent-LLF runtime key similarity; no production edits.

Capture actual post-transform bases and normalized head mixing through Linen
interception. Save per-sequence statistics, never entire activation tensors.
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

BASE = 'BamLlama2MediumV2C256LocalFetchC8LocalVLLFScan'
KEYS = ('inputs', 'targets', 'inputs_position', 'inputs_segmentation', 'targets_segmentation')


class LocalQKVKeyProbe(getattr(exp, BASE)):
  only_eval = True
  enable_checkpointing = True
  per_device_batch_size = 4
  eval_per_device_batch_size = 4
  tensorboard_dir = '/tmp/local-qkv-key-tb'
  load_parameters_path = ('gs://newproject-1-llm_projects_us-east5/log/' + BASE
                          + '/checkpoints/13500/items')


exp.LocalQKVKeyProbe = LocalQKVKeyProbe


def apply_capture(model, params, batch, rng):
  owners = []
  original = attentions.factorized_head_bam_read

  def read(M, x, W_R, W_head_mix, **kw):
    result = original(M, x, W_R, W_head_mix, **kw)
    if not owners:
      return result
    owner = owners[-1]
    module, route, count = owner
    arm = ('q' if count == 0 else 'k') if route == 'qk' else 'v'
    owner[2] += 1
    rank = kw.get('rank', 1)
    assert kw.get('rank_routing', 'legacy') == 'legacy'
    logits = kw.get('key_gate_logits')
    project_kw = {k: v for k, v in kw.items() if k in (
        'rms_epsilon', 'rms_statistics_dtype', 'key_mode', 'key_scale',
        'key_row_norm', 'key_col_norm', 'use_learned_key_norm',
        'key_row_activation', 'key_col_activation')}
    project_kw['key_gate_logits'] = logits[..., None, :] if rank > 1 else logits
    raw_row, raw_col, row, col = attentions._project_bam_read_keys(
        M.shape[-2], x, W_R, **project_kw)
    mix = W_head_mix(x)
    if rank == 1:
      mix = normalizations.rms_norm(mix, dtype=row.dtype,
                                   epsilon=kw['rms_epsilon'], axis=-2)[..., None]
      raw_row, raw_col, row, col = (a[..., None, :] for a in (raw_row, raw_col, row, col))
    else:
      mix = jnp.stack([normalizations.rms_norm(mix[..., s, :], dtype=row.dtype,
          epsilon=kw['rms_epsilon'], axis=(-2, -1)) / jnp.sqrt(jnp.asarray(rank, row.dtype))
          for s in range(2)], axis=-2)
    for side, raw, key, h in (('row', raw_row, row, mix[..., 0, :]),
                              ('col', raw_col, col, mix[..., 1, :])):
      module.sow('intermediates', f'probe_{arm}_{side}_raw', raw.astype(jnp.float32))
      module.sow('intermediates', f'probe_{arm}_{side}_key', key.astype(jnp.float32))
      module.sow('intermediates', f'probe_{arm}_{side}_mix', h.astype(jnp.float32))
    return result

  def intercept(next_fun, args, kwargs, ctx):
    route = {'_read_local_qk': 'qk', '_read_local_v': 'v'}.get(ctx.method_name)
    if route is None:
      return next_fun(*args, **kwargs)
    owners.append([ctx.module, route, 0])
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
  assert len(raw) == (24*2 + 16)*2*3, len(raw)
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
    arms = ['q', 'k'] + (['v'] if l % 3 != 2 else [])
    for side in ('row', 'col'):
      prefix = f'L{l:02d}_{side}'
      for stage in ('raw', 'key'):
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
      bases = {a: unit(raw[f'L{l:02d}_{a}_{side}_key'].astype(np.float64))[0] for a in arms}
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
      effective = {a: np.einsum('btnr,btrd->btnd', raw[f'L{l:02d}_{a}_{side}_mix'],
                               raw[f'L{l:02d}_{a}_{side}_key']) for a in arms}
      for i, a in enumerate(arms):
        for other in arms[i+1:]:
          u, va = unit(effective[a]); v, vb = unit(effective[other])
          cos = np.sum(u*v, -1)
          valid = va & vb & mask[..., None]
          out[f'{prefix}_effective_{a}{other}_cos'] = np.sum(np.where(valid, cos, 0), (1,2)) / np.maximum(valid.sum((1,2)), 1)
          out[f'{prefix}_effective_{a}{other}_cos2'] = np.sum(np.where(valid, cos*cos, 0), (1,2)) / np.maximum(valid.sum((1,2)), 1)
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
  metadata = dict(base_class=BASE, checkpoint=config.load_parameters_path, checkpoint_step=13500,
      training_commit='f6af33c7d1cb313a8db06bb55aabc133b1b450e5',
      diagnostic_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(), sequence_hashes=hashes,
      basis_order_local=['q0','k0','v0','v1'], basis_order_fetch=['q0','k0'],
      scope='runtime key correlation; not causal loss attribution or retraining benefit')
  (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
  for start in range(0,128,4):
    path = output/f'batch_{start:03d}.npz'
    if path.exists():
      continue
    batch = {k: jnp.asarray(cohort[k][start:start+4]) for k in KEYS}
    with mesh, partitioning.axis_rules(config.logical_axis_rules):
      losses, raw = jax.device_get(capture(state.params, batch))
    assert np.isfinite(losses).all()
    assert all(np.isfinite(v).all() for v in raw.values())
    print(f'FIRST_STEP batch={start} mean_loss={losses.mean():.7f}', flush=True)
    results = stats(raw, cohort['targets_segmentation'][start:start+4] != 0)
    pending = output/f'.pending_{start:03d}.npz'
    np.savez_compressed(pending, loss=losses, **results)
    pending.replace(path)
    print(f'BATCH_DONE {start}', flush=True)
  if writer:
    writer.flush()
  print('DONE', flush=True)


if __name__ == '__main__':
  app.run(lambda argv: run(pyconfig.initialize(argv)))
