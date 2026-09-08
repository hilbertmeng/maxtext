"""Read-only shared-LLF O/V gate probe. No production forward modifications.

Capture both gates and ungated side energy with Linen interception; replace gate
parameters in a fresh pytree for paired whole-network loss interventions.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'MaxText'))
from absl import app
from flax import linen as nn
from flax.core import FrozenDict, freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning
import jax
import jax.numpy as jnp
import numpy as np
import exp
import max_utils
import pyconfig
import train

BASE = 'BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan'
COHORT = '/tmp/pile_eval_cohort.npz'
KEYS = ('inputs', 'targets', 'inputs_position', 'inputs_segmentation', 'targets_segmentation')
LAYERS = [l for l in range(24) if l % 3 != 2]


class LocalOVGateProbe(getattr(exp, BASE)):
  only_eval = True
  enable_checkpointing = False
  per_device_batch_size = 4
  eval_per_device_batch_size = 4
  tensorboard_dir = '/tmp/local-ov-gate-tb'
  load_parameters_path = ('gs://newproject-1-llm_projects_us-east5/log/' + BASE
                          + '/checkpoints/13500/items')


exp.LocalOVGateProbe = LocalOVGateProbe


def swap_gate_params(params, layer, side, direction):
  """direction 0: V <- O; 1: O <- V. layer=-1 means all L, -2 identity."""
  flat = dict(flatten_dict(params))
  original = dict(flat)
  found = 0
  for path in original:
    if path[-2:] != ('W_lv_gate', 'kernel'):
      continue
    offset = next(int(re.fullmatch(r'local_(\d+)', p)[1])
                  for p in path if re.fullmatch(r'local_(\d+)', p))
    prefix = path[:-2]
    for suffix_v, suffix_o in ((('W_lv_gate', 'kernel'), ('W_R_gate', 'kernel')),
                               (('W_lv_gate_b0',), ('W_R_gate_b0',))):
      vp, op = prefix + suffix_v, prefix + suffix_o
      v, o_full = original[vp], original[op]
      o = jnp.squeeze(o_full, axis=-2)
      assert v.shape == o.shape, (vp, v.shape, o.shape)
      # param_scan_axis=1, shared by kernels and head/side bias arrays.
      assert v.shape[1] == 8, (vp, v.shape)
      block_mask = ((jnp.arange(8) * 3 + offset == layer) | (layer == -1))
      shape = [1] * v.ndim
      shape[1] = 8
      mask = block_mask.reshape(shape) & (jnp.arange(2) == side)
      flat[vp] = jnp.where(mask & (direction == 0), o, v)
      flat[op] = jnp.expand_dims(jnp.where(mask & (direction == 1), v, o), -2)
    found += 1
  assert found == 2, f'expected local_0/local_1 gate groups; found {found}'
  result = unflatten_dict(flat)
  return freeze(result) if isinstance(params, FrozenDict) else result


def apply_model(model, params, batch, rng, capture=False):
  remembered = {}

  def interceptor(next_fun, args, kwargs, ctx):
    result = next_fun(*args, **kwargs)
    module = ctx.module
    if not getattr(module, '_local_o', False):
      return result
    path = tuple(module.scope.path)
    if ctx.method_name == '_read_fetched_m' and kwargs.get('ungated', False):
      read, logits_o = result
      remembered[path] = (read, logits_o)
    elif ctx.method_name == '_gate_local_output' and path in remembered:
      read, logits_o = remembered.pop(path)
      logits_v = args[1]
      # First call is V, second O; shape [B,T,N,side,destination].
      logits = jnp.stack((logits_o, logits_v), -1).astype(jnp.float32)
      energy = jnp.stack((
          jnp.sum(jnp.square(read[..., module.bam_k:module.bam_k + module._abs_v_dim].astype(jnp.float32)), -1),
          jnp.sum(jnp.square(read[..., :module.bam_k].astype(jnp.float32)), -1)), -1)
      # Keep scalar gates/energy, not read vectors or M. Full-token statistics can
      # subsequently be recomputed; no sampling of individual source positions.
      module.sow('intermediates', 'ov_logits', logits)
      module.sow('intermediates', 'ov_energy', energy)
    return result

  kw = dict(decoder_segment_ids=batch['inputs_segmentation'],
            decoder_target_mask=batch['targets_segmentation'],
            decoder_target_tokens=batch['targets'], enable_dropout=False,
            rngs={'params': rng, 'dropout': rng})
  if capture:
    with nn.intercept_methods(interceptor):
      output, collections = model.apply(params, batch['inputs'], batch['inputs_position'],
                                        mutable=['intermediates'], **kw)
  else:
    output = model.apply(params, batch['inputs'], batch['inputs_position'], **kw)
  loss = output[0]
  mask = batch['targets_segmentation'] != 0
  per_sequence = jnp.sum(loss * mask, -1) / jnp.maximum(jnp.sum(mask, -1), 1)
  if not capture:
    return per_sequence
  raw = {}
  for path, value in flatten_dict(collections['intermediates']).items():
    if path[-1] not in ('ov_logits', 'ov_energy'):
      continue
    offset = next(int(re.fullmatch(r'local_(\d+)', p)[1])
                  for p in path if re.fullmatch(r'local_(\d+)', p))
    while isinstance(value, (tuple, list)) and len(value) == 1:
      value = value[0]
    assert value.shape[0] == 8, (path, value.shape)
    for block in range(8):
      raw[f'L{3 * block + offset:02d}_{path[-1]}'] = value[block]
  assert len(raw) == 32, list(raw)
  return per_sequence, raw


def parameter_spectra(params):
  flat = flatten_dict(params)
  result = {}
  for p, value in flat.items():
    if p[-2:] != ('W_lv_gate', 'kernel'):
      continue
    offset = next(int(re.fullmatch(r'local_(\d+)', x)[1])
                  for x in p if re.fullmatch(r'local_(\d+)', x))
    v = np.asarray(value, np.float64)
    o = np.asarray(flat[p[:-2] + ('W_R_gate', 'kernel')], np.float64).squeeze(-2)
    for block in range(8):
      joint = np.concatenate((o[:, block].reshape(o.shape[0], -1),
                              v[:, block].reshape(v.shape[0], -1)), axis=-1)
      s = np.linalg.svd(joint, compute_uv=False)
      result[str(block * 3 + offset)] = (s * s / max(np.sum(s * s), 1e-30)).tolist()
  return result


def run(config):
  started = time.time()
  cohort_path = Path(os.environ.get('OV_COHORT', COHORT))
  output = Path(os.environ.get('OV_OUTPUT', '/tmp/local-ov-gate'))
  output.mkdir(parents=True, exist_ok=True)
  with np.load(cohort_path) as data:
    cohort = {k: np.asarray(data[k]) for k in (*KEYS, 'sequence_hashes')}
  n = int(os.environ.get('OV_SEQUENCES', '128'))
  bs = int(os.environ.get('OV_BATCH', '4'))
  assert n <= len(cohort['inputs']) and n % bs == 0
  hashes = [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in cohort['inputs'][:n]]
  assert hashes == list(cohort['sequence_hashes'][:n]), 'cohort hashes differ'
  rng, writer, manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  state, _, _, _ = max_utils.setup_training_state(model, None, tx, config, rng, mesh, manager)
  params = state.params
  flat_shapes = {'/'.join(p): list(v.shape) for p, v in flatten_dict(params).items()}
  (output / 'parameter_shapes.json').write_text(json.dumps(flat_shapes, indent=2))
  capture = jax.jit(lambda p, b: apply_model(model, p, b, rng, True))
  forward = jax.jit(lambda p, b, l, s, d:
                    apply_model(model, swap_gate_params(p, l, s, d), b, rng))
  arms = [(l, s, d) for l in [-1, *LAYERS] for s in range(2) for d in range(2)]
  metadata = dict(base_class=BASE, diagnostic_class='LocalOVGateProbe',
      training_commit='f6af33c7d1cb313a8db06bb55aabc133b1b450e5',
      diagnostic_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
      checkpoint=config.load_parameters_path, checkpoint_step=13500,
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
      sequence_hashes=hashes, arms=arms, side_order=['row/address', 'col/data'],
      destination_order=['O', 'V'], parameter_spectra=parameter_spectra(params),
      devices=[str(x) for x in jax.devices()], batch_size=bs)
  (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
  for start in range(0, n, bs):
    path = output / f'batch_{start:03d}.npz'
    if path.exists():
      continue
    batch = {k: jnp.asarray(cohort[k][start:start + bs]) for k in KEYS}
    with mesh, partitioning.axis_rules(config.logical_axis_rules):
      baseline, raw = capture(params, batch)
      baseline, raw = jax.device_get((baseline, raw))
      identity = np.asarray(forward(params, batch, -2, 0, 0))
      np.testing.assert_allclose(identity, baseline, atol=2e-5, rtol=0)
      print(f'FIRST_STEP batch={start} loss={baseline.mean():.7f} identity_max={abs(identity-baseline).max():.3g}', flush=True)
      losses = []
      for index, (layer, side, direction) in enumerate(arms):
        losses.append(np.asarray(forward(params, batch, layer, side, direction)))
        if index % 16 == 0:
          print(f'PROGRESS batch={start} arm={index}/{len(arms)}', flush=True)
    # bf16 training logits -> fp32 captures; lossless compression, all tokens retained.
    np.savez_compressed(path, baseline=baseline, losses=np.stack(losses),
                        mask=np.asarray(batch['targets_segmentation'] != 0), **raw)
    print(f'BATCH_DONE start={start} elapsed={time.time()-started:.1f}', flush=True)
  print(f'DONE output={output} elapsed={time.time()-started:.1f}', flush=True)
  if writer:
    writer.flush()


if __name__ == '__main__':
  app.run(lambda argv: run(pyconfig.initialize(argv)))
