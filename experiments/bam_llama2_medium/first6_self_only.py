"""Full-network causal removal of early-layer cross fetch; no parameter changes.

Uses the fixed residual-attribution cohort and the same model/restore machinery.
Every RMS denominator and every downstream attention/read/write is recomputed.
The diagnostic collection is sliced by layer scan, never broadcast across layers.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from absl import app
from flax import traverse_util
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_attribution as base


class BamFirst6SelfOnly(base.BamResidualAttribution):
  bam_residual_attribution = False
  bam_readout_attribution = False


base.exp.BamFirst6SelfOnly = BamFirst6SelfOnly


def ablation_collection(params, scales, scan_layers):
  flat = traverse_util.flatten_dict(params['params'])
  values = {}
  for path in flat:
    if path[-1] != 'abs_v_cache_projection':
      continue
    if scan_layers:
      value = scales
    else:
      layer = next(int(m.group(1)) for p in path
                   if (m := re.fullmatch(r'layers_(\d+)', p)))
      value = scales[layer]
    values[path[:-1] + ('cross_scale',)] = value
  expected = 1 if scan_layers else len(scales)
  if len(values) != expected:
    raise ValueError(f'expected {expected} layer scopes, found {list(values)}')
  return traverse_util.unflatten_dict(values)


def forward(model, params, batch, rng, scales, scan_layers):
  variables = dict(params)
  if scales is not None:
    variables['causal_ablation'] = ablation_collection(params, scales, scan_layers)
  rng1, aqt_rng = jax.random.split(rng)
  xent, _, _ = model.apply(
      variables, batch['inputs'], batch['inputs_position'],
      decoder_segment_ids=batch['inputs_segmentation'],
      decoder_target_mask=batch['targets_segmentation'],
      decoder_target_tokens=batch['targets'], enable_dropout=False,
      rngs={'dropout': rng1, 'params': aqt_rng})
  valid = batch['targets_segmentation'] != 0
  return base._sequence_mean(xent, valid), xent


def run(config):
  output = Path(os.environ['BAM_SELF_ONLY_OUTPUT'])
  output.mkdir(parents=True, exist_ok=True)
  cohort_path = Path(os.environ['BAM_RESIDUAL_ATTR_COHORT_PATH'])
  with np.load(cohort_path) as data:
    cohort = {k: np.asarray(data[k]) for k in (
        'inputs', 'targets', 'inputs_position', 'inputs_segmentation',
        'targets_segmentation', 'sequence_hashes')}
  batch_size = int(os.environ.get('BAM_RESIDUAL_ATTR_BATCH_SIZE', '1'))
  n = len(cohort['inputs'])
  if n % batch_size:
    raise ValueError('cohort must divide into complete batches')
  start = time.perf_counter()
  rng, writer, manager, mesh, model, _, tx = base.train.setup_mesh_and_model(config)
  iterator, _ = base.create_data_iterator(config, mesh)
  state, _, _, _ = base.max_utils.setup_training_state(
      model, iterator, tx, config, rng, mesh, manager)
  compiled = jax.jit(lambda p, b, r, s: forward(model, p, b, r, s, config.scan_layers))
  original = jax.jit(lambda p, b, r: forward(model, p, b, r, None, config.scan_layers))
  lambdas = [1.0, 0.5, 0.0]
  scales = [jnp.asarray([v] * 6 + [1.] * (config.num_decoder_layers - 6), jnp.float32)
            for v in lambdas]
  metadata = dict(
      base_config_class=base._BASE_CONFIG_CLASS,
      checkpoint=config.load_parameters_path, diagnostic_commit=os.environ['DIAGNOSTIC_COMMIT'],
      trainer_commit=base._TRAINER_COMMIT, scan_layers=config.scan_layers,
      layers=list(range(6)), lambdas=lambdas, batch_size=batch_size, sequences=n,
      cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
      semantics='scale only non-diagonal mixed alpha; recompute the entire network; normal RMS',
      setup_seconds=time.perf_counter()-start)
  print('RESTORED ' + json.dumps(metadata), flush=True)
  all_losses = []
  for offset in range(0, n, batch_size):
    batch = {k:jnp.asarray(v[offset:offset+batch_size]) for k,v in cohort.items()
             if k != 'sequence_hashes'}
    losses, tokens = [], []
    t0 = time.perf_counter()
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      for scale in scales:
        l, tok = jax.device_get(compiled(state.params, batch, rng, scale))
        losses.append(l)
        tokens.append(tok)
      if offset == 0:
        _, reference_tokens = jax.device_get(original(state.params, batch, rng))
        error = float(np.max(np.abs(tokens[0] - reference_tokens)))
        metadata['lambda1_original_token_max_error'] = error
        if error > 1e-6:
          raise ValueError(f'lambda=1 does not reproduce unmodified forward: {error}')
    losses = np.stack(losses, axis=1)
    if not np.isfinite(losses).all():
      raise ValueError('non-finite loss')
    np.savez_compressed(output / f'batch_{offset:03d}.npz',
                        loss=losses, token_loss=np.stack(tokens, axis=1),
                        valid=np.asarray(batch['targets_segmentation']) != 0,
                        sequence_hashes=cohort['sequence_hashes'][offset:offset+batch_size])
    all_losses.append(losses)
    print(f'BATCH {offset+batch_size}/{n} seconds={time.perf_counter()-t0:.2f} '
          f'loss={losses.mean(0).tolist()} delta={(losses-losses[:,:1]).mean(0).tolist()}', flush=True)
    interim = np.concatenate(all_losses)
    metadata['completed_sequences'] = len(interim)
    metadata['results'] = []
    for i,value in enumerate(lambdas):
      delta = interim[:,i]-interim[:,0]
      metadata['results'].append(dict(cross_scale=value, loss=float(interim[:,i].mean()),
          delta_mean=float(delta.mean()), delta_ci95=float(1.96*delta.std(ddof=1)/np.sqrt(len(delta)))
          if len(delta)>1 else None, delta_median=float(np.median(delta)),
          delta_p95=float(np.quantile(delta,.95)), delta_max=float(delta.max()),
          fraction_harmed=float((delta>0).mean())))
    metadata['elapsed_seconds'] = time.perf_counter()-start
    (output/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
  if writer:
    writer.flush()
  print('SELF_ONLY_COMPLETE '+json.dumps(metadata), flush=True)


def main(argv):
  config = base.pyconfig.initialize(argv)
  base.train.validate_train_config(config)
  run(config)


if __name__ == '__main__':
  app.run(main)
