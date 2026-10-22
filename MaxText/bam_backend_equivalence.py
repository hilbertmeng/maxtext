"""Read-only, same-checkpoint/same-cohort JAX versus Pallas loss comparison."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

from absl import app
from flax.linen import partitioning as nn_partitioning
import jax
import numpy as np

from bam_wr_ablation import _delta_summary, _forward
from input_pipeline.input_pipeline_interface import create_data_iterator
import max_utils
import pyconfig
import train


def latest_checkpoint(root):
  result = subprocess.run(['gsutil', 'ls', root.rstrip('/') + '/*/commit_success.txt'],
                          check=True, capture_output=True, text=True)
  steps = [int(m[1]) for line in result.stdout.splitlines()
           if (m := re.search(r'/(\d+)/commit_success\.txt$', line))]
  if not steps:
    raise ValueError('No committed source checkpoint')
  step = max(steps)
  return root.rstrip('/') + f'/{step}/items'


def config_with(config, **values):
  return pyconfig.HyperParameters(SimpleNamespace(keys=dict(config.get_keys(), **values)))


def run(config):
  if not config.only_eval or config.enable_checkpointing or config.bam_diagnostics:
    raise ValueError('Requires only_eval=True, enable_checkpointing=False, bam_diagnostics=False')
  if not config.bam_pallas_core or config.bam_pallas_body != 'v8t':
    raise ValueError('Use the sealed v8t experiment configuration')
  root = os.environ.get('BAM_BACKEND_CHECKPOINT_ROOT')
  if not root and not config.load_parameters_path:
    raise ValueError('A source checkpoint is required; random weights are not a comparison')
  if root:
    config = config_with(config, load_parameters_path=latest_checkpoint(root))
  started = time.perf_counter()
  init_rng, writer, manager, mesh, model, _, tx = train.setup_mesh_and_model(config)
  cohort = Path(os.environ.get('BAM_BACKEND_COHORT', '/tmp/bam_backend_cohort.npz'))
  train_iterator, eval_iterator = (None, None) if cohort.exists() else create_data_iterator(config, mesh)
  # A running trainer may prune its older periodic checkpoint during setup.
  for attempt in range(2):
    if root:
      config = config_with(config, load_parameters_path=latest_checkpoint(root))
    try:
      state, _, _, _ = max_utils.setup_training_state(
          model, train_iterator, tx, config, init_rng, mesh, manager)
      jax.block_until_ready(state.params)
      break
    except Exception:
      if not root or attempt:
        raise
      print('SOURCE_RESTORE_RETRY latest committed checkpoint', flush=True)
  print('SOURCE_RESTORED ' + config.load_parameters_path, flush=True)
  count = int(os.environ.get('BAM_BACKEND_SEQUENCES', '32'))
  if cohort.exists():
    with np.load(cohort, allow_pickle=False) as data:
      batches = {k: np.array(data[k]) for k in data.files}
    if len(next(iter(batches.values()))) != count:
      raise ValueError('Existing cohort has the wrong sequence count')
  else:
    if eval_iterator is None:
      raise ValueError('Enable a shuffled Pile eval iterator')
    collected = [jax.device_get(next(eval_iterator)) for _ in range(count)]
    batches = {k: np.stack([np.asarray(b[k]) for b in collected]) for k in collected[0]}
    if batches['inputs'].shape[1] != 1:
      raise ValueError('This numerical probe uses one sequence per eval batch')
    cohort.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cohort, **batches)
  hashes = [hashlib.sha256(batches['inputs'][i].tobytes()).hexdigest() for i in range(count)]
  reference_cfg = config_with(config, bam_pallas_core=False)
  models = {'jax': model.clone(config=reference_cfg), 'pallas': model}
  forwards = {name: jax.jit(lambda p, b, r, m=m: _forward(m, p, b, r))
              for name, m in models.items()}
  results = {name: [] for name in models}
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    for index in range(count):
      batch = {k: jax.device_put(v[index]) for k, v in batches.items()}
      rng = jax.random.fold_in(init_rng, 0)
      for name in ('jax', 'pallas'):
        metrics = jax.device_get(jax.block_until_ready(forwards[name](state.params, batch, rng)))
        loss = float(metrics['loss'])
        if not np.isfinite(loss):
          raise ValueError(f'Nonfinite {name} loss at sequence {index}')
        results[name].append({'loss': loss, 'sequence_weights':
                             int(np.asarray(metrics['sequence_weights']).sum())})
      print(f"PAIRED_SEQUENCE index={index} jax={results['jax'][-1]['loss']:.9f} "
            f"pallas={results['pallas'][-1]['loss']:.9f}", flush=True)
  weights = np.asarray([r['sequence_weights'] for r in results['jax']])
  if not np.array_equal(weights, [r['sequence_weights'] for r in results['pallas']]):
    raise ValueError('Backend masks disagree')
  losses = {name: np.asarray([r['loss'] for r in values]) for name, values in results.items()}
  delta = losses['pallas'] - losses['jax']
  weighted = {name: float(np.average(values, weights=weights)) for name, values in losses.items()}
  report = {'experiment': config.exp_class, 'checkpoint': config.load_parameters_path,
      'diagnostic_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
      'cohort_seed': config.data_shuffle_seed, 'eval_shuffle_buffer_size': config.eval_shuffle_buffer_size,
      'dataset_path': config.dataset_path,
      'sequence_length': config.max_target_length, 'sequences': count,
      'cohort_sha256': hashlib.sha256(cohort.read_bytes()).hexdigest(),
      'sequence_hashes': hashes, 'valid_tokens': int(weights.sum()),
      'loss': weighted, 'loss_delta_pallas_minus_jax': weighted['pallas'] - weighted['jax'],
      'sequence_deltas': delta.tolist(), 'sequence_delta_summary': _delta_summary(delta),
      'max_abs_sequence_delta': float(np.max(np.abs(delta))), 'per_sequence': results,
      'dtype': str(config.dtype), 'pallas_body': config.bam_pallas_body,
      'checkpoint_mutated': False, 'seconds': time.perf_counter() - started}
  output = Path(os.environ.get('BAM_BACKEND_OUTPUT', '/tmp/bam_backend_equivalence.json'))
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2) + '\n')
  print('BAM_BACKEND_EQUIVALENCE_DONE ' + json.dumps({k: report[k] for k in
      ('experiment', 'checkpoint', 'loss', 'loss_delta_pallas_minus_jax', 'max_abs_sequence_delta')}), flush=True)
  if writer is not None:
    writer.flush()


def main(argv):
  config = pyconfig.initialize(argv)
  train.validate_train_config(config)
  os.environ['TFDS_DATA_DIR'] = config.dataset_path
  run(config)


if __name__ == '__main__':
  app.run(main)
