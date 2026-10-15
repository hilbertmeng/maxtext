"""Benchmark real Pile batches after a read-only checkpoint restore."""
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from pile_bucket_eval import prepare_data, make_batch, forward, restore_parameters
from pile_validation_data import atomic_json


def benchmark(config):
  import jax
  from jax.sharding import Mesh
  from flax.linen import partitioning
  import max_utils
  import pyconfig
  from layers.models import Transformer
  from layers import quantizations

  if not config.only_eval or not config.load_parameters_path or config.load_full_state_path:
    raise ValueError('Require explicit read-only parameter restore')
  if jax.default_backend() != 'tpu' or jax.process_count() != 1:
    raise ValueError('Require one TPU host')
  output = Path(os.environ['PILE_BUCKET_OUTPUT'])
  output.mkdir(parents=True, exist_ok=True)
  raw = dict(config.get_keys())
  for key in raw:
    if key.startswith('bam_record_') or key in (
        'record_internal_nn_metrics', 'record_training_health_metrics', 'bam_diagnostics'):
      raw[key] = False
  config = pyconfig.HyperParameters(SimpleNamespace(keys=raw))
  meta = prepare_data(os.environ['PILE_BUCKET_SOURCE'], output/'data', config.max_target_length)
  tokens = np.load(output/'data/tokens.npy', mmap_mode='r')
  lengths = np.load(output/'data/lengths.npy', mmap_mode='r')
  rng = jax.random.PRNGKey(config.init_weights_seed)
  mesh = Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
  model = Transformer(config, mesh, quant=quantizations.configure_quantization(config))
  state = restore_parameters(model, config, rng, mesh)
  batches = [int(x) for x in os.environ.get('PILE_BUCKET_BENCH_BATCHES', '16,32,64,128').split(',')]
  repetitions = int(os.environ.get('PILE_BUCKET_BENCH_REPETITIONS', '8'))
  result = {
      'benchmark_only': True, 'complete': False, 'exp_class': config.exp_class,
      'checkpoint': config.load_parameters_path,
      'runtime_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'source_sha256': meta['source_integrity']['sha256'],
      'sequence_length': config.max_target_length,
      'parameter_count': int(max_utils.calculate_num_params_from_pytree(state.params)),
      'repetitions': repetitions, 'rows': [],
  }
  reference = None
  for size in batches:
    compiled = jax.jit(lambda params,batch,key: forward(model,params,batch,key,256))
    try:
      batch, count = make_batch(tokens,lengths,0,size,config.max_target_length)
      if count != size:
        raise ValueError('Not enough full real sequences for benchmark')
      compile_start = time.monotonic()
      with mesh, partitioning.axis_rules(config.logical_axis_rules):
        measured = jax.device_get(jax.block_until_ready(compiled(state.params,batch,rng)))
      compile_seconds = time.monotonic()-compile_start
      if int(measured['nonfinite_tokens']):
        raise FloatingPointError('Nonfinite real token loss')
      prefix = np.asarray(measured['sum_excluding_eos'])[:batches[0]]
      if reference is None:
        reference = prefix.copy()
      prefix_delta = float(np.max(np.abs(prefix-reference))/256)
      # Report numerical agreement rather than silently assuming batch invariance.
      if prefix_delta > 0.002:
        raise ValueError(f'Batch shape changed loss materially: {prefix_delta}')
      seconds = []
      for _ in range(repetitions+2):
        t0 = time.monotonic()
        with mesh, partitioning.axis_rules(config.logical_axis_rules):
          measured = jax.device_get(jax.block_until_ready(compiled(state.params,batch,rng)))
        seconds.append(time.monotonic()-t0)
      median = float(np.median(seconds[2:]))
      row = {'batch_size': size, 'median_batch_seconds': median,
             'target_tokens_per_second': size*config.max_target_length/median,
             'compile_and_first_batch_seconds': compile_seconds,
             'prefix_max_bucket_loss_delta': prefix_delta,
             'device_memory_stats': jax.devices()[0].memory_stats()}
      result['rows'].append(row)
      print('PILE_BATCH_BENCH '+json.dumps(row),flush=True)
    except Exception as error:
      result['rows'].append({'batch_size':size,'error':repr(error)})
      atomic_json(output/'benchmark.json',result)
      if 'RESOURCE_EXHAUSTED' not in str(error) and 'out of memory' not in str(error).lower():
        raise
      print('PILE_BATCH_BENCH_OOM '+str(size),flush=True)
      break
    finally:
      compiled.clear_cache()
      del compiled
      gc.collect()
    atomic_json(output/'benchmark.json',result)
  good = [r for r in result['rows'] if 'target_tokens_per_second' in r]
  result['best_batch_size'] = max(good,key=lambda r:r['target_tokens_per_second'])['batch_size']
  atomic_json(output/'benchmark.json',result)
  gcs = os.environ.get('PILE_BUCKET_GCS')
  if gcs:
    from google.cloud import storage
    bucket,prefix = gcs[5:].split('/',1)
    storage.Client(project='newproject-1-451205').bucket(bucket).blob(
        prefix.rstrip('/')+'/benchmark.json').upload_from_filename(output/'benchmark.json',checksum='crc32c')
  print('PILE_BATCH_BENCH_DONE '+str(result['best_batch_size']),flush=True)


def main(argv):
  import pyconfig
  config = pyconfig.initialize(argv)
  if os.environ.get('PILE_BUCKET_CONFIG_ONLY') == '1':
    if not config.only_eval or not config.load_parameters_path or config.load_full_state_path != '':
      raise ValueError('Invalid restore contract')
    print('PILE_BUCKET_CONFIG_OK '+config.exp_class,flush=True)
    return
  benchmark(config)


if __name__ == '__main__':
  from absl import app
  app.run(main)
