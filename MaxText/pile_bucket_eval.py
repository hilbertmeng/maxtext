"""Read-only full-file checkpoint evaluation with position-bucket cross entropy.

PILE_BUCKET_SOURCE points to the local original TFRecord; PILE_BUCKET_OUTPUT is
an output directory. PILE_BUCKET_GCS is a unique, per-model artifact prefix.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from pile_validation_data import atomic_json, file_integrity, read_records

SOURCE_URI = 'gs://newproject-1-common_datasets_us-east5/pythia_model_test/pile_test/val_with_eos.tfrecord'
SOURCE_SHA256 = '44c3c5efa34f5a1f6c4cc2f3f6fe92974a8c7ef3fd14af029c9bcb8f485a4f40'
SOURCE_RECORDS = 166398
# Actual training registries; evaluation window length is independently recorded.
TRAINING_RUNS = {
    'BamMHAMediumPropC256T2048Matched': ('e9f77281a61d1e3dfa8416574655f48d28d98481', 2048, 432121200),
    'MuddLlama2MediumPropC256T2048': ('e9f77281a61d1e3dfa8416574655f48d28d98481', 2048, 434076747),
    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048':
        ('55deb7c34ae85530b2276b933a5aac5e99e6c0ba', 2048, 432111104),
    'BamMHAMediumPropC256TruePile': ('0d31a3d4f2df68885d5c6e3d61cceeb8986adf2f', 4096, 432121200),
    'MuddLlama2MediumPropTruePile': ('4fee4903a40d0e3f1fa461f572464daa78df4499', 4096, 434076747),
    'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile':
        ('64e3fc326b39d9d7ec9ebcfcdbe961c7ed46bdbb', 4096, 432111104),
}


def prepare_data(source, directory, sequence_length=2048, expected_sha256=SOURCE_SHA256,
                 expected_records=SOURCE_RECORDS):
  """Preserve original record boundaries, EOS and the short final record."""
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  ready = directory / 'data.json'
  if ready.exists():
    meta = json.loads(ready.read_text())
    if (meta['source_integrity']['sha256'] != expected_sha256 or
        meta['records'] != expected_records or meta['sequence_length'] != sequence_length):
      raise ValueError('Cached validation source/shape differs')
    for name in ('tokens.npy', 'lengths.npy'):
      if file_integrity(directory / name) != meta['arrays'][name]:
        raise ValueError('Cached array checksum differs')
    return meta
  integrity = file_integrity(source)
  if integrity['sha256'] != expected_sha256:
    raise ValueError('Validation TFRecord differs from the audited original')
  tokens = np.lib.format.open_memmap(directory / 'tokens.npy', mode='w+', dtype='<u2',
                                    shape=(expected_records, sequence_length + 1))
  lengths = np.lib.format.open_memmap(directory / 'lengths.npy', mode='w+', dtype='<u2',
                                     shape=(expected_records,))
  tokens[:] = 0
  count = stored = 0
  for i, row in enumerate(read_records(source)):
    if i >= expected_records or not 2 <= len(row) <= sequence_length + 1:
      raise ValueError('Unexpected validation record count or length')
    tokens[i, :len(row)] = row
    lengths[i] = len(row)
    stored += len(row)
    count += 1
  if count != expected_records:
    raise ValueError('Validation record count differs')
  tokens.flush()
  lengths.flush()
  meta = {
      'source_uri': SOURCE_URI, 'source_integrity': integrity,
      'records': count, 'stored_tokens': stored,
      'target_tokens_before_eos_mask': stored - count,
      'sequence_length': sequence_length, 'record_order': 'original; no rewindow or shuffle',
      'arrays': {name: file_integrity(directory / name) for name in ('tokens.npy','lengths.npy')},
  }
  atomic_json(ready, meta)
  return meta


def make_batch(tokens, lengths, first, batch_size, sequence_length):
  count = min(batch_size, len(lengths) - first)
  ids = np.zeros((batch_size, sequence_length + 1), dtype=np.int32)
  valid_lengths = np.zeros(batch_size, dtype=np.int32)
  ids[:count] = tokens[first:first+count]
  valid_lengths[:count] = np.asarray(lengths[first:first+count], np.int32) - 1
  positions = np.broadcast_to(np.arange(sequence_length, dtype=np.int32), (batch_size, sequence_length))
  mask = (positions < valid_lengths[:, None]).astype(np.int32)
  return {
      'inputs': ids[:, :-1], 'targets': ids[:, 1:],
      'inputs_position': positions, 'inputs_segmentation': mask,
      'targets_segmentation': mask,
  }, count


def summarize_xent(xent, targets, valid, bucket_size=256):
  import jax.numpy as jnp
  b, t = xent.shape
  if t % bucket_size:
    raise ValueError('Sequence length must be divisible by bucket size')
  valid = valid != 0
  result = {}
  for policy, mask in (('including_eos', valid), ('excluding_eos', valid & (targets != 0))):
    masked = jnp.where(mask, xent.astype(jnp.float32), 0)
    result['sum_' + policy] = masked.reshape(b, t // bucket_size, bucket_size).sum(-1)
    result['count_' + policy] = mask.reshape(b, t // bucket_size, bucket_size).sum(-1)
    result['total_' + policy] = masked.sum()
  result['nonfinite_tokens'] = jnp.sum(valid & ~jnp.isfinite(xent))
  return result


def forward(model, params, batch, rng, bucket_size):
  xent, _, _ = model.apply(
      params, batch['inputs'], batch['inputs_position'],
      decoder_segment_ids=batch['inputs_segmentation'],
      decoder_target_mask=batch['targets_segmentation'],
      decoder_target_tokens=batch['targets'], enable_dropout=False,
      rngs={'dropout': rng, 'params': rng})
  return summarize_xent(xent, batch['targets'], batch['targets_segmentation'], bucket_size)


def restore_parameters(model, config, rng, mesh):
  # The helper manages its own logical-axis scopes. An outer scope makes Flax
  # constrain abstract ShapeDtypeStruct leaves while unboxing outside JIT.
  import max_utils
  state, _ = max_utils.setup_decode_state(model, config, rng, mesh, None)
  return state


def run(config):
  from flax.linen import partitioning as nn_partitioning
  import jax
  from jax.sharding import Mesh
  import max_utils
  import pyconfig
  from layers.models import Transformer
  from layers import quantizations

  started = time.monotonic()
  if not config.only_eval or not config.load_parameters_path:
    raise ValueError('Require only_eval=True and an explicit parameters path')
  if jax.default_backend() != 'tpu':
    raise ValueError('Checkpoint evaluation requires a TPU backend')
  if config.load_full_state_path or jax.process_count() != 1:
    raise ValueError('Use parameter-only restore on one TPU host')
  output = Path(os.environ['PILE_BUCKET_OUTPUT'])
  output.mkdir(parents=True, exist_ok=True)
  source = Path(os.environ['PILE_BUCKET_SOURCE'])
  gcs = os.environ.get('PILE_BUCKET_GCS', '').rstrip('/')
  bucket_size = int(os.environ.get('PILE_BUCKET_SIZE', '256'))
  save_every = int(os.environ.get('PILE_BUCKET_SAVE_EVERY', '64'))
  max_batches = int(os.environ.get('PILE_BUCKET_MAX_BATCHES', '0'))
  batch_size = int(config.eval_per_device_batch_size * jax.device_count())
  if bucket_size < 1 or save_every < 1 or max_batches < 0 or batch_size < 1:
    raise ValueError('Invalid evaluation limits')
  raw = dict(config.get_keys())
  overrides = {'record_internal_nn_metrics': False, 'record_training_health_metrics': False,
               'bam_diagnostics': False}
  for key in raw:
    if key.startswith('bam_record_'):
      overrides[key] = False
  raw.update(overrides)
  config = pyconfig.HyperParameters(SimpleNamespace(keys=raw))
  meta = prepare_data(source, output / 'data', config.max_target_length)
  tokens = np.load(output / 'data/tokens.npy', mmap_mode='r')
  lengths = np.load(output / 'data/lengths.npy', mmap_mode='r')
  commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
  identity = {
      'exp_class': config.exp_class, 'checkpoint': config.load_parameters_path,
      'checkpoint_step': int(config.load_parameters_path.rstrip('/').split('/')[-2]),
      'training_runtime_commit': TRAINING_RUNS.get(config.exp_class, (None,None,None))[0],
      'training_sequence_length': TRAINING_RUNS.get(config.exp_class, (None,None,None))[1],
      'runtime_commit': commit, 'source_sha256': meta['source_integrity']['sha256'],
      'sequence_length': config.max_target_length, 'bucket_size': bucket_size,
      'batch_size': batch_size, 'health_overrides': overrides,
      'resolved_config': json.loads(json.dumps(config.get_keys(), default=str)),
  }
  fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
  k = config.max_target_length // bucket_size
  totals = {name: np.zeros(k, dtype=np.float64 if name.startswith('sum_') else np.int64)
            for name in ('sum_including_eos','sum_excluding_eos','count_including_eos','count_excluding_eos')}
  first = 0
  chunk_number = 0
  progress = output / 'progress.json'
  if progress.exists():
    saved = json.loads(progress.read_text())
    if saved['fingerprint'] != fingerprint:
      raise ValueError('Resume identity differs from the saved evaluation')
    first, chunk_number = saved['records'], saved['chunks']
    totals = {name: np.asarray(saved['totals'][name], dtype=totals[name].dtype) for name in totals}
  elif (output / 'result.json').exists():
    raise ValueError('Result exists without a resume ledger')

  upload_bucket = upload_prefix = None
  if gcs:
    from google.cloud import storage
    bucket_name, upload_prefix = gcs[5:].split('/', 1)
    upload_bucket = storage.Client(project='newproject-1-451205').bucket(bucket_name)

  def upload(path, destination=None):
    if upload_bucket is not None:
      upload_bucket.blob(upload_prefix + '/' + (destination or path.name)).upload_from_filename(
          str(path), checksum='crc32c', timeout=120)

  def save_progress(chunk):
    nonlocal chunk_number
    if chunk:
      name = f'sequences_{chunk_number:05d}.npz'
      np.savez_compressed(output / name,
          first_record=np.array(chunk[0]['first']),
          **{key: np.concatenate([row[key] for row in chunk]) for key in totals})
      upload(output / name)
      chunk_number += 1
    state = {'fingerprint': fingerprint, 'identity': identity, 'records': first,
             'chunks': chunk_number, 'totals': {name: arr.tolist() for name,arr in totals.items()}}
    atomic_json(progress, state)
    upload(progress)

  print(f'PILE_BUCKET_PREPARED records={meta["records"]} batch={batch_size} resume={first}', flush=True)
  rng = jax.random.PRNGKey(config.init_weights_seed)
  mesh = Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
  model = Transformer(config, mesh, quant=quantizations.configure_quantization(config))
  state = restore_parameters(model, config, rng, mesh)
  param_count = int(max_utils.calculate_num_params_from_pytree(state.params))
  expected_params = TRAINING_RUNS.get(config.exp_class, (None,None,None))[2]
  if expected_params is not None and param_count != expected_params:
    raise ValueError(f'Checkpoint parameter count differs: {param_count}')
  compiled = jax.jit(lambda params,batch,key: forward(model,params,batch,key,bucket_size))
  setup_seconds = time.monotonic() - started
  print(f'PILE_BUCKET_RESTORED parameters={param_count} setup_seconds={setup_seconds:.2f}', flush=True)
  count_batches = 0
  chunk = []
  timings = []
  eval_started = time.monotonic()
  while first < meta['records'] and (not max_batches or count_batches < max_batches):
    batch, count = make_batch(tokens, lengths, first, batch_size, config.max_target_length)
    t0 = time.monotonic()
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      measured = jax.device_get(jax.block_until_ready(compiled(state.params,batch,rng)))
    seconds = time.monotonic() - t0
    if int(measured['nonfinite_tokens']):
      raise FloatingPointError('Nonfinite loss on valid target positions')
    row = {'first': first}
    for key in totals:
      values = np.asarray(measured[key])[:count]
      if key.startswith('sum_'):
        policy = key.removeprefix('sum_')
        if not np.isclose(values.sum(dtype=np.float64), float(measured['total_'+policy]),rtol=2e-6,atol=0.02):
          raise ValueError('Bucket/global loss sums differ')
      row[key] = values
      totals[key] += values.sum(axis=0,dtype=totals[key].dtype)
    chunk.append(row)
    first += count
    count_batches += 1
    timings.append(seconds)
    if count_batches <= 3 or count_batches % 25 == 0:
      loss = totals['sum_including_eos'].sum()/totals['count_including_eos'].sum()
      print(f'PILE_BUCKET_PROGRESS records={first}/{meta["records"]} loss={loss:.8f} '
            f'batch_seconds={seconds:.4f} eval_seconds={time.monotonic()-eval_started:.1f}',flush=True)
    if len(chunk) >= save_every:
      save_progress(chunk)
      chunk = []
  save_progress(chunk)
  complete = first == meta['records']
  if complete:
    expected_count = np.zeros(k,dtype=np.int64)
    expected_non_eos = np.zeros(k,dtype=np.int64)
    for start in range(0,len(lengths),1024):
      batch,_ = make_batch(tokens,lengths,start,min(1024,len(lengths)-start),config.max_target_length)
      mask = batch['targets_segmentation'].astype(bool)
      expected_count += mask.reshape(-1,k,bucket_size).sum(axis=(0,2))
      expected_non_eos += (mask & (batch['targets']!=0)).reshape(-1,k,bucket_size).sum(axis=(0,2))
    if not np.array_equal(expected_count,totals['count_including_eos']) or not np.array_equal(expected_non_eos,totals['count_excluding_eos']):
      raise ValueError('Full-file bucket target coverage differs')
  result = {
      'complete': complete, 'identity': identity, 'dataset': meta,
      'records_evaluated': first, 'parameter_count': param_count,
      'position_semantics': 'zero-based prediction slots [start,end); slot s predicts original record token index s+1',
      'bucket_size': bucket_size, 'bucket_count': k,
      'policies': {}, 'setup_seconds': setup_seconds,
      'session_eval_seconds': time.monotonic()-eval_started,
      'session_batches': count_batches,
      'median_batch_seconds_excluding_first': float(np.median(timings[1:])) if len(timings)>1 else None,
      'sequence_chunks': chunk_number, 'jax_version': jax.__version__,
      'devices': [str(d) for d in jax.devices()],
      'checkpoint_manager_created': False, 'checkpoint_writer_created': False,
  }
  for policy in ('including_eos','excluding_eos'):
    sums,counts = totals['sum_'+policy],totals['count_'+policy]
    all_loss = float(sums.sum()/counts.sum())
    result['policies'][policy] = {
        'global_loss': all_loss, 'perplexity': math.exp(all_loss),
        'loss_sum': float(sums.sum()), 'token_count': int(counts.sum()),
        'buckets': [{'index':i,'prediction_start':i*bucket_size,'prediction_end_exclusive':(i+1)*bucket_size,
                     'loss_sum':float(sums[i]),'token_count':int(counts[i]),
                     'loss':float(sums[i]/counts[i]) if counts[i] else None}
                    for i in range(k)]}
  atomic_json(output/'result.json',result)
  upload(output/'result.json')
  print('PILE_BUCKET_DONE '+json.dumps({'complete':complete,'records':first,
        **{p:result['policies'][p]['global_loss'] for p in result['policies']}}),flush=True)


def main(argv):
  import pyconfig
  config = pyconfig.initialize(argv)
  if os.environ.get('PILE_BUCKET_CONFIG_ONLY') == '1':
    if not config.only_eval or not config.load_parameters_path or config.load_full_state_path != '':
      raise ValueError('Invalid parameter-only evaluation launch contract')
    if config.max_target_length != 2048 or config.eval_per_device_batch_size != 16:
      raise ValueError('Original Pile evaluation must use T2048 and batch16')
    if config.exp_class in ('BamMHAMediumPropC256TruePile', 'MuddLlama2MediumPropTruePile') and config.bam_splash_attention:
      raise ValueError('Historical MHA/Mudd require the C256 attention implementation')
    print('PILE_BUCKET_CONFIG_OK ' + config.exp_class, flush=True)
    return
  run(config)


if __name__ == '__main__':
  from absl import app
  app.run(main)
