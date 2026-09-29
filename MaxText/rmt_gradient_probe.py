"""Paired from-scratch RMT optimizer probe; never loads or saves training checkpoints.

Uses the official initializer, Pile loader, loss, clipping and Adam optimizer.
RMT_PROBE_ARMS=baseline,no_embedding_bias; RMT_PROBE_STEPS=200.
The no-bias arm substitutes a constant zero ONLY for embedding address_up_bias.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from absl import app
from flax import core
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import optax
import max_utils
import maxtext_utils
import pyconfig
import train
from input_pipeline.input_pipeline_interface import create_data_iterator

BIAS = ('params', 'decoder', 'dynamic_embedding_write', 'address_up_bias')


def without_embedding_bias(params):
  flat = flatten_dict(params)
  if BIAS not in flat:
    raise ValueError(f'Missing expected parameter {BIAS}')
  flat[BIAS] = jnp.zeros_like(flat[BIAS])
  result = unflatten_dict(flat)
  return core.freeze(result) if isinstance(params, core.FrozenDict) else result


def main(argv):
  cfg = pyconfig.initialize(argv)
  if cfg.enable_checkpointing or cfg.load_parameters_path or cfg.load_full_state_path:
    raise ValueError('This initialization probe forbids checkpoint loading/writing')
  if not cfg.base_output_directory.startswith('/tmp/'):
    raise ValueError('Diagnostic output must be local /tmp')
  if cfg.gradient_accumulation_steps != 1:
    raise ValueError('Probe requires one microbatch')
  cfg.get_keys()['rmt_record_dynamic_health'] = False
  rng, writer, manager, mesh, model, lr, tx = train.setup_mesh_and_model(cfg)
  iterator, _ = create_data_iterator(cfg, mesh)
  initial, _, shardings, iterator = max_utils.setup_training_state(
      model, iterator, tx, cfg, rng, mesh, manager)
  count = int(os.environ.get('RMT_PROBE_STEPS', '200'))
  arms = os.environ.get('RMT_PROBE_ARMS', 'baseline,no_embedding_bias').split(',')
  output = Path('/tmp/rmt-gradient-probe')
  output.mkdir(exist_ok=True)
  hashes = []
  # Keep batches on their original device shards; all arms consume identical batches.
  batches = []
  for i in range(count):
    batch = next(iterator)
    batches.append(batch)
    hashes.append([{'index': str(s.index), 'sha256': hashlib.sha256(
        np.asarray(s.data).tobytes()).hexdigest()} for s in batch['inputs'].addressable_shards])
  (output / f'cohort-{jax.process_index()}.json').write_text(json.dumps(hashes))
  print(f'PROBE_READY batches={count} devices={jax.device_count()}', flush=True)

  def step(state, batch, key, *, no_bias):
    def objective(params):
      if no_bias:
        params = without_embedding_bias(params)
      return train.loss_fn(model, cfg, dict(batch), key, params, is_train=True)[0]
    loss, raw = jax.value_and_grad(objective)(state.params)
    norm = optax.global_norm(raw)
    clipped = maxtext_utils.apply_gradient_clipping(raw, state, cfg.gradient_clipping_threshold)
    updated = state.apply_gradients(grads=clipped)
    flat = flatten_dict(raw)
    flat_clipped = flatten_dict(clipped)
    before = flatten_dict(state.params)
    after = flatten_dict(updated.params)
    metrics = {'loss': loss, 'grad_norm': norm, 'bias_grad_norm': jnp.linalg.norm(flat[BIAS]),
               'lr': lr(state.step)}
    for path in flat:
      name = '/'.join(path)
      if (path == BIAS or 'logits_dense' in name or
          ('layers' in name and ('/mlp/' in name or '/MlpBlock_' in name) and name.endswith('kernel'))):
        # Scalars only, never gather complete parameter or gradient arrays.
        metrics[name + '/raw_rms'] = jnp.sqrt(jnp.mean(jnp.square(flat[path].astype(jnp.float32))))
        metrics[name + '/clipped_rms'] = jnp.sqrt(jnp.mean(jnp.square(flat_clipped[path].astype(jnp.float32))))
        metrics[name + '/update_rms'] = jnp.sqrt(jnp.mean(jnp.square(
            after[path].astype(jnp.float32) - before[path].astype(jnp.float32))))
    return updated, metrics

  compiled = jax.jit(step, static_argnames=('no_bias',), out_shardings=(shardings, None))
  for arm in arms:
    if arm not in ('baseline', 'no_embedding_bias'):
      raise ValueError(arm)
  states = {arm: initial for arm in arms}
  started = time.monotonic()
  # Interleave arms so paired evidence is available at every observation point.
  for i, batch in enumerate(batches):
    for arm in arms:
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        states[arm], metrics = compiled(states[arm], batch, jax.random.fold_in(rng, i),
                                       no_bias=arm != 'baseline')
      metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
      record = {'arm': arm, 'step': i, 'elapsed': time.monotonic()-started, **metrics}
      if jax.process_index() == 0:
        with (output / f'{arm}.jsonl').open('a') as f:
          f.write(json.dumps(record) + '\n')
        if i % 10 == 0 or i == count-1:
          print('PROBE ' + json.dumps({k: record[k] for k in
                ('arm', 'step', 'loss', 'grad_norm', 'bias_grad_norm', 'elapsed')}), flush=True)
  if writer:
    writer.close()
  print('PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
  app.run(main)
