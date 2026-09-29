"""Read-only initialization diagnosis with scalar forward/backward matrix taps.

All interventions share identical common parameters and cached Pile sequences.
Stops in dynamic-write gradients change only the backward path, not the forward.
No optimizer update, checkpoint restore or checkpoint save is performed.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from absl import app
from flax import core
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import max_utils
import pyconfig
import train
from input_pipeline.input_pipeline_interface import create_data_iterator

BIAS = ('params', 'decoder', 'dynamic_embedding_write', 'address_up_bias')
CASES = {
    'vector': (True, False, 'all'),
    'matrix': (False, False, 'all'),
    'both': (True, True, 'all'),
    'stop_address': (True, False, 'stop_address'),
    'stop_content': (True, False, 'stop_content'),
    'stop_both': (True, False, 'stop_both'),
    'stop_attn_content': (True, False, 'stop_attn_content'),
    'stop_mlp_content': (True, False, 'stop_mlp_content'),
    'stop_qk': (True, False, 'all'),
    'stop_v': (True, False, 'all'),
    'qk_norm': (True, False, 'all'),
}


def main(argv):
  cfg = pyconfig.initialize(argv)
  if not cfg.only_eval or cfg.enable_checkpointing or cfg.load_parameters_path or cfg.load_full_state_path:
    raise ValueError('Initialization-only probe: only_eval=True, no checkpoints')
  cfg.get_keys().update(rmt_record_dynamic_health=False, rmt_norm_probe=False,
                        rmt_vector_pre_norm=True, rmt_probe_write_grad='all')
  rng, writer, manager, mesh, model, _, tx = train.setup_mesh_and_model(cfg)
  iterator, _ = create_data_iterator(cfg, mesh)
  state, _, _, iterator = max_utils.setup_training_state(model, iterator, tx, cfg, rng, mesh, manager)
  flat = flatten_dict(state.params)
  # All cases keep the same parameter tree. Unused matrix gains are constants at
  # identity in vector mode; no random reinitialization of common parameters.
  shape = [int(cfg.rmt_reskey_dim), int(cfg.head_dim)]
  shape.insert(cfg.param_scan_axis, int(cfg.num_decoder_layers))
  for arm in ('attn_norm', 'mlp_norm'):
    flat[('params', 'decoder', 'layers', arm, 'scale')] = jnp.ones(shape, cfg.weight_dtype)
  params = unflatten_dict(flat)
  if isinstance(state.params, core.FrozenDict):
    params = core.freeze(params)
  bias = flat[BIAS]
  count = int(os.environ.get('RMT_NORM_BATCHES', '4'))
  output = Path(os.environ.get('RMT_NORM_OUTPUT', '/tmp/rmt-norm-probe'))
  output.mkdir(parents=True, exist_ok=True)
  batches, hashes = [], []
  for _ in range(count):
    batch = next(iterator)
    batches.append(batch)
    hashes.append([{'index': str(s.index), 'sha256': hashlib.sha256(np.asarray(s.data).tobytes()).hexdigest()}
                   for s in batch['inputs'].addressable_shards])
  metadata = dict(exp=cfg.exp_class, layers=int(cfg.num_decoder_layers), heads=int(cfg.num_query_heads),
                  head_dim=int(cfg.head_dim), shape_M=[int(cfg.rmt_reskey_dim), int(cfg.head_dim)],
                  init_seed=int(cfg.init_weights_seed), sequence_length=int(cfg.max_target_length),
                  global_batch=int(cfg.global_batch_size_to_train_on), hashes=hashes)
  (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
  for case in os.environ.get('RMT_NORM_CASES', ','.join(CASES)).split(','):
    vector, matrix, write_grad = CASES[case]
    cfg.get_keys().update(rmt_norm_probe=True, rmt_vector_pre_norm=vector,
                          rmt_probe_matrix_pre_norm=matrix, rmt_probe_write_grad=write_grad,
                          rmt_probe_attention_grad=case if case in ('stop_qk', 'stop_v') else 'all',
                          rmt_probe_qk_matrix_rms=case == 'qk_norm')
    def objective(p, b, batch):
      f = flatten_dict(p)
      f[BIAS] = b
      effective = unflatten_dict(f)
      if isinstance(p, core.FrozenDict): effective = core.freeze(effective)
      return train.loss_fn(model, cfg, dict(batch), rng, effective, is_train=True)[0]
    compiled = jax.jit(jax.value_and_grad(objective, argnums=1))
    started = time.monotonic()
    for index, batch in enumerate(batches):
      filename = output/f'{case}-{index}-taps.jsonl'
      filename.write_text('')
      os.environ['RMT_NORM_TAP_FILE'] = str(filename)
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        loss, grad = compiled(params, bias, batch)
      loss, grad = jax.device_get(jax.block_until_ready((loss, grad)))
      jax.effects_barrier()
      record = dict(case=case, batch=index, loss=float(loss), bias_grad_norm=float(np.linalg.norm(grad)),
                    bias_grad_rms=float(np.sqrt(np.mean(np.square(grad)))), elapsed=time.monotonic()-started)
      with (output/'summary.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
      print('NORM_PROBE '+json.dumps(record), flush=True)
    del compiled
    jax.clear_caches()
  if writer: writer.close()
  print('NORM_PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
  app.run(main)
