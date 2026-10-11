"""Read-only checkpoint probe separating token and head static-gate variation."""
import argparse
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import pyconfig
from layers import attentions, quantizations
from layers.models import Transformer

EXP = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
original_gate = attentions.BamAttention._static_gate
capture = True


def measured_gate(self, name, x):
  value = original_gate(self, name, x)
  if capture and not self.is_initializing() and name == 'general_v_static_gate':
    self.sow('intermediates', 'diag_v_static_gate', value.astype(jnp.float32))
  return value


attentions.BamAttention._static_gate = measured_gate


def select_gates(collection):
  selected = {}
  for path, value in flatten_dict(collection).items():
    if path[-1] != 'diag_v_static_gate':
      continue
    while isinstance(value, (tuple, list)) and len(value) == 1:
      value = value[0]
    offsets = [int(m.group(1)) for part in path
               if (m := re.fullmatch(r'(?:local|fetch)_(\d+)', part))]
    assert len(offsets) == 1, path
    assert value.ndim == 4, (path, value.shape)
    for block in range(value.shape[0]):
      layer = 3 * block + offsets[0]
      assert layer not in selected, (layer, path)
      selected[layer] = value[block]
  assert selected, 'No captured gates'
  return jnp.stack([selected[i] for i in range(len(selected))])


def gate_moments(gates):
  # gates[layer, sequence, token, head]; full tokens, not saved stride samples.
  g = np.asarray(gates, np.float64)
  assert g.ndim == 4 and np.isfinite(g).all()
  per_head_mean = g.mean(axis=(1, 2))
  per_head_var = g.var(axis=(1, 2))
  total = g.var(axis=(1, 2, 3))
  token = per_head_var.mean(axis=-1)
  head = per_head_mean.var(axis=-1)
  within_sequence = g.var(axis=2).mean(axis=(1, 2))
  sequence = g.mean(axis=2).var(axis=1).mean(axis=-1)
  np.testing.assert_allclose(total, token + head, rtol=1e-9, atol=1e-12)
  np.testing.assert_allclose(token, within_sequence + sequence, rtol=1e-9, atol=1e-12)
  return dict(mean=g.mean(axis=(1, 2, 3)), total_variance=total,
              within_head_token_variance=token, between_head_mean_variance=head,
              within_sequence_token_variance=within_sequence,
              between_sequence_within_head_variance=sequence,
              token_fraction=np.divide(token, total, out=np.zeros_like(total), where=total > 0),
              per_head_mean=per_head_mean, per_head_token_std=np.sqrt(per_head_var),
              per_head_p05=np.quantile(g, .05, axis=(1, 2)),
              per_head_p95=np.quantile(g, .95, axis=(1, 2)))


def forward(model, params, tokens, targets, measured=True):
  positions = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
  segments = jnp.ones_like(tokens)
  result = model.apply({'params': params}, tokens, positions,
      decoder_segment_ids=segments, decoder_target_mask=segments,
      decoder_target_tokens=targets, enable_dropout=False,
      rngs={'aqt': jax.random.PRNGKey(0)},
      mutable=['intermediates'] if measured else False)
  if measured:
    (xent, _, _), collections = result
    return xent, select_gates(collections['intermediates'])
  return result[0]


def self_test():
  # Head-specific constants must NOT be mistaken for token adaptation.
  a = np.broadcast_to(np.array([.2, .8])[None, None, None], (1, 2, 5, 2))
  stats = gate_moments(a)
  np.testing.assert_allclose(stats['within_head_token_variance'], 0, atol=1e-15)
  np.testing.assert_allclose(stats['between_head_mean_variance'], .09)
  b = np.broadcast_to(np.array([.2, .8])[None, None, :, None], (1, 2, 2, 3))
  stats = gate_moments(b)
  np.testing.assert_allclose(stats['token_fraction'], 1)
  with tempfile.TemporaryDirectory() as root:
    cfg = make_config(root, length=4, checkpoint='')
    cfg.get_keys().update(base_emb_dim=300, emb_dim=300, base_num_query_heads=4,
        base_num_kv_heads=4, num_query_heads=4, num_kv_heads=4, emb_bam_num_head=4,
        bam_mlp_write_num_heads=0, bam_write_v_bottleneck_dim=16,
        emb_bam_v_bottleneck_dim=16, bam_mlp_write_address_rank=16,
        base_num_decoder_layers=3, num_decoder_layers=3,
        bam_layer_modes=['local_qk+local_v+local_o'] * 3,
        base_mlp_dim=32, mlp_dim=32, mlp_dim_by_block=[32, 24, 32],
        vocab_size=128, dtype='float32', weight_dtype='float32')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(cfg.mesh_axes)), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.array([[1, 4, 8, 2]], jnp.int32)
    pos = jnp.arange(4)[None]
    rng = {n: jax.random.key(i) for i, n in enumerate(('params', 'dropout', 'aqt'))}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      variables = model.init(rng, tokens, pos, jnp.ones_like(tokens), tokens)
      assert not any('diag_v_static_gate' in '/'.join(p) for p in flatten_dict(variables))
      params = nn.unbox(variables['params'])
      observed, gates = jax.jit(lambda p: forward(model, p, tokens, tokens))(params)
      native = jax.jit(lambda p: forward(model, p, tokens, tokens, False))(params)
    np.testing.assert_allclose(observed, native, rtol=1e-6, atol=1e-7)
    assert gates.shape == (3, 1, 4, 4), gates.shape
    np.testing.assert_allclose(gates, .99, rtol=1e-6)
  print('GATE_PROBE_SELF_TEST_OK', flush=True)


def make_config(root, length, checkpoint):
  Path(root, 'gateprobe').mkdir(parents=True, exist_ok=True)
  return pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=EXP,
      run_name='gateprobe', base_output_directory=root + '/', only_eval=True,
      enable_checkpointing=False, dataset_type='synthetic', per_device_batch_size=1.,
      max_target_length=length, max_prefill_predict_length=length,
      query_chunk_size=min(256, length),
      load_parameters_path=checkpoint, jax_cache_dir='', log_config=False,
      bam_splash_attention=False if jax.default_backend() == 'cpu' else True)


def run(args):
  out = Path(args.output)
  out.mkdir(parents=True, exist_ok=True)
  cohort = np.load(args.cohort)
  tokens, targets = cohort['inputs'][:args.samples], cohort['targets'][:args.samples]
  hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in tokens]
  start = time.time()
  cfg = make_config(str(out / 'runtime'), tokens.shape[1], args.checkpoint)
  assert cfg.num_decoder_layers == 18 and cfg.bam_general_column_read
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = Transformer(cfg, mesh, quantizations.configure_quantization(cfg))
  state, _ = max_utils.setup_decode_state(model, cfg, jax.random.PRNGKey(cfg.init_weights_seed), mesh, None)
  count = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(state.params))
  assert count == 432111104, count
  print('PARAMS_RESTORED', args.checkpoint, int(count), time.time() - start, flush=True)
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    compiled = jax.jit(lambda p, t, y: forward(model, p, t, y))
    native = jax.jit(lambda p, t, y: forward(model, p, t, y, False))
    all_gates, sequence_loss = [], []
    for i, (t, y) in enumerate(zip(tokens, targets)):
      before = time.time()
      xent, gates = compiled(state.params, jnp.asarray(t[None]), jnp.asarray(y[None]))
      xent, gates = jax.device_get((xent, gates))
      assert gates.shape == (18, 1, tokens.shape[1], 16), gates.shape
      if i == 0:
        reference = np.asarray(native(state.params, jnp.asarray(t[None]), jnp.asarray(y[None])))
        np.testing.assert_allclose(xent, reference, rtol=1e-5, atol=1e-5)
        print('CAPTURE_FORWARD_PARITY_OK', float(np.max(np.abs(xent - reference))), flush=True)
      all_gates.append(gates)
      sequence_loss.append(float(np.mean(xent)))
      np.savez_compressed(out / f'gates-{i:03d}.npz', gate=gates[:, :, ::16],
          token_positions=np.arange(0, tokens.shape[1], 16), sequence_hash=hashes[i])
      print('GATE_PROBE_SEQUENCE', i, time.time() - before, sequence_loss[-1], flush=True)
  gates = np.concatenate(all_gates, axis=1)
  stats = gate_moments(gates)
  result = dict(exp=EXP, checkpoint=args.checkpoint, samples=len(tokens),
      length=tokens.shape[1], hashes=hashes, sequence_loss=sequence_loss,
      capture_parity=True, layers=[], elapsed_seconds=time.time() - start)
  for layer in range(18):
    row = {'layer': layer}
    for key, value in stats.items():
      row[key] = value[layer].tolist()
    row['per_sequence'] = []
    for i in range(len(tokens)):
      one = gate_moments(gates[layer:layer+1, i:i+1])
      row['per_sequence'].append({key: value[0].tolist() for key, value in one.items()})
    result['layers'].append(row)
  (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
  print('GATE_PROBE_DONE', str(out / 'summary.json'), result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--self-test', action='store_true')
  parser.add_argument('--checkpoint')
  parser.add_argument('--cohort')
  parser.add_argument('--output')
  parser.add_argument('--samples', type=int, default=32)
  args = parser.parse_args()
  if args.self_test:
    self_test()
  else:
    assert args.checkpoint and args.cohort and args.output
    run(args)
