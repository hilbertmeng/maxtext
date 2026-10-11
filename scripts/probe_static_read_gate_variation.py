"""Read-only checkpoint probe separating token and head static-gate variation."""
import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

import max_utils
import pyconfig
from layers import attentions, quantizations
from layers.models import Transformer

EXP = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
original_gate = attentions.BamAttention._static_gate
capture = True
gate_controls = None
ARMS = ('q', 'k', 'v', 'o')


def measured_gate(self, name, x):
  value = original_gate(self, name, x)
  arm = name.removeprefix('general_').removesuffix('_static_gate')
  if capture and not self.is_initializing() and arm in ARMS:
    self.sow('intermediates', 'diag_' + arm + '_static_gate', value.astype(jnp.float32))
  if gate_controls is not None and arm in ARMS:
    # Fixed effective read multiplier=1, i.e. parent behavior. Physical gate .99.
    value = jnp.where(gate_controls[ARMS.index(arm)],
        jnp.full_like(value, self.config.bam_static_read_gate_init), value)
  return value


attentions.BamAttention._static_gate = measured_gate


def select_gates(collection, arm='v'):
  selected = {}
  for path, value in flatten_dict(collection).items():
    if path[-1] != 'diag_' + arm + '_static_gate':
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



def paired_vo_moments(v, o):
  """Pair exactly the same layer/sequence/token/head; avoid unpaired band means."""
  v, o = np.asarray(v, np.float64), np.asarray(o, np.float64)
  assert v.shape == o.shape and v.ndim == 4
  delta = v - o
  total_variance = delta.var(axis=(1, 2, 3))
  within_head = delta.var(axis=(1, 2)).mean(axis=-1)
  between_head = delta.mean(axis=(1, 2)).var(axis=-1)
  np.testing.assert_allclose(total_variance, within_head + between_head, rtol=1e-9, atol=1e-12)
  ratio = v / np.maximum(o, 1e-8)
  balance = v / np.maximum(v + o, 1e-8)
  rows = []
  for layer in range(len(v)):
    flat_v, flat_o = v[layer].ravel(), o[layer].ravel()
    vv = v[layer] - v[layer].mean(axis=(0, 1), keepdims=True)
    oo = o[layer] - o[layer].mean(axis=(0, 1), keepdims=True)
    denominator = np.sqrt(np.mean(vv**2) * np.mean(oo**2))
    row = dict(layer=layer, mean_delta=float(delta[layer].mean()),
        delta_std=float(delta[layer].std()), delta_quantiles=np.quantile(delta[layer], [.05, .25, .5, .75, .95]).tolist(),
        v_less_o_fraction=float(np.mean(delta[layer] < 0)),
        v_lt_half_o_fraction=float(np.mean(v[layer] < .5 * o[layer])),
        ratio_quantiles=np.quantile(ratio[layer], [.05, .25, .5, .75, .95]).tolist(),
        v_share_quantiles=np.quantile(balance[layer], [.05, .25, .5, .75, .95]).tolist(),
        overall_correlation=float(np.corrcoef(flat_v, flat_o)[0, 1]),
        within_head_token_correlation=float(np.mean(vv * oo) / denominator) if denominator else None,
        delta_within_head_token_variance=float(within_head[layer]),
        delta_between_head_mean_variance=float(between_head[layer]),
        delta_within_sequence_token_variance=float(delta[layer].var(axis=1).mean()),
        delta_token_variance_fraction=float(within_head[layer] / total_variance[layer]) if total_variance[layer] else 0.,
        per_head_mean_delta=delta[layer].mean(axis=(0, 1)).tolist(),
        per_head_delta_token_std=delta[layer].std(axis=(0, 1)).tolist())
    rows.append(row)
  return rows

def forward(model, params, tokens, targets, measured=True, controls=None):
  global gate_controls
  previous = gate_controls
  gate_controls = controls
  positions = jnp.broadcast_to(jnp.arange(tokens.shape[1]), tokens.shape)
  segments = jnp.ones_like(tokens)
  try:
    result = model.apply({'params': params}, tokens, positions,
      decoder_segment_ids=segments, decoder_target_mask=segments,
      decoder_target_tokens=targets, enable_dropout=False,
      rngs={'aqt': jax.random.PRNGKey(0)},
      mutable=['intermediates'] if measured else False)
  finally:
    gate_controls = previous
  if measured:
    (xent, _, _), collections = result
    return xent, jnp.stack([select_gates(collections['intermediates'], arm) for arm in ARMS])
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
  vo = paired_vo_moments(a, a * 2)
  np.testing.assert_allclose(vo[0]['mean_delta'], -.5)
  np.testing.assert_allclose(vo[0]['delta_token_variance_fraction'], 0, atol=1e-12)
  with tempfile.TemporaryDirectory() as root:
    restore_cfg = make_config(root, length=4, checkpoint='gs://probe-only/items')
    assert restore_cfg.only_eval and restore_cfg.enable_checkpointing
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
      params = decode_parameter_tree({'params': nn.unbox(variables['params'])})
      observed, gates = jax.jit(lambda p: forward(model, p, tokens, tokens))(params)
      native = jax.jit(lambda p: forward(model, p, tokens, tokens, False))(params)
      fixed = jax.jit(lambda p: forward(model, p, tokens, tokens, False, jnp.ones(4, bool)))(params)
      biased = zero_biases(params, ('q', 'k', 'vo'))
      assert len(biased[1]) == 9, biased[1]
      folded, _ = absorb_mean_qk_gates(params, {a: np.full((3, 4), .99) for a in ('q', 'k')}, ('q', 'k'))
      folded_loss = jax.jit(lambda p: forward(model, p, tokens, tokens, False))(folded)
    np.testing.assert_allclose(observed, native, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fixed, native, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(folded_loss, native, rtol=1e-6, atol=1e-7)
    assert gates.shape == (4, 3, 1, 4, 4), gates.shape
    np.testing.assert_allclose(gates, .99, rtol=1e-6)
  print('GATE_PROBE_SELF_TEST_OK', flush=True)


def make_config(root, length, checkpoint):
  Path(root, 'gateprobe').mkdir(parents=True, exist_ok=True)
  return pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=EXP,
      run_name='gateprobe', base_output_directory=root + '/', only_eval=True,
      enable_checkpointing=bool(checkpoint), dataset_type='synthetic', per_device_batch_size=1.,
      max_target_length=length, max_prefill_predict_length=length,
      query_chunk_size=min(256, length),
      load_parameters_path=checkpoint, jax_cache_dir='', log_config=False,
      skip_jax_distributed_system=os.environ.get('JAX_PLATFORMS') == 'cpu',
      bam_splash_attention=os.environ.get('JAX_PLATFORMS') != 'cpu')


def decode_parameter_tree(state_params):
  # MaxText decode TrainState stores the complete variable dict, unlike model.init['params'].
  assert set(state_params) == {'params'}, tuple(state_params)
  return state_params['params']


def zero_biases(params, arms):
  flat = flatten_dict(params)
  names = {'general_' + arm + '_pre_bias' for arm in arms}
  selected = []
  for path, value in flat.items():
    if path[-1] in names:
      selected.append('/'.join(path))
      flat[path] = jnp.zeros_like(value)
  assert all(any(p.endswith('/' + name) for p in selected) for name in names), selected
  return unflatten_dict(flat), selected



def absorb_mean_qk_gates(params, means, arms, opening=.99):
  """Remove dynamic gate variation, absorbing calibrated per-head amplitude into static keys."""
  flat = flatten_dict(params)
  modified = []
  for path, value in flat.items():
    for arm in arms:
      static = path[-1] == 'static_' + arm + '_key'
      bias = path[-1] == 'general_' + arm + '_static_gate_bias'
      kernel = path[-1] == 'kernel' and path[-2] == 'general_' + arm + '_static_gate'
      if not (static or bias or kernel):
        continue
      offsets = [int(m.group(1)) for part in path
                 if (m := re.fullmatch(r'(?:local|fetch)_(\d+)', part))]
      assert len(offsets) == 1, path
      band = np.asarray(means[arm])[offsets[0]::3]  # [scan_block, head]
      if static:
        assert value.shape[1:] == band.shape, (path, value.shape, band.shape)
        flat[path] = value * jnp.asarray(band[None] / opening, value.dtype)
      elif bias:
        flat[path] = jnp.full_like(value, np.log(opening / (1.1 - opening)))
      else:
        flat[path] = jnp.zeros_like(value)
      modified.append('/'.join(path))
  assert len(modified) == 9 * len(arms), modified
  return unflatten_dict(flat), modified

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
  params = decode_parameter_tree(state.params)
  count = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
  assert count == 432111104, count
  print('PARAMS_RESTORED', args.checkpoint, int(count), time.time() - start, flush=True)
  variants = {
      'original': ((), ()),
      'q_static_fixed1': (('q',), ()),
      'k_static_fixed1': (('k',), ()),
      'qk_static_fixed1': (('q', 'k'), ()),
      'v_static_fixed1': (('v',), ()),
      'q_pre_bias_zero': ((), ('q',)),
      'k_pre_bias_zero': ((), ('k',)),
      'vo_pre_bias_zero': ((), ('vo',)),
      'all_pre_bias_zero': ((), ('q', 'k', 'vo')),
      'qk_fixed1_all_bias_zero': (('q', 'k'), ('q', 'k', 'vo')),
  }
  prepared = {}
  for name, (fixed, biases) in variants.items():
    p, leaves = zero_biases(params, biases) if biases else (params, [])
    prepared[name] = (p, jnp.array([arm in fixed for arm in ARMS]), leaves)
  # Same-shape parameter variants reuse one executable; flags are runtime arguments.
  sequence_loss = {name: [] for name in variants}
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    compiled = jax.jit(lambda p, t, y: forward(model, p, t, y))
    ablated = jax.jit(lambda p, t, y, c: forward(model, p, t, y, False, c))
    all_gates = []
    for i, (t, y) in enumerate(zip(tokens, targets)):
      before = time.time()
      t, y = jnp.asarray(t[None]), jnp.asarray(y[None])
      xent, gates = jax.device_get(compiled(params, t, y))
      assert gates.shape == (4, 18, 1, tokens.shape[1], 16), gates.shape
      all_gates.append(gates)
      sequence_loss['original'].append(float(np.mean(xent)))
      np.savez_compressed(out / f'gates-{i:03d}.npz', gate=gates[:, :, :, ::16],
          arms=np.array(ARMS), token_positions=np.arange(0, tokens.shape[1], 16), sequence_hash=hashes[i])
      for name, (p, controls, _) in prepared.items():
        if name == 'original' and i > 0:
          continue
        result = np.asarray(ablated(p, t, y, controls))
        if name == 'original':
          np.testing.assert_allclose(xent, result, rtol=1e-5, atol=1e-5)
          print('CAPTURE_FORWARD_PARITY_OK', float(np.max(np.abs(xent - result))), flush=True)
        else:
          sequence_loss[name].append(float(np.mean(result)))
      partial = {name: {'loss': loss, 'delta': (np.array(loss) - np.array(sequence_loss['original'])).tolist()}
                 for name, loss in sequence_loss.items() if name != 'original'}
      (out / 'ablation-progress.json').write_text(json.dumps(dict(sequences=i+1, variants=partial), indent=2) + '\n')
      print('GATE_PROBE_SEQUENCE', i, time.time() - before, sequence_loss['original'][-1],
            {name: round(row['delta'][-1], 6) for name, row in partial.items()}, flush=True)
  result = dict(exp=EXP, checkpoint=args.checkpoint, samples=len(tokens),
      length=tokens.shape[1], hashes=hashes, sequence_loss=sequence_loss['original'],
      capture_parity=True, fixed_gate_convention='effective read multiplier 1; physical gate .99',
      gate_statistics={}, ablations={}, elapsed_seconds=time.time() - start)
  for arm_index, arm in enumerate(ARMS):
    gates = np.concatenate([g[arm_index] for g in all_gates], axis=1)
    stats = gate_moments(gates)
    rows = []
    for layer in range(18):
      row = {'layer': layer}
      for key, value in stats.items():
        row[key] = value[layer].tolist()
      row['per_sequence'] = []
      for i in range(len(tokens)):
        one = gate_moments(gates[layer:layer+1, i:i+1])
        row['per_sequence'].append({key: value[0].tolist() for key, value in one.items()})
      rows.append(row)
    result['gate_statistics'][arm] = rows
  # Calibrate on first8 sequences; evaluate on remaining24 to avoid evaluation self-calibration.
  calibration = min(8, len(tokens) // 2)
  means = {arm: np.concatenate([g[ARMS.index(arm)] for g in all_gates[:calibration]], axis=1).mean(axis=(1, 2))
           for arm in ('q', 'k')}
  result['gate_calibration'] = dict(sequence_count=calibration, hashes=hashes[:calibration],
      evaluation_hashes=hashes[calibration:], per_layer_head_mean={a: m.tolist() for a, m in means.items()})
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    for arms in (('q',), ('k',), ('q', 'k')):
      name = ''.join(arms) + '_static_mean_absorbed'
      p, modified = absorb_mean_qk_gates(params, means, arms)
      losses = []
      for t, y in zip(tokens[calibration:], targets[calibration:]):
        losses.append(float(np.asarray(ablated(p, jnp.asarray(t[None]), jnp.asarray(y[None]), jnp.zeros(4, bool))).mean()))
      delta = np.array(losses) - np.array(sequence_loss['original'][calibration:])
      result['ablations'][name] = dict(sequence_loss=losses, paired_delta=delta.tolist(),
          mean_delta=float(delta.mean()), standard_error=float(delta.std(ddof=1) / np.sqrt(len(delta))),
          loss_increased_sequences=int(np.sum(delta > 0)), evaluated_sequence_indices=list(range(calibration, len(tokens))),
          modified_parameter_paths=modified)
      print('MEAN_GATE_ABLATION', name, result['ablations'][name]['mean_delta'], flush=True)
  result['paired_vo_gate'] = paired_vo_moments(
      np.concatenate([g[ARMS.index('v')] for g in all_gates], axis=1),
      np.concatenate([g[ARMS.index('o')] for g in all_gates], axis=1))
  for name, losses in sequence_loss.items():
    delta = np.array(losses) - np.array(sequence_loss['original'])
    result['ablations'][name] = dict(sequence_loss=losses, paired_delta=delta.tolist(),
        mean_delta=float(delta.mean()), standard_error=float(delta.std(ddof=1) / np.sqrt(len(delta))),
        loss_increased_sequences=int(np.sum(delta > 0)), fixed_static_gate_arms=variants[name][0],
        zeroed_bias_arms=variants[name][1], zeroed_parameter_paths=prepared[name][2])
  result['elapsed_seconds'] = time.time() - start
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
