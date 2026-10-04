"""Read-only BAM checkpoint probe: per-layer token-shared energy of the carried matrix M.

Matches the RMT concentration diagnostic: fixed cohort, positions stride-1::stride, statistic
||mean_t M_l||^2 / mean_t ||M_l||^2 on layer outputs. Restores params only; no updates/saves.
"""
import json
import os
from pathlib import Path
from absl import app
from flax.traverse_util import flatten_dict
from flax.linen import partitioning as nn_partitioning
import jax
import numpy as np
import max_utils
import pyconfig
import train


def main(argv):
  cfg = pyconfig.initialize(argv)
  cfg.get_keys()['load_parameters_path'] = os.environ['DIAG_CHECKPOINT']
  if not cfg.only_eval or cfg.enable_checkpointing or not cfg.base_output_directory.startswith('/tmp/'):
    raise ValueError('Require read-only restore and local /tmp output')
  rng, _, manager, mesh, model, _, _ = train.setup_mesh_and_model(cfg)
  state, _ = max_utils.setup_decode_state(model, cfg, rng, mesh, manager)
  params = state.params
  out = Path(os.environ['DIAG_OUT']); out.mkdir(parents=True, exist_ok=True)
  stride = int(os.environ.get('DIAG_STRIDE', '16'))
  cohort = Path(os.environ['DIAG_COHORT'])
  batches = []
  for i in range(int(os.environ.get('DIAG_SEQS', '8'))):
    d = json.loads((cohort / f'cohort-{i:03d}.json').read_text())
    t = len(d['inputs'])
    b = {k: np.asarray(d[k], np.int32).reshape(1, t) for k in ('inputs', 'targets', 'targets_segmentation')}
    b['inputs_segmentation'] = np.ones((1, t), np.int32)
    b['inputs_position'] = np.arange(t, dtype=np.int32)[None]
    batches.append(b)
  cfg.get_keys()['bam_diag_capture'] = stride
  def capture(p, b):
    (xent, _, _), inter = model.apply(
        p, b['inputs'], b['inputs_position'], decoder_segment_ids=b['inputs_segmentation'],
        decoder_target_mask=b['targets_segmentation'], decoder_target_tokens=b['targets'],
        enable_dropout=False, rngs={'dropout': rng, 'params': rng}, mutable=['intermediates'])
    flat = flatten_dict(inter['intermediates'])
    return jax.numpy.mean(xent), {'/'.join(k): (v[0] if isinstance(v, tuple) else v)
                                  for k, v in flat.items() if k[-1] in ('diag_M_out', 'diag_x_out', 'diag_x_in', 'diag_x_mid')}
  cap = jax.jit(capture)
  per_layer = {}
  losses = []
  block = getattr(cfg, 'bam_local_fetch_block_size', None) or 2
  for b in batches:
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      loss, diag = cap(params, b)
    losses.append(float(loss))
    for key, v in diag.items():
      v = np.asarray(v, np.float32)
      parts = key.split('/')
      kind = parts[-1]
      if 'final_local_layer' in parts:
        per_layer.setdefault((cfg.num_decoder_layers - 1, kind), []).append(v.reshape(-1, *v.shape[-2:]) if kind == 'diag_M_out' else v.reshape(-1, v.shape[-1]))
        continue
      sub = [s for s in parts if s.startswith('local_') or s.startswith('fetch_')]
      offset = int(sub[0].split('_')[1])
      for blk in range(v.shape[0]):
        a = v[blk]
        per_layer.setdefault((block * blk + offset, kind), []).append(a.reshape(-1, *a.shape[-2:]) if kind == 'diag_M_out' else a.reshape(-1, a.shape[-1]))
  res = {'checkpoint': cfg.load_parameters_path, 'exp': cfg.exp_class, 'ce': losses, 'layers': []}
  layers = sorted(set(l for l, _ in per_layer))
  for l in layers:
    row = {'layer': l}
    for kind, label in (('diag_M_out', 'M'), ('diag_x_out', 'x')):
      if (l, kind) not in per_layer:
        continue
      a = np.concatenate(per_layer[(l, kind)], 0).reshape(-1, int(np.prod(per_layer[(l, kind)][0].shape[1:]))).astype(np.float64)
      e = np.mean(np.sum(a * a, 1)); mu = a.mean(0)
      row[f'{label}_shared_mean_fraction'] = float(np.sum(mu * mu) / e)
      row[f'{label}_rms'] = float(np.sqrt(e / a.shape[1]))
      if label == 'M':
        row['M_shape'] = list(per_layer[(l, kind)][0].shape[1:])
    res['layers'].append(row)
  # Token-mean vectors of the residual at attention input, MLP input and final output.
  L = cfg.num_decoder_layers
  def mean_of(l, kind):
    a = np.concatenate(per_layer[(l, kind)], 0).reshape(-1, per_layer[(l, kind)][0].shape[-1])
    return a.mean(0), a
  c_in = np.stack([mean_of(l, 'diag_x_in')[0] for l in range(L)]).astype(np.float32)
  c_mid = np.stack([mean_of(l, 'diag_x_mid')[0] for l in range(L)]).astype(np.float32)
  c_final, x_final = mean_of(L - 1, 'diag_x_out')
  np.save(out / 'means_in.npy', c_in); np.save(out / 'means_mid.npy', c_mid)
  np.save(out / 'means_final.npy', c_final.astype(np.float32))
  for l, row in enumerate(res['layers']):
    for kind, label in (('diag_x_in', 'x_in'), ('diag_x_mid', 'x_mid')):
      mu, a = mean_of(l, kind); a = a.astype(np.float64)
      row[f'{label}_shared_mean_fraction'] = float(np.sum(mu.astype(np.float64) ** 2) / np.mean(np.sum(a * a, 1)))
  pflat = flatten_dict(jax.device_get(params))
  g = np.asarray([v for k, v in pflat.items() if 'decoder_norm' in k and k[-1] == 'scale'][0], np.float64)
  W = np.asarray([v for k, v in pflat.items() if k[-1] == 'logits_dense'][0], np.float64)
  xf = x_final.astype(np.float64)
  h = g * xf / np.sqrt(np.mean(xf * xf, -1, keepdims=True) + 1e-6)
  hm = h.mean(0)
  res['final_hidden_shared_fraction'] = float(np.sum(hm ** 2) / np.mean(np.sum(h * h, 1)))
  counts = np.zeros(cfg.vocab_size)
  for f in sorted(cohort.glob('cohort-*.json')):
    ids = np.asarray(json.loads(f.read_text())['inputs'], np.int64)
    counts += np.bincount(ids, minlength=cfg.vocab_size)[:cfg.vocab_size]
  ml = hm @ W; ml -= ml.mean(); keep = counts >= 3
  rank = lambda a: np.argsort(np.argsort(a)).astype(np.float64)
  res['output_prior'] = {'tokens': int(keep.sum()), 'pearson_logfreq': float(np.corrcoef(ml[keep], np.log(counts[keep]))[0, 1]),
                         'spearman_logfreq': float(np.corrcoef(rank(ml[keep]), rank(np.log(counts[keep])))[0, 1]),
                         'mean_logit_centered_rms': float(np.sqrt(np.mean(ml ** 2)))}
  del per_layer, cap
  jax.clear_caches()
  cfg.get_keys()['bam_diag_capture'] = 0
  def objective(p, batch):
    return train.loss_fn(model, cfg, dict(batch), rng, p, is_train=False)[0]
  def run_ce():
    f = jax.jit(objective); vals = []
    for b in batches:
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        vals.append(float(f(params, b)))
    del f; jax.clear_caches(); return vals
  base = run_ce(); res['baseline_ce_full'] = base
  res['bias_ablations'] = {}
  for target in [t for t in os.environ.get('DIAG_BIAS_TARGETS', 'attn;mlp;final;attn,mlp,final').split(';') if t]:
    cfg.get_keys().update(bam_diag_bias_remove=target, bam_diag_means_in=str(out / 'means_in.npy'),
                          bam_diag_means_mid=str(out / 'means_mid.npy'), bam_diag_means_final=str(out / 'means_final.npy'))
    vals = run_ce()
    res['bias_ablations'][target] = {'losses': vals, 'delta_mean': float(np.mean(np.asarray(vals) - np.asarray(base)))}
    print('BAM_BIAS ' + target + ' ' + str(res['bias_ablations'][target]['delta_mean']), flush=True)
  cfg.get_keys().update(bam_diag_bias_remove='')
  (out / 'results.json').write_text(json.dumps(res, indent=2))
  print('BAM_DIAG_DONE ' + json.dumps({r['layer']: round(r.get('M_shared_mean_fraction', -1), 3) for r in res['layers']}), flush=True)


if __name__ == '__main__':
  app.run(main)
