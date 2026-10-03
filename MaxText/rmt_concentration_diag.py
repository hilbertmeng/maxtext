"""Read-only RMT final-state concentration diagnosis.

Questions: what is the direction that the final normalized matrix concentrates on,
which layers/sublayers build it, how much of the unembedding readout and logits it
carries, and whether removing it changes CE. Restores params only; no updates/saves.
Statistics run on the host of the diagnostic TPU; only compact JSON/npy leave it.
"""
import json
import os
from pathlib import Path
import time
from absl import app
from flax.traverse_util import flatten_dict
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
import max_utils
import pyconfig
import train


def load_batches(cohort, count):
  batches = []
  for i in range(count):
    d = json.loads((cohort / f'cohort-{i:03d}.json').read_text())
    t = len(d['inputs'])
    b = {k: np.asarray(d[k], np.int32).reshape(1, t) for k in ('inputs', 'targets', 'targets_segmentation')}
    b['inputs_segmentation'] = np.ones((1, t), np.int32)
    b['inputs_position'] = np.arange(t, dtype=np.int32)[None]
    batches.append(b)
  return batches


def energy_stats(x):
  """x: (N, D) samples. Shared-mean fraction and uncentered/centered spectra."""
  x = x.astype(np.float64)
  total = np.mean(np.sum(x * x, axis=1))
  mu = x.mean(0)
  shared = float(np.sum(mu * mu) / total)
  _, s, vt = np.linalg.svd(x, full_matrices=False)
  e = s ** 2 / np.sum(s ** 2)
  xc = x - mu
  _, sc, vtc = np.linalg.svd(xc, full_matrices=False)
  ec = sc ** 2 / np.sum(sc ** 2)
  mu_hat = mu / max(np.linalg.norm(mu), 1e-30)
  return {
      'mean_sq_norm': float(total), 'shared_mean_fraction': shared,
      'uncentered_top': e[:8].tolist(), 'uncentered_participation': float(1 / np.sum(e ** 2)),
      'centered_top': ec[:8].tolist(), 'centered_participation': float(1 / np.sum(ec ** 2)),
      'cos_top1_mean': float(abs(vt[0] @ mu_hat)),
  }, mu_hat, vt[0], vtc[0]


def main(argv):
  cfg = pyconfig.initialize(argv)
  cfg.get_keys()['load_parameters_path'] = os.environ['DIAG_CHECKPOINT']
  if not cfg.only_eval or cfg.enable_checkpointing or not cfg.base_output_directory.startswith('/tmp/'):
    raise ValueError('Require read-only restore and local /tmp output')
  rng, _, manager, mesh, model, _, _ = train.setup_mesh_and_model(cfg)
  state, _ = max_utils.setup_decode_state(model, cfg, rng, mesh, manager)
  params = state.params
  out = Path(os.environ['DIAG_OUT']); out.mkdir(parents=True, exist_ok=True)
  batches = load_batches(Path(os.environ['DIAG_COHORT']), int(os.environ.get('DIAG_SEQS', '8')))
  stride = int(os.environ.get('DIAG_STRIDE', '16'))
  heads, key_dim, value_dim = cfg.num_query_heads, cfg.rmt_reskey_dim, cfg.head_dim
  meta = {'exp': cfg.exp_class, 'checkpoint': cfg.load_parameters_path, 'sequences': len(batches),
          'stride': stride, 'heads': heads, 'key_dim': key_dim, 'value_dim': value_dim,
          'layers': cfg.num_decoder_layers, 'commit': os.environ.get('DIAG_COMMIT')}
  (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
  print('DIAG_READY ' + json.dumps(meta), flush=True)

  def objective(p, batch):
    return train.loss_fn(model, cfg, dict(batch), rng, p, is_train=False)[0]

  def run_ce(tag):
    f = jax.jit(objective)
    losses = []
    for b in batches:
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        losses.append(float(f(params, b)))
    del f
    jax.clear_caches()
    print(f'DIAG_CE {tag} ' + json.dumps(losses), flush=True)
    return losses

  results = {'baseline_ce': run_ce('baseline')}

  # Capture pass.
  cfg.get_keys()['rmt_diag_capture'] = stride
  def capture(p, b):
    (xent, _, _), inter = model.apply(
        p, b['inputs'], b['inputs_position'], decoder_segment_ids=b['inputs_segmentation'],
        decoder_target_mask=b['targets_segmentation'], decoder_target_tokens=b['targets'],
        enable_dropout=False, rngs={'dropout': rng, 'params': rng}, mutable=['intermediates'])
    flat = flatten_dict(inter['intermediates'])
    diag = {'/'.join(k): v[0] if isinstance(v, tuple) else v for k, v in flat.items() if any('diag_' in s for s in k)}
    return xent[:, stride - 1::stride], diag
  cap = jax.jit(capture)
  store = {}
  xents = []
  started = time.monotonic()
  for i, b in enumerate(batches):
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      xe, diag = cap(params, b)
    xents.append(np.asarray(xe, np.float32).reshape(-1))
    for k, v in diag.items():
      store.setdefault(k.split('/')[-1], []).append(np.asarray(v, np.float32))
    print(f'DIAG_CAPTURE {i} {time.monotonic()-started:.1f}s', flush=True)
  del cap
  jax.clear_caches()
  cfg.get_keys()['rmt_diag_capture'] = 0
  print('DIAG_KEYS ' + json.dumps({k: list(v[0].shape) for k, v in store.items()}), flush=True)

  # Assemble (N, ...) arrays; layer arrays come stacked (L, B, P, K, V).
  def tokens(name):
    return np.concatenate([a.reshape((-1,) + a.shape[-2:]) if a.ndim >= 4 else a.reshape(-1, a.shape[-1])
                           for a in store[name]], 0)
  def layer_tokens(name):
    arr = np.concatenate([a.reshape(a.shape[0], -1, *a.shape[-2:]) for a in store[name]], 1)
    return arr  # (L, N, K, V)
  F = tokens('diag_final_normed')
  R = tokens('diag_final_raw')
  N = F.shape[0]
  positions = np.tile(np.arange(stride - 1, cfg.max_target_length, stride), len(batches))
  flat = lambda a: a.reshape(a.shape[0], -1)

  stats_f, mu_hat, u1, uc1 = energy_stats(flat(F))
  stats_r, mu_raw_hat, _, _ = energy_stats(flat(R))
  results['final_normed'] = stats_f
  results['final_raw'] = stats_r
  results['raw_rms_mean'] = float(np.mean(np.sqrt(np.mean(flat(R) ** 2, 1))))
  mu = flat(F).mean(0).reshape(key_dim, value_dim)
  row_energy = np.sum(mu ** 2, 1)
  results['shared_mean_row_energy'] = {'first': float(row_energy[:heads].sum() / row_energy.sum()),
                                       'tail': float(row_energy[heads:].sum() / row_energy.sum()),
                                       'per_row': (row_energy / row_energy.sum()).tolist()}
  # Per-token coefficient on the shared direction, by position.
  coeff = flat(F) @ mu_hat
  frac_tok = coeff ** 2 / np.sum(flat(F) ** 2, 1)
  bins = [(0, 64), (64, 256), (256, 1024), (1024, 4096)]
  results['shared_direction_by_position'] = {
      f'{a}-{b}': {'coef_mean': float(coeff[(positions >= a) & (positions < b)].mean()),
                   'coef_std': float(coeff[(positions >= a) & (positions < b)].std()),
                   'energy_fraction': float(frac_tok[(positions >= a) & (positions < b)].mean())}
      for a, b in bins if np.any((positions >= a) & (positions < b))}
  results['shared_direction_token_cv'] = float(coeff.std() / max(abs(coeff.mean()), 1e-30))

  # Readout and logits attribution.
  Hs = tokens('diag_hidden_static').reshape(N, -1)
  Hd = tokens('diag_hidden_dynamic').reshape(N, -1) if 'diag_hidden_dynamic' in store else np.zeros_like(Hs)
  H = Hs + Hd
  pflat = flatten_dict(jax.device_get(params))
  read_key = np.asarray([v for k, v in pflat.items() if k[-1] == 'final_read_key'][0], np.float32)
  W = np.asarray([v for k, v in pflat.items() if k[-1] == 'logits_dense'][0], np.float32)
  Fm = F.astype(np.float64)
  shared_part = np.einsum('kv,kn->nv', mu, read_key).reshape(-1)
  def frac(part, whole):
    return float(np.sum(part ** 2) / np.mean(np.sum(whole ** 2, 1)))
  results['readout'] = {
      'static_rms': float(np.sqrt(np.mean(Hs ** 2))), 'dynamic_rms': float(np.sqrt(np.mean(Hd ** 2))),
      'static_shared_mean_fraction': frac(Hs.mean(0), Hs),
      'static_from_final_mean_fraction': frac(shared_part, Hs),
      'dynamic_shared_mean_fraction': frac(Hd.mean(0), Hd),
      'total_shared_mean_fraction': frac(H.mean(0), H),
      'static_dynamic_cos_of_means': float(Hs.mean(0) @ Hd.mean(0) / max(np.linalg.norm(Hs.mean(0)) * np.linalg.norm(Hd.mean(0)), 1e-30)),
  }
  sel = np.arange(0, N, max(1, N // 512))
  logits = H[sel] @ W
  common = logits.mean(1)
  centered = logits - common[:, None]
  logit_mean_h = H.mean(0) @ W
  var_logits = H[sel] - H.mean(0)
  centered_var_part = (var_logits @ W)
  centered_var_part -= centered_var_part.mean(1, keepdims=True)
  results['logits'] = {
      'common_offset_mean': float(common.mean()), 'common_offset_std': float(common.std()),
      'centered_rms': float(np.sqrt(np.mean(centered ** 2))),
      'mean_hidden_logit_common': float(logit_mean_h.mean()),
      'mean_hidden_logit_centered_rms': float(np.sqrt(np.mean((logit_mean_h - logit_mean_h.mean()) ** 2))),
      'token_varying_centered_rms': float(np.sqrt(np.mean(centered_var_part ** 2))),
  }
  results['captured_xent_mean'] = float(np.mean(np.concatenate(xents)))

  # Layer build-up along the final shared direction, expressed in raw coordinates.
  gain = np.asarray([v for k, v in pflat.items() if k[-2:] == ('final_matrix_norm', 'scale')][0], np.float32).reshape(-1)
  d_raw = mu_hat / np.where(np.abs(gain) > 1e-6, gain, 1e-6)
  d_raw /= np.linalg.norm(d_raw)
  d_mu_raw = mu_raw_hat
  layers = []
  if 'diag_M_out' in store:
    Mo = layer_tokens('diag_M_out'); Ma = layer_tokens('diag_M_attn')
    prev = None
    for l in range(Mo.shape[0]):
      mo, ma = flat(Mo[l]).astype(np.float64), flat(Ma[l]).astype(np.float64)
      e_mo = np.mean(np.sum(mo ** 2, 1))
      row = {'layer': l, 'rms': float(np.sqrt(e_mo / mo.shape[1])),
             'shared_mean_fraction': float(np.sum(mo.mean(0) ** 2) / e_mo),
             'dir_energy_fraction': float(np.mean((mo @ d_raw) ** 2) / e_mo),
             'dir_coef_mean': float(np.mean(mo @ d_raw)),
             'rawmean_dir_coef_mean': float(np.mean(mo @ d_mu_raw))}
      if prev is not None:
        da, dm = ma - prev, mo - ma
        row.update(attn_write_dir_mean=float(np.mean(da @ d_raw)), mlp_write_dir_mean=float(np.mean(dm @ d_raw)),
                   attn_write_rms=float(np.sqrt(np.mean(da ** 2))), mlp_write_rms=float(np.sqrt(np.mean(dm ** 2))),
                   attn_write_shared_fraction=float(np.sum(da.mean(0) ** 2) / np.mean(np.sum(da ** 2, 1))),
                   mlp_write_shared_fraction=float(np.sum(dm.mean(0) ** 2) / np.mean(np.sum(dm ** 2, 1))))
      else:
        row.update(mlp_write_dir_mean=float(np.mean((mo - ma) @ d_raw)),
                   mlp_write_rms=float(np.sqrt(np.mean((mo - ma) ** 2))))
      layers.append(row)
      prev = mo
  results['layers'] = layers
  Mo = layer_tokens('diag_M_out') if 'diag_M_out' in store else None
  # Per-head attention write contributions along the final direction (raw coordinates).
  if 'diag_head_out' in store and 'diag_attn_address' in store:
    Y = layer_tokens('diag_head_out')          # (L, N, heads, value)
    A = layer_tokens('diag_attn_address')      # (L, N, heads, key)
    G = np.concatenate([a.reshape(a.shape[0], -1, a.shape[-1]) for a in store['diag_attn_gate']], 1)  # (L, N, heads)
    U = d_raw.reshape(key_dim, value_dim)
    static_keys = np.asarray([v for k, v in pflat.items() if k[-1] == 'attn_write_key'][0], np.float32)  # (heads, L, key) scanned
    static_keys = np.moveaxis(static_keys, cfg.param_scan_axis, 0) if static_keys.ndim == 3 else static_keys
    content_norm = bool(cfg.get_keys().get('rmt_static_write_content_norm', False))
    heads_out = []
    for l in range(Y.shape[0]):
      y = Y[l].astype(np.float64)
      yrms = np.sqrt(np.mean(y ** 2, -1))                     # (N, heads)
      yn = y / np.maximum(yrms[..., None], 1e-6)
      a = A[l].astype(np.float64)
      a = a / np.maximum(np.sqrt(np.mean(a ** 2, -1, keepdims=True)), 1e-6)
      dyn = G[l] * np.einsum('nhk,kv,nhv->nh', a, U, yn)      # per-token per-head projection on U
      sk = static_keys[l].astype(np.float64)                 # (heads, key)
      ys = yn if content_norm else y
      sta = np.einsum('hk,kv,nhv->nh', sk, U, ys)
      ymean = y.mean(0)
      for h in range(y.shape[1]):
        heads_out.append({'layer': l, 'head': h, 'y_rms': float(yrms[:, h].mean()),
          'y_shared_fraction': float(np.sum(ymean[h] ** 2) / np.mean(np.sum(y[:, h] ** 2, -1))),
          'gate_mean': float(G[l][:, h].mean()), 'dyn_dir_mean': float(dyn[:, h].mean()),
          'static_dir_mean': float(sta[:, h].mean()), 'dyn_dir_std': float(dyn[:, h].std())})
    results['attention_heads'] = heads_out
  del store
  directions = {'shared_mean': mu_hat, 'top1': u1, 'centered_top1': uc1}
  for name, v in directions.items():
    np.save(out / f'direction_{name}.npy', v.reshape(1, key_dim, value_dim).astype(np.float32))
  (out / 'results.json').write_text(json.dumps(results, indent=2))
  print('DIAG_STATS_DONE', flush=True)

  # Functional ablations on the same cohort.
  ablations = {}
  for name in os.environ.get('DIAG_ABLATE', 'shared_mean,top1').split(','):
    if not name:
      continue
    for mode in ('pre', 'post'):
      path = out / f'direction_{name}.npy'
      if mode == 'pre' and name != 'shared_mean_raw':
        # Pre-norm removal uses the corresponding raw-coordinate direction.
        v = directions[name].reshape(-1) / np.where(np.abs(gain) > 1e-6, gain, 1e-6)
        v /= np.linalg.norm(v)
        path = out / f'direction_{name}_raw.npy'
        np.save(path, v.reshape(1, key_dim, value_dim).astype(np.float32))
      cfg.get_keys().update(rmt_diag_project_file=str(path), rmt_diag_project_mode=mode)
      losses = run_ce(f'{name}-{mode}')
      ablations[f'{name}-{mode}'] = {'losses': losses,
          'delta_mean': float(np.mean(np.asarray(losses) - np.asarray(results['baseline_ce'])))}
  cfg.get_keys().update(rmt_diag_project_file='', rmt_diag_project_mode='')
  results['ablations'] = ablations
  # Stage 2: keep only the cohort-mean coefficient along the concentrated direction.
  if os.environ.get('DIAG_FIX', '0') == '1':
    u = directions['top1'].reshape(-1)
    u_raw = u / np.where(np.abs(gain) > 1e-6, gain, 1e-6); u_raw /= np.linalg.norm(u_raw)
    sign = np.sign(np.mean(flat(F) @ u)) or 1.
    u, u_raw = u * sign, u_raw * sign
    np.save(out / 'fix_u.npy', u.reshape(1, key_dim, value_dim).astype(np.float32))
    np.save(out / 'fix_u_raw.npy', u_raw.reshape(1, key_dim, value_dim).astype(np.float32))
    c_post, c_pre = float(np.mean(flat(F) @ u)), float(np.mean(flat(R) @ u_raw))
    # Layer-input means: input of layer l is the output of layer l-1 (layer 0 untouched).
    layer_means = np.zeros(cfg.num_decoder_layers, np.float32)
    layer_stats = []
    for l in range(1, cfg.num_decoder_layers):
      c = flat(Mo[l - 1]).astype(np.float64) @ u_raw
      e = np.mean(np.sum(flat(Mo[l - 1]).astype(np.float64) ** 2, 1))
      layer_means[l] = c.mean()
      layer_stats.append({'layer': l, 'coef_mean': float(c.mean()), 'coef_std': float(c.std()),
                          'energy_fraction': float(np.mean(c ** 2) / e),
                          'variation_energy_fraction': float(np.var(c) / e)})
    np.save(out / 'fix_layer_means.npy', layer_means)
    fixes = {'coef_post': c_post, 'coef_pre': c_pre, 'layer_inputs': layer_stats}
    variants = [('final-fixpost', dict(rmt_diag_project_file=str(out / 'fix_u.npy'), rmt_diag_project_mode='fixpost', rmt_diag_project_target=c_post)),
                ('final-fixpre', dict(rmt_diag_project_file=str(out / 'fix_u_raw.npy'), rmt_diag_project_mode='fixpre', rmt_diag_project_target=c_pre)),
                ('layers-fix', dict(rmt_diag_layer_mode='fix', rmt_diag_layer_direction=str(out / 'fix_u_raw.npy'), rmt_diag_layer_means=str(out / 'fix_layer_means.npy'))),
                ('layers-remove', dict(rmt_diag_layer_mode='remove', rmt_diag_layer_direction=str(out / 'fix_u_raw.npy'), rmt_diag_layer_means=str(out / 'fix_layer_means.npy'))),
                ('layers-fix+final-fixpre', dict(rmt_diag_layer_mode='fix', rmt_diag_layer_direction=str(out / 'fix_u_raw.npy'), rmt_diag_layer_means=str(out / 'fix_layer_means.npy'),
                                                 rmt_diag_project_file=str(out / 'fix_u_raw.npy'), rmt_diag_project_mode='fixpre', rmt_diag_project_target=c_pre))]
    for name, upd in variants:
      cfg.get_keys().update(upd)
      losses = run_ce(name)
      fixes[name] = {'losses': losses, 'delta_mean': float(np.mean(np.asarray(losses) - np.asarray(results['baseline_ce'])))}
      cfg.get_keys().update(rmt_diag_project_file='', rmt_diag_project_mode='', rmt_diag_layer_mode='')
    results['mean_fix'] = fixes
    print('DIAG_FIX ' + json.dumps({k: v['delta_mean'] for k, v in fixes.items() if isinstance(v, dict)}), flush=True)
  (out / 'results.json').write_text(json.dumps(results, indent=2))
  print('DIAG_COMPLETE ' + json.dumps({k: v['delta_mean'] for k, v in ablations.items()}), flush=True)


if __name__ == '__main__':
  app.run(main)
