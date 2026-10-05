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


def address_analysis(cfg, per_layer, params):
  """Address-axis overlap between attention and MLP writes, and reader sensitivity."""
  L = cfg.num_decoder_layers
  cat = lambda l, k: np.concatenate(per_layer[(l, k)], 0).astype(np.float64)
  A = {l: cat(l, 'diag_attn_addr') for l in range(L) if (l, 'diag_attn_addr') in per_layer}
  Ga = {l: cat(l, 'diag_attn_gate') for l in A}
  Mw = {l: cat(l, 'diag_mlp_addr') for l in range(L) if (l, 'diag_mlp_addr') in per_layer}
  Gm = {l: cat(l, 'diag_mlp_gate') for l in Mw}
  V = next(iter(A.values())).shape[-1]
  # Per-token, per-layer write second moments in address space: sum_h g^2 a a^T.
  Catt = {l: np.einsum('nh,nhv,nhw->nvw', Ga[l] ** 2, A[l], A[l]) for l in A}
  Cmlp = {l: np.einsum('nh,nhv,nhw->nvw', Gm[l] ** 2, Mw[l], Mw[l]) for l in Mw}
  def unit(C):
    return C / np.maximum(np.trace(C, axis1=-2, axis2=-1)[..., None, None], 1e-30)
  Ca_all = unit(sum(Catt.values())); Cm_all = unit(sum(Cmlp.values()))
  out = {'address_dim': V, 'attn_layers': sorted(A), 'mlp_layers': sorted(Mw),
         'write_energy_attn_total': float(np.mean(np.trace(sum(Catt.values()), axis1=-2, axis2=-1))),
         'write_energy_mlp_total': float(np.mean(np.trace(sum(Cmlp.values()), axis1=-2, axis2=-1)))}
  # Measure 1: same token, all layers.
  rho = V * np.einsum('nvw,nwv->n', Ca_all, Cm_all)
  evals, evecs = np.linalg.eigh(Ca_all)
  evals, evecs = evals[:, ::-1], evecs[:, :, ::-1]
  m1 = {'rho_mean': float(rho.mean()), 'rho_percentiles': np.percentile(rho, [10, 50, 90]).tolist()}
  for k in (5, 10, 20):
    U = evecs[:, :, :k]
    share = np.einsum('nvk,nvw,nwk->n', U, Cm_all, U)
    m1[f'mlp_share_in_attn_top{k}'] = float(share.mean())
    m1[f'attn_self_share_top{k}'] = float(evals[:, :k].sum(1).mean())
    m1[f'isotropic_top{k}'] = k / V
  # Self-overlap baseline: overlap of attention with itself and of an isotropic write.
  m1['rho_attn_self_mean'] = float((V * np.einsum('nvw,nwv->n', Ca_all, Ca_all)).mean())
  m1['rho_mlp_self_mean'] = float((V * np.einsum('nvw,nwv->n', Cm_all, Cm_all)).mean())
  out['measure1_token_all_layers'] = m1
  # Measure 2: statistical subspaces across tokens.
  ga, gm = Ca_all.mean(0), Cm_all.mean(0)
  ea, ua = np.linalg.eigh(ga); ea, ua = ea[::-1], ua[:, ::-1]
  em, um = np.linalg.eigh(gm); em, um = em[::-1], um[:, ::-1]
  m2 = {'rho_global': float(V * np.trace(ga @ gm)), 'attn_eig_top': (ea[:10] / ea.sum()).tolist(),
        'mlp_eig_top': (em[:10] / em.sum()).tolist()}
  for k in (5, 10, 20):
    s = np.linalg.svd(ua[:, :k].T @ um[:, :k], compute_uv=False)
    m2[f'principal_cos_top{k}'] = s.tolist()
    m2[f'mlp_global_share_in_attn_top{k}'] = float(np.trace(ua[:, :k].T @ gm @ ua[:, :k]))
  out['measure2_global'] = m2
  # Measure 3: readers. Dynamic local reads see span(P_l); static V/O keys read full address.
  pflat = flatten_dict(jax.device_get(params))
  def layer_param(name, l):
    block = getattr(cfg, 'bam_local_fetch_block_size', None) or 2
    for k, v in pflat.items():
      if k[-1] != name:
        continue
      v = np.asarray(v, np.float64)
      if 'final_local_layer' in k:
        if l == L - 1:
          return v
        continue
      sub = [p for p in k if p.startswith('local_') or p.startswith('fetch_')]
      if not sub or l == L - 1:
        continue
      if int(sub[0].split('_')[1]) != l % block:
        continue
      return np.take(v, l // block, axis=cfg.param_scan_axis)
    return None
  rows = []
  for l in range(1, L):
    ca = unit(sum(Catt[j] for j in Catt if j < l)).mean(0)
    mlp_prev = [j for j in Cmlp if j < l]
    cm = unit(sum(Cmlp[j] for j in mlp_prev)).mean(0) if mlp_prev else None
    row = {'layer': l}
    P = layer_param('abs_v_cache_projection', l)
    if P is not None:
      Q, _ = np.linalg.qr(P)
      row['compressed_dim'] = int(P.shape[-1])
      row['attn_energy_in_compressed'] = float(np.trace(Q.T @ ca @ Q))
      if cm is not None:
        row['mlp_energy_in_compressed'] = float(np.trace(Q.T @ cm @ Q))
    for arm in ('v', 'o'):
      key = layer_param(f'static_{arm}_key', l)
      if key is None or cm is None:
        continue
      kk = key / np.maximum(np.linalg.norm(key, axis=0, keepdims=True), 1e-30)   # (V, heads)
      sa = V * np.einsum('vh,vw,wh->h', kk, ca, kk)
      sm = V * np.einsum('vh,vw,wh->h', kk, cm, kk)
      row[f'static_{arm}_attn_sensitivity_median'] = float(np.median(sa))
      row[f'static_{arm}_mlp_sensitivity_median'] = float(np.median(sm))
      row[f'static_{arm}_mlp_over_attn_log2_median'] = float(np.median(np.log2(sm / sa)))
    rows.append(row)
  out['measure3_readers'] = rows
  return out


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
                                  for k, v in flat.items() if k[-1] in ('diag_M_out', 'diag_x_out', 'diag_x_in', 'diag_x_mid', 'diag_attn_addr', 'diag_attn_gate', 'diag_mlp_addr', 'diag_mlp_gate')}
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
        per_layer.setdefault((cfg.num_decoder_layers - 1, kind), []).append(v.reshape((-1,) + v.shape[2:]))
        continue
      sub = [s for s in parts if s.startswith('local_') or s.startswith('fetch_')]
      offset = int(sub[0].split('_')[1])
      for blk in range(v.shape[0]):
        a = v[blk]
        per_layer.setdefault((block * blk + offset, kind), []).append(a.reshape((-1,) + a.shape[2:]))
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
  if os.environ.get('DIAG_ADDR', '0') == '1':
    res['address'] = address_analysis(cfg, per_layer, params)
    (out / 'results.json').write_text(json.dumps(res, indent=2))
    print('BAM_ADDR ' + json.dumps({k: v for k, v in res['address'].items() if not isinstance(v, (list, dict))}), flush=True)
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
