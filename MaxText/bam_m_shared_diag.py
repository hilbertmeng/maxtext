"""Read-only BAM checkpoint probe: per-layer token-shared energy of the carried matrix M.

Matches the RMT concentration diagnostic: fixed cohort, positions stride-1::stride, statistic
||mean_t M_l||^2 / mean_t ||M_l||^2 on layer outputs. Restores params only; no updates/saves.
"""
import json
import os
from pathlib import Path
from absl import app
from flax.traverse_util import flatten_dict, unflatten_dict
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


def route_diagnostics(cfg, model, params, rng, mesh, batches, out):
  """Per-(token, head-chunk) importance of the write-layer MLP output for residual vs M."""
  from layers import bam_route_probe as rp
  from flax import core
  every = int(cfg.bam_mlp_write_every); offset = int(cfg.bam_mlp_write_offset)
  write_layers = [l for l in range(cfg.num_decoder_layers) if (l + 1) % every == offset % every]
  def loss_of(p, b):
    return train.loss_fn(model, cfg, dict(b), rng, p, is_train=False)[0]
  def set_mode(mode):
    cfg.get_keys()['bam_diag_route'] = mode
    jax.clear_caches()
  # 1. Gradient split: first-order effects for both consumer paths.
  set_mode('grad')
  flat = flatten_dict(params)
  emb_key = [k for k in flat if 'token_embedder' in k][0]
  def emb_loss(emb, p, b):
    f = dict(flat); f[emb_key] = emb
    tree = unflatten_dict(f)
    tree = core.freeze(tree) if isinstance(p, core.FrozenDict) else tree
    return loss_of(tree, b)
  grad_fn = jax.jit(jax.grad(emb_loss))
  effects = {}
  for i, b in enumerate(batches):
    rp.STORE.clear()
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      g = grad_fn(flat[emb_key], params, b)
    jax.block_until_ready(g); jax.effects_barrier()
    for (l, path), arrs in rp.STORE.items():
      effects.setdefault((l, path), []).append(arrs[0].reshape(arrs[0].shape[-2], arrs[0].shape[-1]))
    print(f'ROUTE_GRAD {i} keys={len(rp.STORE)}', flush=True)
  del grad_fn
  layers = sorted(set(l for l, _ in effects))
  E = {(l, p): np.stack(effects[(l, p)]) for (l, p) in effects}   # (S, T, H) signed effects
  # 2. Gates at full resolution.
  set_mode('gate')
  def gates_of(p, b):
    _, inter = model.apply(p, b['inputs'], b['inputs_position'], decoder_segment_ids=b['inputs_segmentation'],
                           decoder_target_mask=b['targets_segmentation'], decoder_target_tokens=b['targets'],
                           enable_dropout=False, rngs={'dropout': rng, 'params': rng}, mutable=['intermediates'])
    f = flatten_dict(inter['intermediates'])
    return {'/'.join(k): (v[0] if isinstance(v, tuple) else v) for k, v in f.items() if k[-1] == 'diag_route_gate'}
  gfn = jax.jit(gates_of)
  block = getattr(cfg, 'bam_local_fetch_block_size', None) or 2
  G = {}
  for b in batches:
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      d = gfn(params, b)
    for key, v in d.items():
      v = np.asarray(v, np.float32)
      sub = [x for x in key.split('/') if x.startswith('local_') or x.startswith('fetch_')]
      off = int(sub[0].split('_')[1])
      for blk in range(v.shape[0]):
        G.setdefault(block * blk + off, []).append(v[blk].reshape(v.shape[-2], v.shape[-1]))
  del gfn
  G = {l: np.stack(v) for l, v in G.items()}
  # Statistics.
  rank = lambda a: np.argsort(np.argsort(a.ravel())).astype(np.float64)
  def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])
  res = {'write_layers': write_layers, 'layers': []}
  masks_oracle, masks_random = {}, {}
  rng_np = np.random.default_rng(0)
  for l in layers:
    er, em = E[(l, 'res')], E[(l, 'm')]
    ir, im = np.abs(er), np.abs(em)
    tot = ir + im
    share_m = im / np.maximum(tot, 1e-30)
    hi_r, hi_m = ir > np.quantile(ir, .75), im > np.quantile(im, .75)
    oracle_m = (im > ir).astype(np.float32)
    frac_m = float(oracle_m.mean())
    random_m = (rng_np.random(oracle_m.shape) < frac_m).astype(np.float32)
    masks_oracle[l], masks_random[l] = oracle_m, random_m
    # First-order predicted loss change of removing a path: -effect.
    pred_oracle = float(np.mean(np.sum(-np.where(oracle_m > 0, er, em), axis=(1, 2))))
    pred_random = float(np.mean(np.sum(-np.where(random_m > 0, er, em), axis=(1, 2))))
    pred_res_off = float(np.mean(np.sum(-er, axis=(1, 2)))); pred_m_off = float(np.mean(np.sum(-em, axis=(1, 2))))
    row = {'layer': l, 'importance_res_mean': float(ir.mean()), 'importance_m_mean': float(im.mean()),
           'spearman_res_m': spear(ir, im), 'pearson_log': float(np.corrcoef(np.log(ir.ravel() + 1e-12), np.log(im.ravel() + 1e-12))[0, 1]),
           'share_m_percentiles': np.percentile(share_m, [10, 25, 50, 75, 90]).tolist(),
           'quadrant_m_only': float(np.mean(hi_m & ~hi_r)), 'quadrant_res_only': float(np.mean(hi_r & ~hi_m)),
           'quadrant_both': float(np.mean(hi_r & hi_m)), 'oracle_frac_to_m': frac_m,
           'pred_dce_res_off': pred_res_off, 'pred_dce_m_off': pred_m_off,
           'pred_dce_oracle': pred_oracle, 'pred_dce_random': pred_random,
           'share_m_by_head': np.median(share_m, axis=(0, 1)).tolist()}
    if l in G:
      g = G[l]
      row.update(gate_mean=float(g.mean()), spearman_gate_im=spear(g, im), spearman_gate_ir=spear(g, ir),
                 spearman_gate_share_m=spear(g, share_m))
    res['layers'].append(row)
  res['pred_total'] = {k: float(sum(r[k] for r in res['layers'])) for k in
                       ('pred_dce_res_off', 'pred_dce_m_off', 'pred_dce_oracle', 'pred_dce_random')}
  (out / 'route_results.json').write_text(json.dumps(res, indent=2))
  print('ROUTE_STATS ' + json.dumps(res['pred_total']), flush=True)
  # 3. Forward ablations.
  def run_ce(mode, masks=None):
    set_mode(mode)
    f = jax.jit(loss_of); vals = []
    for i, b in enumerate(batches):
      if masks is not None:
        rp.MASK.clear()
        for l, m in masks.items():
          rp.MASK[l] = m[i][None]
      with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        vals.append(float(f(params, b)))
    del f; return vals
  ce = {'baseline': run_ce('')}
  for mode in ('res_off', 'm_off', 'both_off', 'couple'):
    ce[mode] = run_ce(mode)
  ce['oracle'] = run_ce('mask', masks_oracle)
  ce['random'] = run_ce('mask', masks_random)
  base = np.asarray(ce['baseline'])
  res['ablations'] = {k: {'losses': v, 'delta_mean': float(np.mean(np.asarray(v) - base))} for k, v in ce.items()}
  set_mode('')
  (out / 'route_results.json').write_text(json.dumps(res, indent=2))
  print('ROUTE_ABLATIONS ' + json.dumps({k: round(v['delta_mean'], 4) for k, v in res['ablations'].items()}), flush=True)


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
  if os.environ.get('DIAG_ROUTE', '0') == '1':
    route_diagnostics(cfg, model, params, rng, mesh, batches, out)
    return
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
