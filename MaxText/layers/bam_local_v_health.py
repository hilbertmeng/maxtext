"""Compact, per-layer summaries for independent + shared LocalV reads.

Side order is row/address, col/data. Read vectors are measured after each
branch's own gate, in matching injected coordinates, before their addition.
"""
import re
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict


GATE_NAMES = ('mean', 'std', 'frac_lt_005', 'frac_gt_095',
              *(f'bin_{lo:02d}_{lo + 20:02d}' for lo in range(0, 100, 20)))
READ_NAMES = ('independent_rms', 'shared_rms', 'std_rms', 'sum_rms',
              'independent_over_std', 'shared_over_std', 'sum_over_std',
              'cosine', 'pearson', 'interference_fraction',
              'gate_ind_shared_corr', 'gate_ind_o_corr', 'gate_shared_o_corr',
              'both_gate_gt_005', 'both_gate_gt_020', 'both_gate_gt_080')


def _corr(a, b):
  a, b = a - jnp.mean(a), b - jnp.mean(b)
  return jnp.mean(a * b) / jnp.maximum(jnp.sqrt(jnp.mean(a * a) * jnp.mean(b * b)), 1e-12)


def local_v_dual_stats(independent, shared, std, gates, k, c):
  independent, shared, std = (jax.lax.stop_gradient(x).astype(jnp.float32)
                              for x in (independent, shared, std))
  gates = tuple(jax.lax.stop_gradient(g).astype(jnp.float32) for g in gates)
  gate_stats, read_stats, bin_stats = [], [], []
  for side, sl in enumerate((slice(k, k + c), slice(0, k))):
    a, b, v = independent[..., sl], shared[..., sl], std[..., sl]
    ga, gb, go = (g[..., side] for g in gates)
    per_branch = []
    for g in (ga, gb, go):
      masks = [(g >= lo / 5) & ((g < (lo + 1) / 5) if lo < 4 else (g <= 1))
               for lo in range(5)]
      per_branch.append(jnp.stack((jnp.mean(g), jnp.std(g), jnp.mean(g < .05),
                                   jnp.mean(g > .95), *(jnp.mean(m) for m in masks))))
    gate_stats.append(jnp.stack(per_branch))
    aa, bb, vv, ab = (jnp.mean(x) for x in (a*a, b*b, v*v, a*b))
    ar, br, vr, sr = (jnp.sqrt(jnp.maximum(x, 0)) for x in (aa, bb, vv, aa+bb+2*ab))
    read_stats.append(jnp.stack((ar, br, vr, sr, ar/jnp.maximum(vr, 1e-12),
        br/jnp.maximum(vr, 1e-12), sr/jnp.maximum(vr, 1e-12),
        ab/jnp.maximum(ar*br, 1e-12), _corr(a,b), 2*ab/jnp.maximum(aa+bb, 1e-12),
        _corr(ga,gb), _corr(ga,go), _corr(gb,go),
        *(jnp.mean((ga > p) & (gb > p)) for p in (.05, .2, .8)))))
    branches = []
    for x, g in ((a,ga), (b,gb)):
      energy = jnp.mean(x*x, axis=-1)
      std_energy = jnp.mean(v*v, axis=-1)
      rows = []
      for lo in range(5):
        mask = (g >= lo/5) & ((g < (lo+1)/5) if lo < 4 else (g <= 1))
        e = jnp.sum(jnp.where(mask, energy, 0))
        ve = jnp.sum(jnp.where(mask, std_energy, 0))
        rows.append(jnp.stack((jnp.sqrt(e/jnp.maximum(ve, 1e-12)),
                               e/jnp.maximum(jnp.sum(energy), 1e-12))))
      branches.append(jnp.stack(rows))
    bin_stats.append(jnp.stack(branches))
  return dict(local_v_dual_gate_stats=jnp.stack(gate_stats),
              local_v_dual_read_stats=jnp.stack(read_stats),
              local_v_dual_bin_stats=jnp.stack(bin_stats))


def record_local_v_dual_metrics(metrics, intermediates, config):
  """Map both block-scan and non-scan captures to true layer indices."""
  flat = flatten_dict(intermediates['intermediates']['decoder'])
  for path, values in flat.items():
    if not path[-1].startswith('local_v_dual_'):
      continue
    data = values[0]
    if config.scan_layers:
      offset = next(int(p.split('_')[-1]) for p in path if re.fullmatch(r'local_\d+', p))
      layers = ((i * config.bam_local_fetch_block_size + offset, data[i])
                for i in range(data.shape[0]))
    else:
      layer = next(int(p.split('_')[-1]) for p in path if re.fullmatch(r'layers_\d+', p))
      layers = ((layer, data),)
    for layer, stats in layers:
      for side_id, side in enumerate(('row', 'col')):
        prefix = f'bam/local_v_dual/{side}/layer_{layer:03d}'
        if path[-1] == 'local_v_dual_read_stats':
          entries = [(name, stats[side_id, i]) for i, name in enumerate(READ_NAMES)]
        elif path[-1] == 'local_v_dual_gate_stats':
          entries = [(f'{branch}_gate/{name}', stats[side_id, j, i])
                     for j, branch in enumerate(('independent', 'shared', 'local_o'))
                     for i, name in enumerate(GATE_NAMES)]
        else:
          entries = [(f'{branch}_gate/bin_{20*i:02d}_{20*i+20:02d}/{name}', stats[side_id,j,i,s])
                     for j, branch in enumerate(('independent', 'shared')) for i in range(5)
                     for s, name in enumerate(('read_over_std', 'read_energy_fraction'))]
        metrics['scalar'].update((f'{prefix}/{name}', value) for name, value in entries)
