"""Read-only dynamic-O gamma extraction and head/coordinate decomposition.

Only the two gamma leaves are restored by ``extract``; optimizer/model weights
are never requested. Statistics use canonical [layer, head, address] order.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def coarsen(gamma, mode, xp=np):
  """Keep the parameter shape; replace address coordinates by one head scalar."""
  if mode == 'original':
    return gamma
  mean = xp.mean(gamma, axis=-1, keepdims=True)
  if mode == 'head_mean':
    value = mean
  elif mode == 'head_signed_rms':
    value = xp.where(mean < 0, -1., 1.) * xp.sqrt(xp.mean(gamma * gamma, axis=-1, keepdims=True))
  else:
    raise ValueError(mode)
  return xp.broadcast_to(value, gamma.shape)


def canonical_gamma(leaves):
  """This runtime scans 31 layers on parameter axis1, not axis0."""
  body = [np.asarray(v) for k, v in leaves.items() if 'layers' in k]
  tail = [np.asarray(v) for k, v in leaves.items() if 'final_local_layer' in k]
  if len(body) != 1 or len(tail) != 1:
    raise ValueError([(k, v.shape) for k, v in leaves.items()])
  if body[0].shape != (32, 31, 64) or tail[0].shape != (32, 64):
    raise ValueError((body[0].shape, tail[0].shape))
  return np.concatenate((body[0].transpose(1, 0, 2), tail[0][None]), axis=0)


def distribution(x):
  x = np.asarray(x, np.float64)
  return dict(mean=float(x.mean()), min=float(x.min()),
              p10=float(np.percentile(x, 10)), p50=float(np.median(x)),
              p90=float(np.percentile(x, 90)), p99=float(np.percentile(x, 99)), max=float(x.max()))


def summarize(gamma, initial=.2):
  g = np.asarray(gamma, np.float64)
  if g.shape != (32, 32, 64) or not np.isfinite(g).all():
    raise ValueError('Expected finite [32 layers,32 heads,64 address coordinates]')
  mean = g.mean(axis=-1, keepdims=True)
  std = g.std(axis=-1)
  rms = np.sqrt(np.mean(g * g, axis=-1))
  cv = std / np.maximum(np.abs(mean[..., 0]), 1e-12)
  within = float(np.sum((g - mean) ** 2))
  shift = float(64 * np.sum((mean - initial) ** 2))
  update = float(np.sum((g - initial) ** 2))
  assert np.isclose(within + shift, update, rtol=1e-12, atol=1e-12)
  angle = np.degrees(np.arccos(np.clip(np.abs(mean[..., 0]) / rms, 0., 1.)))
  return dict(
      shape=list(g.shape), initial=initial, negative_fraction=float(np.mean(g < 0)),
      head_mean=distribution(mean), within_head_std=distribution(std),
      within_head_cv=distribution(cv), angle_to_best_scalar_degrees=distribution(angle),
      within_head_energy=within, head_mean_shift_energy=shift, update_energy=update,
      within_head_fraction_of_update=within / update if update else 0.,
      within_head_fraction_of_scale_energy=within / float(np.sum(g * g)),
      head_means=mean[..., 0].tolist(), head_stds=std.tolist(), head_cvs=cv.tolist(),
      head_rms=rms.tolist(), layer_mean=g.mean(axis=(1, 2)).tolist())


def restore_gamma(path):
  import jax
  import orbax.checkpoint as ocp
  from flax.traverse_util import flatten_dict, unflatten_dict
  cp = ocp.PyTreeCheckpointer()
  metadata = cp.metadata(path)
  tree = metadata.item_metadata.tree if hasattr(metadata, 'item_metadata') else metadata
  sharding = jax.sharding.SingleDeviceSharding(jax.devices('cpu')[0])
  selected = {k: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=sharding)
              for k, v in flatten_dict(tree).items()
              if k[0] == 'params' and k[-1] == 'local_o_dynamic_norm_scale'}
  if sum(np.prod(v.shape) for v in selected.values()) != 65536:
    raise ValueError([(k, v.shape) for k, v in selected.items()])
  target = unflatten_dict(selected)
  args = ocp.checkpoint_utils.construct_restore_args(target)
  restored = cp.restore(path, args=ocp.args.PyTreeRestore(
      item=target, restore_args=args, partial_restore=True))
  leaves = {k: np.asarray(v) for k, v in flatten_dict(restored).items()}
  return canonical_gamma(leaves), {'/'.join(k): list(v.shape) for k, v in leaves.items()}


def analyze(output):
  rows, all_stats = [], []
  for file in sorted(output.glob('gamma-*.npz'), key=lambda p: int(p.stem.split('-')[-1])):
    step = int(file.stem.split('-')[-1])
    result = dict(step=step, **summarize(np.load(file)['gamma']))
    (output / f'stats-{step}.json').write_text(json.dumps(result, indent=2) + '\n')
    all_stats.append(result)
    rows.append(dict(step=step, head_mean=result['head_mean']['mean'],
                     std_p50=result['within_head_std']['p50'], cv_p50=result['within_head_cv']['p50'],
                     cv_p90=result['within_head_cv']['p90'], cv_max=result['within_head_cv']['max'],
                     intra_update=result['within_head_fraction_of_update'],
                     intra_total=result['within_head_fraction_of_scale_energy'],
                     negative_fraction=result['negative_fraction']))
  if not rows:
    raise ValueError('No extracted gamma snapshots')
  with (output / 'summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  (output / 'summary.json').write_text(json.dumps(all_stats, indent=2) + '\n')
  print(json.dumps(rows), flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--checkpoint-root')
  p.add_argument('--steps', nargs='+', type=int)
  p.add_argument('--output', required=True, type=Path)
  a = p.parse_args()
  a.output.mkdir(parents=True, exist_ok=True)
  for step in a.steps or []:
    file = a.output / f'gamma-{step}.npz'
    if not file.exists():
      if not a.checkpoint_root:
        p.error('--checkpoint-root required for extraction')
      path = a.checkpoint_root.rstrip('/') + f'/{step}/items'
      gamma, shapes = restore_gamma(path)
      np.savez_compressed(file, gamma=gamma)
      (a.output / f'restore-{step}.json').write_text(json.dumps(
          dict(checkpoint=path, selected_shapes=shapes, values=65536, bytes=262144), indent=2) + '\n')
      print('GAMMA_EXTRACTED', step, flush=True)
  analyze(a.output)


if __name__ == '__main__':
  main()
