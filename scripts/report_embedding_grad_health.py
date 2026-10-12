"""Read early TensorBoard samples and attribute global clipping to embedding leaves."""
import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util


def read_points(directory, end):
  points = {}
  for path in sorted(directory.glob('events.out.tfevents*')):
    with path.open('rb') as stream:
      while header := stream.read(12):
        if len(header) != 12:
          break
        size = struct.unpack('<Q', header[:8])[0]
        payload = stream.read(size)
        if len(payload) != size or len(stream.read(4)) != 4:
          break  # The active event may end with an incomplete record.
        event = event_pb2.Event.FromString(payload)
        if event.step > end:
          break
        for value in event.summary.value:
          if not (value.tag.startswith('raw_grads/') or value.tag in (
              'learning/raw_grad_norm', 'learning/grad_norm', 'learning/loss')):
            continue
          kind = value.WhichOneof('value')
          number = value.simple_value if kind == 'simple_value' else (
              np.asarray(tensor_util.make_ndarray(value.tensor)).item() if kind == 'tensor' else None)
          if number is not None:
            points.setdefault(int(event.step), {})[value.tag] = float(number)
  return points


def describe(points, steps, threshold):
  rows = []
  for step in steps:
    p = points[step]
    norm = p['learning/raw_grad_norm']
    leaves = {k: v for k, v in p.items() if k.startswith('raw_grads/')}
    if not leaves or not math.isclose(sum(v*v for v in leaves.values()), norm*norm, rel_tol=1e-3, abs_tol=1e-12):
      raise RuntimeError(f'Incomplete or inconsistent leaf gradients at step {step}')
    entry = {k: v for k, v in leaves.items() if '/embedding_bam_write/' in k or '/embedding_bam_matrix/' in k}
    entry_energy = sum(v*v for v in entry.values())
    coefficient = min(1., threshold / norm) if norm else 1.
    rows.append(dict(step=step, raw_norm=norm, clipped_norm=p.get('learning/grad_norm'),
        clip_coefficient=coefficient, loss=p.get('learning/loss'),
        entry_norm=math.sqrt(entry_energy), entry_energy_fraction=entry_energy/(norm*norm) if norm else 0.,
        leaf_energy_ratio=sum(v*v for v in leaves.values())/(norm*norm) if norm else 0.,
        top_leaves=[dict(path=k,norm=v,energy_fraction=v*v/(norm*norm) if norm else 0.)
                    for k,v in sorted(leaves.items(),key=lambda z:-z[1])[:5]],
        entry_leaves=entry))
  summaries = {}
  for label, selected in [('including_step0', rows), ('updates10plus', [r for r in rows if r['step']>=10])]:
    if not selected:
      continue
    values=np.array([r['raw_norm'] for r in selected])
    summaries[label]=dict(samples=len(selected),median=float(np.median(values)),p90=float(np.quantile(values,.9)),
        max=float(values.max()),max_step=selected[int(values.argmax())]['step'],
        clipped_samples=int(sum(r['raw_norm']>threshold for r in selected)),
        mean_clip_coefficient=float(np.mean([r['clip_coefficient'] for r in selected])),
        min_clip_coefficient=min(r['clip_coefficient'] for r in selected),
        median_entry_energy_fraction=float(np.median([r['entry_energy_fraction'] for r in selected])))
  return dict(summary=summaries,rows=rows)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('runs',nargs='+');p.add_argument('--root',type=Path,default=Path('/data0/xd/tensorboard_logs'))
  p.add_argument('--end',type=int,default=200);p.add_argument('--threshold',type=float,default=1.)
  p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  points={name:read_points(a.root/name,a.end) for name in a.runs}
  available={name:{step for step,v in d.items() if 'learning/raw_grad_norm' in v} for name,d in points.items()}
  common=sorted(set.intersection(*available.values()))
  if not common:
    raise RuntimeError('No common raw-gradient samples')
  result=dict(end=a.end,threshold=a.threshold,common_steps=common,
      available_steps={name:sorted(steps) for name,steps in available.items()},
      runs={name:describe(points[name],common,a.threshold) for name in a.runs})
  a.output.write_text(json.dumps(result,indent=2)+'\n')
  print('COMMON_STEPS',common)
  for name,d in result['runs'].items():
    print(name,json.dumps(d['summary']))
    top=max((r for r in d['rows'] if r['step']>=10),key=lambda r:r['raw_norm'],default=d['rows'][0])
    print('PEAK',json.dumps(top))


if __name__=='__main__':
  main()
