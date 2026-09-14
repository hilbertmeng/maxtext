"""Read existing TB only; latest event file wins repeated steps after recovery."""
import argparse
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2


def read(root):
  points = {}
  for path in sorted(root.glob('events.out.tfevents.*')):
    for raw in RawEventFileLoader(str(path)).Load():
      event = event_pb2.Event.FromString(raw)
      for v in event.summary.value:
        if v.WhichOneof('value') != 'simple_value':
          continue
        if v.tag.startswith(('bam/std_tail_write/', 'learning/', 'raw_grads/')):
          points.setdefault(v.tag, {})[event.step] = float(v.simple_value)
  return points


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', default='0,100,200,400,600,800,1000')
  args = parser.parse_args()
  result = {}
  for arm in ['Orth', 'Normal']:
    run = 'BamMediumIndependentLLFBAlignedRowStdTailWrite' + arm
    p = read(Path('/data0/xd/tensorboard_logs') / run)
    last = max(p['learning/raw_grad_norm'])
    out = {'run': run, 'max_step': last, 'windows': {}}
    for step in map(int, args.steps.split(',')):
      if step and step + 20 > last:
        continue
      grid = [0] if step == 0 else list(range(step-20, step+21, 10))
      def mean(tag):
        vals = [p.get(tag, {}).get(s, np.nan) for s in grid]
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else None
      entry = {'raw_grad': mean('learning/raw_grad_norm'), 'layers': {}}
      raw = [p['learning/raw_grad_norm'][s] for s in grid if s in p['learning/raw_grad_norm']]
      grad = [p.get('learning/grad_norm', {}).get(s, np.nan) for s in grid if s in p['learning/raw_grad_norm']]
      entry['clip_fraction'] = float(np.mean(np.asarray(raw) > np.asarray(grad) * 1.0001))
      entry['clip_multiplier'] = float(np.nanmean(np.asarray(grad) / np.asarray(raw)))
      for layer in range(24):
        pre = f'bam/std_tail_write/layer_{layer:03d}/'
        entry['layers'][layer] = {tag[len(pre):]:mean(tag) for tag in p if tag.startswith(pre)}
      for name in ['write_std_tail_projection','write_std_tail_bias','W_R/kernel']:
        tags = [tag for tag in p if tag.startswith('raw_grads/') and tag.endswith('/'+name)]
        energies = [sum(p[tag].get(s, 0.)**2 for tag in tags) for s in grid]
        entry['grad_squared_fraction_' + name] = float(np.mean([
            e / p['learning/raw_grad_norm'][s]**2 for e,s in zip(energies,grid)
            if s in p['learning/raw_grad_norm']]))
        entry['grad_l2_' + name] = float(np.mean(np.sqrt(energies)))
      component_energy = {}
      for tag in p:
        if not tag.startswith('raw_grads/'):
          continue
        component = tag.split('/block/')[-1] if '/block/' in tag else tag
        component_energy[component] = component_energy.get(component, 0.) + np.mean([
            p[tag].get(s, 0.)**2 / p['learning/raw_grad_norm'][s]**2
            for s in grid if s in p['learning/raw_grad_norm']])
      entry['component_squared_grad_fractions'] = component_energy
      out['windows'][step] = entry
    result[arm] = out
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
  main()
