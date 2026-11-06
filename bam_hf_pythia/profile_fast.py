"""Per-kernel GPU time of the compiled fast path (torch.profiler), prefill or one decode step.

  python bam_hf_pythia/profile_fast.py --model bam|pythia [--prefill 8x2048 | --decode 16] [--top 40]
"""
import argparse
import collections

import torch
from torch.profiler import ProfilerActivity, profile

import bench


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--model', default='bam')
  p.add_argument('--prefill', default='')
  p.add_argument('--decode', type=int, default=0)
  p.add_argument('--context', type=int, default=1984)
  p.add_argument('--compile', default='max-autotune')
  p.add_argument('--top', type=int, default=40)
  a = p.parse_args()
  model = bench.load(a.model, 'fast')
  with torch.no_grad():
    if a.prefill:
      b, t = map(int, a.prefill.split('x'))
      ids = torch.randint(0, 50000, (b, t), device='cuda')
      fn = bench.compiled(lambda x: model.prefill(x), a.compile)
      run, reps = (lambda: fn(ids)), 3
    else:
      b = a.decode
      state = model.new_state(b, a.context + 64)
      piece = min(b, 8)
      part = model.new_state(piece, a.context + 64, state.layout)
      ids = torch.randint(0, 50000, (b, a.context), device='cuda')
      for r in range(0, b, piece):
        part.reset()
        model.prefill(ids[r:r + piece], part)
        state.load_rows(part, r)
      del part
      nxt = ids[:, :1]
      fn = bench.compiled(lambda x: model.decode(x, state), a.compile)

      def run():
        state.reset(a.context)
        fn(nxt)
      reps = 10
    for _ in range(3):
      run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
      for _ in range(reps):
        run()
      torch.cuda.synchronize()
  totals, counts = collections.Counter(), collections.Counter()
  for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
      totals[e.name] += e.device_time
      counts[e.name] += 1
  total = sum(totals.values()) / reps
  print(f'GPU total per run: {total / 1e3:.3f} ms, kernels per run: {sum(counts.values()) / reps:.0f}')
  for name, us in totals.most_common(a.top):
    print(f'{us / reps / 1e3:8.3f} ms {counts[name] / reps:6.0f}x  {name[:150]}')


if __name__ == '__main__':
  main()
