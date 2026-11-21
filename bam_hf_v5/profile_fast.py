"""Per-kernel GPU time of the compiled fast path (torch.profiler), prefill or one decode step.

  python bam_hf_v5/profile_fast.py HF_DIR [--prefill 8x4096 | --decode 16] [--context 4032] [--top 40]
"""
import argparse
import collections
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_inference  # noqa: E402
from bench import compiled  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument('hf_dir')
  p.add_argument('--prefill', default='')
  p.add_argument('--decode', type=int, default=0)
  p.add_argument('--context', type=int, default=4032)
  p.add_argument('--compile', default='max-autotune')
  p.add_argument('--top', type=int, default=40)
  a = p.parse_args()
  hf = AutoModelForCausalLM.from_pretrained(a.hf_dir, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
  model = fast_inference.build(hf)
  vocab = hf.config.vocab_size
  with torch.no_grad():
    if a.prefill:
      b, t = map(int, a.prefill.split('x'))
      ids = torch.randint(0, vocab, (b, t), device='cuda')
      fn = compiled(lambda x: model.prefill(x), a.compile)
      run, reps = (lambda: fn(ids)), 3
    else:
      b = a.decode
      state = model.new_state(b, a.context + 64)
      piece = min(b, 8)
      part = model.new_state(piece, a.context + 64, state.layout)
      ids = torch.randint(0, vocab, (b, a.context), device='cuda')
      for r in range(0, b, piece):
        part.reset()
        model.prefill(ids[r:r + piece], part)
        state.load_rows(part, r)
      del part
      nxt = ids[:, :1]
      fn = compiled(lambda x: model.decode(x, state), a.compile)

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
