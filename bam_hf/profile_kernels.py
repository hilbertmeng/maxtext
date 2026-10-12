"""Kernel-time breakdown (GEMM / attention / other) of compiled prefill and decode, BAM vs MHA.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python bam_hf/profile_kernels.py \
      [--compile max-autotune-no-cudagraphs] [--prefill 1x4096] [--decode 16] [--fuse]
"""
import argparse
import collections
import os
import re
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build, compiled  # noqa: E402
from modeling_bam import StaticKVCache  # noqa: E402

GEMM = re.compile(r'gemm|xmma|cutlass|triton_mm|triton_tem|_mm_|bmm|sm80_|ampere_|cublas', re.I)
ATTN = re.compile(r'flash_fwd|fmha|AttentionKernel|flex|decod', re.I)


def breakdown(fn, iters=5):
  for _ in range(3):
    fn()
  torch.cuda.synchronize()
  with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(iters):
      fn()
    torch.cuda.synchronize()
  cats, top = collections.Counter(), collections.Counter()
  for e in prof.key_averages():
    us = e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total
    if us <= 0:
      continue
    cat = 'attention' if ATTN.search(e.key) else 'gemm' if GEMM.search(e.key) else 'other'
    cats[cat] += us / iters / 1e3
    top[(cat, e.key[:90])] += us / iters / 1e3
  return cats, top


def main():
  with profile(activities=[ProfilerActivity.CUDA]):     # CUPTI warm-up: the first session can be empty
    torch.ones(1, device='cuda').add_(1)
    torch.cuda.synchronize()
  p = argparse.ArgumentParser()
  p.add_argument('--models', default='bam,mha')
  p.add_argument('--compile', default='max-autotune-no-cudagraphs')
  p.add_argument('--prefill', default='1x4096')
  p.add_argument('--decode', default='16')
  p.add_argument('--context', type=int, default=4096)
  p.add_argument('--fuse', action='store_true')
  p.add_argument('--top', type=int, default=12)
  a = p.parse_args()
  for name in a.models.split(','):
    model = build(name, torch.bfloat16, 'cuda')
    if a.fuse:
      model.fuse_projections()
    jobs = []
    for shape in filter(None, a.prefill.split(',')):
      b, t = map(int, shape.split('x'))
      ids = torch.randint(0, 50000, (b, t), device='cuda')
      fn = compiled(lambda x: model(x, use_cache=False, logits_to_keep=1).logits, a.compile)
      jobs.append((f'prefill_{shape}', lambda fn=fn, ids=ids: fn(ids)))
    for b in filter(None, a.decode.split(',')):
      b = int(b)
      cache = StaticKVCache(model.config, b, a.context + 64, torch.bfloat16, 'cuda')
      with torch.no_grad():
        for s in range(0, a.context, 512):
          model(torch.randint(0, 50000, (b, 512), device='cuda'), past_key_values=cache, logits_to_keep=1)
      nxt = torch.randint(0, 50000, (b, 1), device='cuda')
      fn = compiled(lambda x, cache=cache: model(x, past_key_values=cache, logits_to_keep=1).logits, a.compile)

      def run(fn=fn, cache=cache, nxt=nxt):
        cache.reset(a.context)
        return fn(nxt)
      jobs.append((f'decode_bs{b}', run))
    for job, fn in jobs:
      with torch.no_grad():
        cats, top = breakdown(fn)
      total = sum(cats.values())
      print(f'== {name} {job}: total {total:.2f} ms  ' + '  '.join(f'{k} {v:.2f}' for k, v in cats.most_common()))
      for (cat, key), ms in top.most_common(a.top):
        print(f'   {ms:8.3f} ms  {cat:9s} {key}')
    del model
    torch._dynamo.reset()
    torch.cuda.empty_cache()


if __name__ == '__main__':
  main()
