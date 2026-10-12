"""Kernel time of one compiled decode step grouped by kernel family (BAM vs MHA)."""
import collections
import os
import re
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build, compiled  # noqa: E402
from modeling_bam import StaticKVCache  # noqa: E402


def family(name):
  name = re.sub(r'_\d+$', '', name)
  for key in ('_decode_read_kernel', '_read_kernel', '_write_kernel', '_partial', '_merge'):
    if name.startswith(key):
      return key
  if 'silu' in name:
    return 'mlp gate/up GEMV'
  if name.startswith('triton_tem') and 'gelu' in name:
    return 'small up GEMV (gelu input)'
  if name.startswith('triton_poi') and 'gelu' in name:
    return 'gelu'
  if 'index_copy' in name:
    return 'kv cache write'
  if ('cos' in name or 'sin' in name) and 'mm' not in name:
    return 'rope'
  if 'mm' in name or 'gemm' in name:
    return 'GEMV/GEMM other'
  return 'other: ' + name[:60]


def main():
  b = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  with profile(activities=[ProfilerActivity.CUDA]):
    torch.ones(1, device='cuda').add_(1)
  for name in ('bam', 'mha'):
    model = build(name, torch.bfloat16, 'cuda').fuse_projections()
    cache = StaticKVCache(model.config, b, 4160, torch.bfloat16, 'cuda')
    with torch.no_grad():
      model(torch.randint(0, 50000, (b, 512), device='cuda'), past_key_values=cache, logits_to_keep=1)
      nxt = torch.randint(0, 50000, (b, 1), device='cuda')
      fn = compiled(lambda x: model(x, past_key_values=cache, logits_to_keep=1).logits, 'max-autotune-no-cudagraphs')
      for _ in range(3):
        fn(nxt)
      torch.cuda.synchronize()
      with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
          fn(nxt)
        torch.cuda.synchronize()
    fam, cnt = collections.Counter(), collections.Counter()
    for e in prof.events():
      if e.device_type.name == 'CUDA':
        f = family(e.name)
        fam[f] += e.time_range.elapsed_us() / 5
        cnt[f] += 1 / 5
    print(f'== {name} bs{b}: total {sum(fam.values()):.0f} us, {sum(cnt.values()):.0f} kernels')
    for f, us in fam.most_common():
      print(f'   {us:7.0f} us  {cnt[f]:5.0f} kernels  {f}')
    del model, cache
    torch._dynamo.reset()
    torch.cuda.empty_cache()


if __name__ == '__main__':
  main()
