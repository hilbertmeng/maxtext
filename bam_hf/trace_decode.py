"""Per-layer kernel sequence of one compiled decode step (BAM vs MHA), from a CUDA profiler trace."""
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build, compiled  # noqa: E402
from modeling_bam import StaticKVCache  # noqa: E402


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
        fn(nxt)
        torch.cuda.synchronize()
    ev = sorted([e for e in prof.events() if e.device_type.name == 'CUDA'], key=lambda e: e.time_range.start)
    total = sum(e.time_range.elapsed_us() for e in ev)
    print(f'== {name} bs{b}: {len(ev)} kernels/step, {total:.0f} us kernel time, {len(ev) / 28:.1f} kernels/layer')
    # print layer 5's sequence: kernels between the 5th and 6th occurrence of the layer-start kernel are hard to
    # pin generically; print a window from the middle of the step instead
    mid = len(ev) // 2
    for e in ev[mid:mid + int(len(ev) / 28) + 1]:
      print(f'   {e.time_range.elapsed_us():7.1f} us  {e.name[:110]}')
    del model, cache
    torch._dynamo.reset()
    torch.cuda.empty_cache()


if __name__ == '__main__':
  main()
