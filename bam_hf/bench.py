"""Paired inference benchmark: BAM vs the MHA control, same code path, same GPU.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python bam_hf/bench.py \
      [--models bam,mha] [--compile none|default|max-autotune|max-autotune-no-cudagraphs|reduce-overhead] \
      [--prefill 1x4096,4x4096,8x2048] [--decode 1,16,64] [--context 4096] [--steps 32]

Weights are random (speed does not depend on their values). Prefill: one forward without cache,
logits for the last position only. Decode: a fixed-shape step on a StaticKVCache holding `context`
tokens (filled by 512-token chunks), so CUDA-graph modes apply.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configuration_bam import BamConfig  # noqa: E402
from modeling_bam import BamForCausalLM, StaticKVCache  # noqa: E402


def build(name, dtype, device):
  config = BamConfig(bam_enabled=name == 'bam')
  torch.manual_seed(0)
  with torch.device(device):
    model = BamForCausalLM(config).to(dtype).eval()
  return model


def timed(fn, warmup=3, iters=10):
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - start) / iters


def compiled(fn, mode):
  if mode == 'none':
    return fn
  return torch.compile(fn, mode=None if mode == 'default' else mode, fullgraph=True, dynamic=False)


def bench_prefill(model, b, t, mode):
  ids = torch.randint(0, 50000, (b, t), device='cuda')
  fn = compiled(lambda x: model(x, use_cache=False, logits_to_keep=1).logits, mode)
  with torch.no_grad():
    sec = timed(lambda: fn(ids))
  return {'ms': sec * 1e3, 'tok_per_s': b * t / sec}


def bench_decode(model, b, context, steps, mode, chunk=512):
  cache = StaticKVCache(model.config, b, context + steps, torch.bfloat16, 'cuda')
  ids = torch.randint(0, 50000, (b, context), device='cuda')
  nxt = torch.randint(0, 50000, (b, 1), device='cuda')
  step = compiled(lambda x: model(x, past_key_values=cache, logits_to_keep=1).logits, mode)
  with torch.no_grad():
    for s in range(0, context, chunk):
      model(ids[:, s:s + chunk], past_key_values=cache, logits_to_keep=1)

    def run():
      cache.reset(context)
      for _ in range(steps):
        step(nxt)

    sec = timed(run, warmup=2, iters=3) / steps
  del cache
  return {'ms_per_step': sec * 1e3, 'tok_per_s': b / sec}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--models', default='bam,mha')
  p.add_argument('--compile', default='none')
  p.add_argument('--prefill', default='1x4096,4x4096,8x2048')
  p.add_argument('--decode', default='1,16,64')
  p.add_argument('--context', type=int, default=4096)
  p.add_argument('--steps', type=int, default=32)
  p.add_argument('--fuse', action='store_true', help='fuse_projections() (both models)')
  p.add_argument('--coordinate-descent', action='store_true', help='inductor coordinate_descent_tuning')
  p.add_argument('--out')
  a = p.parse_args()
  if a.coordinate_descent:
    import torch._inductor.config as inductor_config
    inductor_config.coordinate_descent_tuning = True
  print('device', torch.cuda.get_device_name(0), 'compile', a.compile, flush=True)
  results = {'device': torch.cuda.get_device_name(0), 'compile': a.compile, 'fuse': a.fuse}
  for name in a.models.split(','):
    model = build(name, torch.bfloat16, 'cuda')
    if a.fuse:
      model.fuse_projections()
    r = results[name] = {'params': sum(x.numel() for x in model.parameters())}
    for shape in filter(None, a.prefill.split(',')):
      b, t = map(int, shape.split('x'))
      torch.cuda.reset_peak_memory_stats()
      r[f'prefill_{shape}'] = bench_prefill(model, b, t, a.compile)
      r[f'prefill_{shape}']['peak_gib'] = torch.cuda.max_memory_allocated() / 2**30
      print(name, f'prefill_{shape}', r[f'prefill_{shape}'], flush=True)
    for b in filter(None, a.decode.split(',')):
      torch.cuda.reset_peak_memory_stats()
      r[f'decode_bs{b}'] = bench_decode(model, int(b), a.context, a.steps, a.compile)
      r[f'decode_bs{b}']['peak_gib'] = torch.cuda.max_memory_allocated() / 2**30
      print(name, f'decode_bs{b}', r[f'decode_bs{b}'], flush=True)
      torch._dynamo.reset()        # release this batch size's CUDA-graph pool before the next
      torch.cuda.empty_cache()
    del model
    torch._dynamo.reset()
    torch.cuda.empty_cache()
  if 'bam' in results and 'mha' in results:
    results['bam_over_mha'] = {k: v['tok_per_s'] / results['mha'][k]['tok_per_s']
                               for k, v in results['bam'].items() if isinstance(v, dict)}
    print('BAM/MHA throughput', json.dumps({k: round(x, 3) for k, x in results['bam_over_mha'].items()}))
  if a.out:
    with open(a.out, 'w') as f:
      json.dump(results, f, indent=1)


if __name__ == '__main__':
  main()
