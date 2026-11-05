"""Inference benchmark: Pythia unit-H BAM vs the Pythia-2.8B baseline, same GPU, BF16.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python bam_hf_pythia/bench.py \
      --models bam,pythia [--impl ref|fast] [--compile none|default|max-autotune] \
      [--prefill 1x2048,8x2048] [--decode 1,16,64] [--context 1984] [--steps 32]

Prefill: one forward without cache, logits for the last position only. Decode: one-token steps after
a prompt of `context` tokens; `--impl fast` uses fixed-shape static caches (CUDA-graph safe).
impl ref: BAM reference path / stock HF GPTNeoX (sdpa), DynamicCache, eager.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BAM_DIR = '/data0/xd/bam_diagnostics/pythia-h-prenorm-qkscale-hf/hf/bam-pythia-h-prenorm-qkscale-43000-bfloat16'
PYTHIA_DIR = ('/data0/mengqy/cache/huggingface/hub/models--EleutherAI--pythia-2.8b/snapshots/'
              '2a259cdd96a4beb1cdf467512e3904197345f6a9')


def load(name, impl):
  path = {'bam': BAM_DIR, 'pythia': PYTHIA_DIR}[name]
  model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16,
                                               attn_implementation='sdpa' if name == 'pythia' else None)
  model = model.cuda().eval()
  if impl == 'fast':
    import fast_inference
    model = fast_inference.build(model)
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


def bench_prefill(model, impl, b, t, mode):
  ids = torch.randint(0, 50000, (b, t), device='cuda')
  if impl == 'fast':
    fn = compiled(lambda x: model.prefill(x), mode)
  else:
    fn = compiled(lambda x: model(x, use_cache=False, logits_to_keep=1).logits, mode)
  with torch.no_grad():
    sec = timed(lambda: fn(ids))
  return {'ms': sec * 1e3, 'tok_per_s': b * t / sec}


def bench_decode(model, impl, b, context, steps, mode):
  ids = torch.randint(0, 50000, (b, context), device='cuda')
  nxt = torch.randint(0, 50000, (b, 1), device='cuda')
  with torch.no_grad():
    if impl == 'fast':
      state = model.new_state(b, context + steps)
      step = compiled(lambda x: model.decode(x, state), mode)
      piece = min(b, 8)
      part = model.new_state(piece, context + steps, state.layout)
      for r in range(0, b, piece):
        part.reset()
        model.prefill(ids[r:r + piece], part)
        state.load_rows(part, r)
      del part

      def run():
        state.reset(context)
        for _ in range(steps):
          step(nxt)
    else:
      out = model(ids, use_cache=True, logits_to_keep=1)
      base = out.past_key_values

      def run():
        import copy
        cache = copy.deepcopy(base)
        for _ in range(steps):
          cache = model(nxt, past_key_values=cache, use_cache=True).past_key_values

    sec = timed(run, warmup=1, iters=3) / steps
  return {'ms_per_step': sec * 1e3, 'tok_per_s': b / sec}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--models', default='bam,pythia')
  p.add_argument('--impl', default='ref', choices=('ref', 'fast'))
  p.add_argument('--compile', default='none')
  p.add_argument('--prefill', default='1x2048,8x2048')
  p.add_argument('--decode', default='1,16,64')
  p.add_argument('--context', type=int, default=1984)
  p.add_argument('--steps', type=int, default=32)
  p.add_argument('--out')
  a = p.parse_args()
  print('device', torch.cuda.get_device_name(0), 'impl', a.impl, 'compile', a.compile, flush=True)
  results = {'device': torch.cuda.get_device_name(0), 'impl': a.impl, 'compile': a.compile}
  for name in a.models.split(','):
    model = load(name, a.impl)
    r = results[name] = {}
    for shape in filter(None, a.prefill.split(',')):
      b, t = map(int, shape.split('x'))
      torch.cuda.reset_peak_memory_stats()
      r[f'prefill_{shape}'] = bench_prefill(model, a.impl, b, t, a.compile)
      r[f'prefill_{shape}']['peak_gib'] = torch.cuda.max_memory_allocated() / 2**30
      print(name, f'prefill_{shape}', r[f'prefill_{shape}'], flush=True)
    for b in filter(None, a.decode.split(',')):
      torch.cuda.reset_peak_memory_stats()
      r[f'decode_bs{b}'] = bench_decode(model, a.impl, int(b), a.context, a.steps, a.compile)
      r[f'decode_bs{b}']['peak_gib'] = torch.cuda.max_memory_allocated() / 2**30
      print(name, f'decode_bs{b}', r[f'decode_bs{b}'], flush=True)
      torch._dynamo.reset()
      torch.cuda.empty_cache()
    del model
    torch._dynamo.reset()
    torch.cuda.empty_cache()
  if 'bam' in results and 'pythia' in results:
    results['bam_over_pythia'] = {k: v['tok_per_s'] / results['pythia'][k]['tok_per_s']
                                  for k, v in results['bam'].items() if isinstance(v, dict)}
    print('BAM/Pythia throughput', json.dumps({k: round(x, 3) for k, x in results['bam_over_pythia'].items()}))
  if a.out:
    with open(a.out, 'w') as f:
      json.dump(results, f, indent=1)


if __name__ == '__main__':
  main()
