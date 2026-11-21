"""Inference benchmark of one BamV5ForCausalLM directory (LoomFormer V5 or MHA), BF16, fast path.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python bam_hf_v5/bench.py HF_DIR \
      [--compile max-autotune] [--prefill 1x4096,8x4096] [--decode 1,16,64] [--context 4032] [--steps 32]
      [--out result.json]

Prefill: one forward without cache, logits for the last position only. Decode: one-token steps after a
prompt of `context` tokens through fixed-shape static caches (CUDA-graph safe); prompts of larger
batches are prefilled 8 rows at a time. Run models in separate processes (bench_pair.sh alternates them):
concurrent max-autotune compiles on one GPU disturb each other.
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
import fast_inference  # noqa: E402


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


def main():
  p = argparse.ArgumentParser()
  p.add_argument('hf_dir')
  p.add_argument('--compile', default='max-autotune')
  p.add_argument('--prefill', default='1x4096,8x4096')
  p.add_argument('--decode', default='1,16,64')
  p.add_argument('--context', type=int, default=4032)
  p.add_argument('--steps', type=int, default=32)
  p.add_argument('--out')
  a = p.parse_args()
  hf = AutoModelForCausalLM.from_pretrained(a.hf_dir, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
  model = fast_inference.build(hf)
  vocab = hf.config.vocab_size
  r = {'device': torch.cuda.get_device_name(0), 'hf_dir': a.hf_dir, 'compile': a.compile,
       'bam': hf.config.bam_enabled}
  with torch.no_grad():
    for shape in filter(None, a.prefill.split(',')):
      b, t = map(int, shape.split('x'))
      ids = torch.randint(0, vocab, (b, t), device='cuda')
      # prefill without CUDA graphs (GPU-bound; capturing the XL LoomFormer prefill intermittently fails with
      # cudaErrorStreamCaptureInvalidated under torch 2.14)
      mode = 'max-autotune-no-cudagraphs' if a.compile == 'max-autotune' else a.compile
      fn = compiled(lambda x: model.prefill(x), mode)
      torch.cuda.reset_peak_memory_stats()
      sec = timed(lambda: fn(ids))
      r[f'prefill_{shape}'] = {'ms': sec * 1e3, 'tok_per_s': b * t / sec,
                               'peak_gib': torch.cuda.max_memory_allocated() / 2 ** 30}
      print(f'prefill_{shape}', r[f'prefill_{shape}'], flush=True)
      torch._dynamo.reset()
    for b in map(int, filter(None, a.decode.split(','))):
      ids = torch.randint(0, vocab, (b, a.context), device='cuda')
      nxt = torch.randint(0, vocab, (b, 1), device='cuda')
      torch.cuda.reset_peak_memory_stats()
      state = model.new_state(b, a.context + a.steps)
      piece = min(b, 8)
      part = model.new_state(piece, a.context + a.steps, state.layout)
      for row in range(0, b, piece):
        part.reset()
        model.prefill(ids[row:row + piece], part)
        state.load_rows(part, row)
      del part
      step = compiled(lambda x: model.decode(x, state), a.compile)

      def run():
        state.reset(a.context)
        for _ in range(a.steps):
          step(nxt)
      sec = timed(run, warmup=1, iters=3) / a.steps
      r[f'decode_bs{b}'] = {'ms_per_step': sec * 1e3, 'tok_per_s': b / sec,
                            'peak_gib': torch.cuda.max_memory_allocated() / 2 ** 30}
      print(f'decode_bs{b}', r[f'decode_bs{b}'], flush=True)
      del state
      torch._dynamo.reset()
      torch.cuda.empty_cache()
  if a.out:
    with open(a.out, 'w') as f:
      json.dump(r, f, indent=1)


if __name__ == '__main__':
  main()
