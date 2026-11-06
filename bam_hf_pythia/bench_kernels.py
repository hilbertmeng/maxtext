"""Microbenchmark of the fused M read/write kernel at prefill size (random inputs).

  python bam_hf_pythia/bench_kernels.py [--rows 16384] [--batch 8] [--tok 8]
"""
import argparse
import time

import torch

import bam_kernels


def inputs(rows, n=32, device='cuda', dt=torch.bfloat16):
  g = torch.Generator(device=device).manual_seed(0)
  r = lambda *s: torch.randn(*s, device=device, dtype=dt, generator=g)
  z = r(rows, 2944)
  return dict(m_old=r(rows, 5120), c1=r(rows, n * 80), g1=r(rows, n), a1=r(rows, n * 64), c2=r(rows, n * 80),
              g2=r(rows, n), a2=r(rows, n * 64), shift=z[:, :64], std=z[:, 64:64 + 2 * n * 20], up=r(rows, 4 * n * 64),
              gates=z[:, 2600:2600 + 8 * n], static_key=torch.randn(4 * n, 64, device=device, generator=g).to(dt),
              norm_w=torch.randn(2, 100, device=device, generator=g))


def call(x, batch, tok, has_write=True):
  dev = x['m_old'].device
  cos = torch.randn(8192, 10, device=dev, dtype=torch.bfloat16)
  return lambda: bam_kernels.fused_read(
      x['m_old'], x['c1'], x['g1'], x['a1'], x['c2'], x['g2'], x['a2'], x['m_old'],
      torch.zeros((), dtype=torch.int32, device=dev), torch.zeros((), dtype=torch.long, device=dev), x['shift'],
      x['std'], x['up'], x['gates'], x['static_key'], x['norm_w'], cos, cos, batch, has_write, False, 1e-6, 1e-5,
      1e-4, 0.1, 1.1 / 0.99, 1.3, 0.9, 0.1, tok)


def timed(fn, iters=20):
  for _ in range(3):
    fn()
  torch.cuda.synchronize()
  s = time.perf_counter()
  for _ in range(iters):
    fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - s) / iters * 1e3


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--rows', type=int, default=16384)
  p.add_argument('--batch', type=int, default=8)
  p.add_argument('--tok', default='8')
  a = p.parse_args()
  x = inputs(a.rows)
  # bytes that must move: inputs once + outputs once
  nbytes = sum(v.numel() * v.element_size() for k, v in x.items() if k not in ('static_key', 'norm_w', 'shift', 'std', 'gates'))
  nbytes += a.rows * (64 + 1280 + 256) * 2 + a.rows * (5120 + 2 * 32 * 112 + 2 * 32 * 80) * 2
  for tok in map(int, a.tok.split(',')):
    ms = timed(call(x, a.batch, tok))
    print(f'tok {tok}: {ms:.3f} ms  ({nbytes / ms / 1e6:.0f} GB/s effective, {nbytes / 1e6:.0f} MB)')
    ms = timed(call(x, a.batch, tok, has_write=False))
    print(f'tok {tok} no write: {ms:.3f} ms')


if __name__ == '__main__':
  main()
