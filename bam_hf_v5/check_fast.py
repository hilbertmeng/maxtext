"""Check the inference fast path against MaxText reference logits and the HF reference path.

  python bam_hf_v5/check_fast.py HF_DIR REF_NPZ [--dtype bfloat16|float32] [--compile none|max-autotune]
      [--decode-steps 16] [--batch 1]

Reports the full prefill and a prefill of the first T - k tokens followed by k one-token decode steps
through the static caches, both against the JAX logits, and the HF reference path at the same dtype.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_inference  # noqa: E402
from compare import stats  # noqa: E402


def compiled(fn, mode):
  if mode == 'none':
    return fn
  return torch.compile(fn, mode=None if mode == 'default' else mode, fullgraph=True, dynamic=False)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('hf_dir')
  p.add_argument('ref')
  p.add_argument('--dtype', default='bfloat16')
  p.add_argument('--compile', default='none')
  p.add_argument('--decode-steps', type=int, default=16)
  p.add_argument('--batch', type=int, default=0, help='first rows only (0: all)')
  a = p.parse_args()
  torch.backends.cuda.matmul.allow_tf32 = False
  ref = np.load(a.ref)
  rows = slice(None, a.batch or None)
  tokens = torch.as_tensor(ref['tokens'][rows], dtype=torch.long, device='cuda')
  b, t = tokens.shape
  hf = AutoModelForCausalLM.from_pretrained(a.hf_dir, trust_remote_code=True, dtype=getattr(torch, a.dtype))
  hf = hf.cuda().eval()
  target = ref['logits'][rows]
  report = {}
  with torch.no_grad():
    hf_logits = hf(tokens, use_cache=False).logits.float()
    report['hf_ref_vs_jax'] = stats(target, hf_logits)
    fast = fast_inference.build(hf)
    prefill = compiled(lambda x: fast.prefill(x, last_only=False), a.compile)
    report['fast_prefill_vs_jax'] = stats(target, prefill(tokens).float())
    k = a.decode_steps
    state = fast.new_state(b, t)
    steps = [fast.prefill(tokens[:, :t - k], state)]
    step = compiled(lambda x: fast.decode(x, state), a.compile)
    for i in range(t - k, t - 1):
      steps.append(step(tokens[:, i:i + 1]).clone())
    report['fast_decode_vs_jax'] = stats(target[:, t - k - 1:t - 1], torch.cat(steps, 1).float())
    report['hf_ref_vs_jax_same_positions'] = stats(target[:, t - k - 1:t - 1], hf_logits[:, t - k - 1:t - 1])
  print(json.dumps(report, indent=1))


if __name__ == '__main__':
  main()
