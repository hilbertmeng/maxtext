"""Check the inference fast path against MaxText reference logits and the HF reference path.

  python bam_hf_pythia/check_fast.py REF_NPZ [--model bam|pythia] [--compile none|max-autotune] [--decode-steps 16]

REF_NPZ holds JAX FP32 logits and tokens (jax_reference.py --reference-out); for pythia only its tokens
are used and the stock HF model is the reference. Reports full prefill, and prefill of the first
T - k tokens followed by k one-token decode steps through the static caches.
"""
import argparse
import json

import numpy as np
import torch

import bench
import fast_inference
from compare import stats


def main():
  p = argparse.ArgumentParser()
  p.add_argument('ref')
  p.add_argument('--model', default='bam')
  p.add_argument('--compile', default='none')
  p.add_argument('--decode-steps', type=int, default=16)
  a = p.parse_args()
  ref = np.load(a.ref)
  tokens = torch.as_tensor(ref['tokens'], dtype=torch.long, device='cuda')
  b, t = tokens.shape
  hf = bench.load(a.model, 'ref')
  report = {}
  with torch.no_grad():
    hf_logits = hf(tokens, use_cache=False).logits.float()
    fast = fast_inference.build(hf)
    target = ref['logits'] if a.model == 'bam' else hf_logits.cpu().numpy()
    if a.model == 'bam':
      report['hf_ref_bf16_vs_target'] = stats(target, hf_logits)
    prefill = bench.compiled(lambda x: fast.prefill(x, last_only=False), a.compile)
    report['fast_prefill_vs_target'] = stats(target, prefill(tokens).float())
    k = a.decode_steps
    state = fast.new_state(b, t)
    steps = [fast.prefill(tokens[:, :t - k], state)]
    step = bench.compiled(lambda x: fast.decode(x, state), a.compile)
    for i in range(t - k, t - 1):
      steps.append(step(tokens[:, i:i + 1]).clone())
    report['fast_decode_vs_target'] = stats(target[:, t - k - 1:t - 1], torch.cat(steps, 1).float())
    report['hf_ref_vs_target_same_positions'] = stats(target[:, t - k - 1:t - 1], hf_logits[:, t - k - 1:t - 1])
  print(json.dumps(report, indent=1))


if __name__ == '__main__':
  main()
