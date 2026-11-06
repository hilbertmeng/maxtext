"""Compare the HuggingFace port with MaxText reference logits, and incremental decoding with full forwards.

  python bam_hf/compare.py HF_DIR REF_NPZ [--dtype float32|bfloat16] [--device cuda:0]
"""
import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def stats(ref, out):
  ref, out = torch.as_tensor(ref, dtype=torch.float64), out.double().cpu()
  diff = (out - ref).abs()
  lp_ref, lp_out = torch.log_softmax(ref, -1), torch.log_softmax(out, -1)
  kl = (lp_ref.exp() * (lp_ref - lp_out)).sum(-1)
  return {
      'max_abs': diff.max().item(), 'mean_abs': diff.mean().item(),
      'ref_rms': ref.pow(2).mean().sqrt().item(),
      'rel_rms': (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(),
      'kl_mean': kl.mean().item(), 'kl_max': kl.max().item(),
      'top1_agree': (ref.argmax(-1) == out.argmax(-1)).double().mean().item(),
  }


def main():
  p = argparse.ArgumentParser()
  p.add_argument('hf_dir')
  p.add_argument('ref')
  p.add_argument('--dtype', default='float32')
  p.add_argument('--device', default='cuda:0')
  p.add_argument('--decode-steps', type=int, default=16)
  a = p.parse_args()
  dtype = getattr(torch, a.dtype)
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  model = AutoModelForCausalLM.from_pretrained(a.hf_dir, trust_remote_code=True, torch_dtype=dtype).to(a.device).eval()
  ref = np.load(a.ref)
  tokens = torch.as_tensor(ref['tokens'], dtype=torch.long, device=a.device)
  report = {'dtype': a.dtype}
  with torch.no_grad():
    logits = model(tokens, use_cache=False).logits.float()
    report['logits'] = stats(ref['logits'], logits)
    xent = -torch.log_softmax(logits, -1).gather(-1, tokens[..., None])[..., 0]
    report['xent'] = {'jax_mean': float(ref['xent'].mean()), 'torch_mean': xent.mean().item(),
                      'max_abs': (xent.cpu().double() - torch.as_tensor(ref['xent'], dtype=torch.float64)).abs().max().item()}
    # Incremental decoding: prefix forward + one token at a time through the KV cache.
    t = tokens.shape[1]
    k = min(a.decode_steps, t - 1)
    out = model(tokens[:, :t - k], use_cache=True)
    cache, steps = out.past_key_values, [out.logits[:, -1:].float()]
    for i in range(t - k, t - 1):
      out = model(tokens[:, i:i + 1], past_key_values=cache, use_cache=True)
      cache = out.past_key_values
      steps.append(out.logits.float())
    inc = torch.cat(steps, 1)
    report['incremental_vs_full'] = stats(logits[:, t - k - 1:t - 1].cpu().numpy(), inc)
    # Left-padded batch: the padded row must reproduce its unpadded logits.
    pad = 5
    ids = torch.cat((torch.zeros_like(tokens[:1, :pad]), tokens[:1, :t - pad]), 1)
    mask = torch.cat((torch.zeros_like(tokens[:1, :pad]), torch.ones_like(tokens[:1, :t - pad])), 1)
    padded = model(ids, attention_mask=mask, use_cache=False).logits.float()[:, pad:]
    plain = model(tokens[:1, :t - pad], use_cache=False).logits.float()
    report['left_pad_vs_plain'] = stats(plain.cpu().numpy(), padded)
  print(json.dumps(report, indent=1))


if __name__ == '__main__':
  main()
