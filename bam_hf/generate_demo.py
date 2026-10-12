"""Greedy generation with the converted model; checks cached `generate` against uncached decoding.

  python bam_hf/generate_demo.py HF_DIR [--dtype bfloat16] [--max-new-tokens 40] [--prompt ...]
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    'The capital of France is',
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    'In 1905, Albert Einstein published',
]


def main():
  p = argparse.ArgumentParser()
  p.add_argument('hf_dir')
  p.add_argument('--dtype', default='bfloat16')
  p.add_argument('--device', default='cuda:0')
  p.add_argument('--max-new-tokens', type=int, default=40)
  p.add_argument('--prompt', action='append')
  a = p.parse_args()
  tok = AutoTokenizer.from_pretrained(a.hf_dir)
  tok.padding_side = 'left'
  if tok.pad_token is None:
    tok.pad_token = tok.eos_token
  model = AutoModelForCausalLM.from_pretrained(a.hf_dir, trust_remote_code=True,
                                               torch_dtype=getattr(torch, a.dtype)).to(a.device).eval()
  prompts = a.prompt or PROMPTS
  batch = tok(prompts, return_tensors='pt', padding=True).to(a.device)
  with torch.no_grad():
    out = model.generate(**batch, max_new_tokens=a.max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    agree = []
    for i, prompt in enumerate(prompts):     # uncached greedy reference, one prompt at a time
      ids = tok(prompt, return_tensors='pt').input_ids.to(a.device)
      for _ in range(a.max_new_tokens):
        nxt = model(ids, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat((ids, nxt), 1)
      new = out[i, batch.input_ids.shape[1]:]
      ref = ids[0, -a.max_new_tokens:]
      agree.append((new == ref).float().mean().item())
      print('=' * 80)
      print(repr(prompt + tok.decode(new, skip_special_tokens=True)))
  print('cached batched generate vs uncached greedy, token agreement per prompt:', agree)


if __name__ == '__main__':
  main()
