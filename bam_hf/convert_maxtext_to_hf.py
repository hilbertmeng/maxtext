"""Convert flat MaxText parameters (jax_reference.py --params-out) to a HuggingFace model directory.

  python bam_hf/convert_maxtext_to_hf.py params.npz OUT_DIR [--dtype bfloat16|float32] [--tokenizer DIR]

Scanned blocks store layer b of `layers/{local_0, local_1, fetch_2}` at axis 1 (param_scan_axis=1):
global layer 3b + offset; `final_local_layer` is the last layer. Every MaxText leaf must be used once
and every PyTorch parameter assigned once.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configuration_bam import BamConfig  # noqa: E402
from modeling_bam import BamForCausalLM  # noqa: E402

BLOCK_LAYERS = ('local_0', 'local_1', 'fetch_2')


def linear(k):
  """DenseGeneral kernel [in, *out] (or [*in, out] for the O projection) -> nn.Linear weight [out, in]."""
  return k.reshape(k.shape[0], -1).T


def layer_map(src, dst):
  """(MaxText suffix under .../block/, PyTorch suffix under model.layers.i., transform)."""
  a = 'self_attention/'
  flat = lambda x: x.reshape(-1)
  m = [
      ('pre_self_attention_layer_norm/scale', 'input_layernorm.weight', None),
      ('post_self_attention_layer_norm/scale', 'post_attention_layernorm.weight', None),
      ('mlp/wi_0/kernel', 'mlp.gate_proj.weight', linear),
      ('mlp/wi_1/kernel', 'mlp.up_proj.weight', linear),
      ('mlp/wo/kernel', 'mlp.down_proj.weight', linear),
      (a + 'query/kernel', 'self_attn.q_proj.weight', linear),
      (a + 'key/kernel', 'self_attn.k_proj.weight', linear),
      (a + 'out/kernel', 'self_attn.o_proj.weight', lambda k: k.reshape(-1, k.shape[-1]).T),
      (a + 'full_m_read_down/kernel', 'self_attn.read_down.weight', linear),
      (a + 'W_lq_c8_up/kernel', 'self_attn.q_read_up.weight', linear),
      (a + 'W_lk_c8_up/kernel', 'self_attn.k_read_up.weight', linear),
      (a + 'W_R_up/kernel', 'self_attn.vo_read_up.weight', linear),
      (a + 'general_q_pre_bias', 'self_attn.q_read_pre_bias', None),
      (a + 'general_k_pre_bias', 'self_attn.k_read_pre_bias', None),
      (a + 'general_vo_pre_bias', 'self_attn.vo_read_pre_bias', lambda x: x.reshape(x.shape[0], -1)),
      (a + 'W_lq_gate/kernel', 'self_attn.q_read_gate.proj.weight', linear),
      (a + 'W_lq_gate_b0', 'self_attn.q_read_gate.bias', flat),
      (a + 'W_lk_gate/kernel', 'self_attn.k_read_gate.proj.weight', linear),
      (a + 'W_lk_gate_b0', 'self_attn.k_read_gate.bias', flat),
      (a + 'W_lv_gate/kernel', 'self_attn.v_read_gate.proj.weight', linear),
      (a + 'W_lv_gate_b0', 'self_attn.v_read_gate.bias', flat),
      (a + 'W_R_gate/kernel', 'self_attn.o_read_gate.proj.weight', linear),
      (a + 'W_R_gate_b0', 'self_attn.o_read_gate.bias', flat),
      (a + 'P_loc_down/kernel', 'self_attn.write_address_down.weight', linear),
      (a + 'P_loc_up/kernel', 'self_attn.write_address_up.weight', linear),
      (a + 'P_loc_up/bias', 'self_attn.write_address_up.bias', flat),
      (a + 'W_gw/kernel', 'self_attn.write_gate.proj.weight', linear),
      (a + 'gw_b0', 'self_attn.write_gate.bias', None),
  ]
  for arm in 'qkvo':
    m += [(a + f'general_{arm}_static_gate/kernel', f'self_attn.{arm}_static_gate.proj.weight', linear),
          (a + f'general_{arm}_static_gate_bias', f'self_attn.{arm}_static_gate.bias', None),
          (a + f'static_{arm}_key', f'self_attn.{arm}_static_key', None)]
  m += [('mlp_write_gate/kernel', 'mlp_write.gate.proj.weight', linear),
        ('mlp_write_gate_bias', 'mlp_write.gate.bias', None),
        ('mlp_address_down/kernel', 'mlp_write.address_down.weight', linear),
        ('mlp_address_up/kernel', 'mlp_write.address_up.weight', linear),
        ('mlp_address_up/bias', 'mlp_write.address_up.bias', flat)]
  return m


def mha_layer_map():
  a = 'self_attention/'
  return [
      ('pre_self_attention_layer_norm/scale', 'input_layernorm.weight', None),
      ('post_self_attention_layer_norm/scale', 'post_attention_layernorm.weight', None),
      ('mlp/wi_0/kernel', 'mlp.gate_proj.weight', linear),
      ('mlp/wi_1/kernel', 'mlp.up_proj.weight', linear),
      ('mlp/wo/kernel', 'mlp.down_proj.weight', linear),
      (a + 'query/kernel', 'self_attn.q_proj.weight', linear),
      (a + 'key/kernel', 'self_attn.k_proj.weight', linear),
      (a + 'value/kernel', 'self_attn.v_proj.weight', linear),
      (a + 'out/kernel', 'self_attn.o_proj.weight', lambda k: k.reshape(-1, k.shape[-1]).T),
  ]


def is_mha(params):
  return 'decoder/embedding_bam_write/W_emb_g/kernel' not in params


def infer_config(params, **kwargs):
  """Layer count, MLP widths and MLP-write layers as stored in the parameters."""
  if is_mha(params):
    wi = params['decoder/layers/block/mlp/wi_0/kernel']
    n = wi.shape[1]
    return BamConfig(bam_enabled=False, num_hidden_layers=n, intermediate_sizes=[wi.shape[2]] * n, **kwargs)
  widths = {name: params[f'decoder/layers/{name}/block/mlp/wi_0/kernel'].shape for name in BLOCK_LAYERS}
  n_blocks = widths[BLOCK_LAYERS[0]][1]
  sizes = [widths[name][2] for _ in range(n_blocks) for name in BLOCK_LAYERS]
  sizes.append(params['decoder/final_local_layer/block/mlp/wi_0/kernel'].shape[1])
  writes = [3 * b + o for b in range(n_blocks) for o, name in enumerate(BLOCK_LAYERS)
            if f'decoder/layers/{name}/block/mlp_write_gate/kernel' in params]
  assert 'decoder/final_local_layer/block/mlp_write_gate/kernel' not in params
  return BamConfig(bam_enabled=True, num_hidden_layers=len(sizes), intermediate_sizes=sizes,
                   mlp_write_layers=writes, **kwargs)


def convert(params, config):
  """params: {maxtext path: np.ndarray}; returns a PyTorch state dict (FP32)."""
  used, out = set(), {}

  def put(name, value):
    assert name not in out, name
    out[name] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))

  def take(path):
    used.add(path)
    return params[path]

  put('model.embed_tokens.weight', take('token_embedder/embedding'))
  put('model.norm.weight', take('decoder/lm_head/decoder_norm/scale'))
  put('lm_head.weight', linear(take('decoder/lm_head/logits_dense')))
  if config.bam_enabled:
    e = 'decoder/embedding_bam_write/'
    put('model.embedding_write.data.weight', linear(take(e + 'W_emb_u/kernel')))
    put('model.embedding_write.address_down.weight', linear(take(e + 'W_emb_v_down/kernel')))
    put('model.embedding_write.address_up.weight', linear(take(e + 'W_emb_v_up/kernel')))
    put('model.embedding_write.address_up.bias', take(e + 'W_emb_v_up/bias').reshape(-1))
    put('model.embedding_write.gate.proj.weight', linear(take(e + 'W_emb_g/kernel')))
    put('model.embedding_write.gate.bias', take(e + 'emb_gw_b0'))
    n_blocks = (config.num_hidden_layers - 1) // len(BLOCK_LAYERS)
    sources = [(f'decoder/layers/{name}/block/', b) for b in range(n_blocks) for name in BLOCK_LAYERS]
    sources.append(('decoder/final_local_layer/block/', None))
    mapping = layer_map(None, None)
  else:     # one scan over all layers
    sources = [('decoder/layers/block/', i) for i in range(config.num_hidden_layers)]
    mapping = mha_layer_map()
  for i, (prefix, block) in enumerate(sources):
    for src, dst, fn in mapping:
      path = prefix + src
      if path not in params:
        assert dst.startswith('mlp_write.') and i not in config.mlp_write_layers, path
        continue
      value = take(path)
      if block is not None:
        value = np.take(value, block, axis=1)
      put(f'model.layers.{i}.{dst}', fn(value) if fn else value)
  unused = sorted(set(params) - used)
  assert not unused, unused[:10]
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('params')
  p.add_argument('out_dir')
  p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float32'])
  p.add_argument('--tokenizer', help='tokenizer directory to copy (GPT-NeoX-20B / Pythia)')
  p.add_argument('--source', default='', help='provenance note stored in config.json')
  a = p.parse_args()
  data = np.load(a.params)
  params = {k: data[k] for k in data.files}
  config = infer_config(params, torch_dtype=a.dtype)
  config.auto_map = {'AutoConfig': 'configuration_bam.BamConfig',
                     'AutoModelForCausalLM': 'modeling_bam.BamForCausalLM'}
  config.architectures = ['BamForCausalLM']
  if a.source:
    config.maxtext_source = a.source
  state = convert(params, config)
  with torch.device('meta'):
    model = BamForCausalLM(config)
  expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
  assert set(expected) == set(state), sorted(set(expected) ^ set(state))[:10]
  for k, v in state.items():
    assert tuple(v.shape) == expected[k], (k, tuple(v.shape), expected[k])
  dtype = getattr(torch, a.dtype)
  os.makedirs(a.out_dir, exist_ok=True)
  save_file({k: v.to(dtype).contiguous() for k, v in state.items()},
            os.path.join(a.out_dir, 'model.safetensors'), metadata={'format': 'pt'})
  config.save_pretrained(a.out_dir)
  here = os.path.dirname(os.path.abspath(__file__))
  for f in ('configuration_bam.py', 'modeling_bam.py', 'bam_decode_attention.py', 'bam_m_kernels.py'):
    shutil.copy(os.path.join(here, f), a.out_dir)
  with open(os.path.join(a.out_dir, 'generation_config.json'), 'w') as f:
    json.dump({'bos_token_id': config.bos_token_id, 'eos_token_id': config.eos_token_id,
               'pad_token_id': config.eos_token_id}, f, indent=2)
  if a.tokenizer:
    for f in os.listdir(a.tokenizer):
      if f.startswith('tokenizer') or f in ('special_tokens_map.json', 'vocab.json', 'merges.txt'):
        shutil.copy(os.path.join(a.tokenizer, f), a.out_dir)
  print('CONVERTED', a.out_dir, sum(v.numel() for v in state.values()), a.dtype)


if __name__ == '__main__':
  main()
