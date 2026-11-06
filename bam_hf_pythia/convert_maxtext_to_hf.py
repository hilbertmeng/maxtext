"""Convert flat MaxText parameters (jax_reference.py --params-out) to a HuggingFace model directory.

  python bam_hf_pythia/convert_maxtext_to_hf.py params.npz OUT_DIR [--dtype bfloat16|float32] [--tokenizer DIR]

Scanned layers store layer i of `decoder/layers/block` at axis 1 (param_scan_axis=1);
`decoder/final_local_layer/block` is the last layer. Generalized read pre-biases become the biases of
the key up-projections and every gate bias the bias of its Linear. Every MaxText leaf must be used
once and every PyTorch parameter assigned once.
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
from configuration_bam_pythia import BamPythiaConfig  # noqa: E402
from modeling_bam_pythia import BamPythiaForCausalLM  # noqa: E402

CODE = ('configuration_bam_pythia.py', 'modeling_bam_pythia.py')


def linear(k):
  """DenseGeneral kernel [in, *out] (or [*in, out] for the O projection) -> nn.Linear weight [out, in]."""
  return k.reshape(k.shape[0], -1).T


def flat(x):
  return x.reshape(-1)


def layer_map(writer):
  """(MaxText suffix under .../block/, PyTorch suffix under model.layers.i., transform)."""
  a, s = 'self_attention/', 'self_attn.'
  m = [
      ('pre_self_attention_layer_norm/scale', 'input_layernorm.weight', None),
      ('pre_self_attention_layer_norm/bias', 'input_layernorm.bias', None),
      ('post_self_attention_layer_norm/scale', 'post_attention_layernorm.weight', None),
      ('post_self_attention_layer_norm/bias', 'post_attention_layernorm.bias', None),
      ('mlp/wi/kernel', 'mlp.fc_in.weight', linear),
      ('mlp/wi/bias', 'mlp.fc_in.bias', None),
      ('mlp/wo/kernel', 'mlp.fc_out.weight', linear),
      ('mlp/wo/bias', 'mlp.fc_out.bias', None),
      (a + 'M_shift_gate/kernel', s + 'm_shift.weight', linear),
      (a + 'M_shift_gate_b0', s + 'm_shift.bias', None),
      (a + 'query/kernel', s + 'query.weight', linear),
      (a + 'query/bias', s + 'query.bias', flat),
      (a + 'key/kernel', s + 'key.weight', linear),
      (a + 'key/bias', s + 'key.bias', flat),
      (a + 'full_m_read_down/kernel', s + 'read_down.weight', linear),
      (a + 'W_lq_c8_up/kernel', s + 'q_key_up.weight', linear),
      (a + 'general_q_pre_bias', s + 'q_key_up.bias', flat),
      (a + 'W_lk_c8_up/kernel', s + 'k_key_up.weight', linear),
      (a + 'general_k_pre_bias', s + 'k_key_up.bias', flat),
      (a + 'W_lv_c8_up/kernel', s + 'v_key_up.weight', linear),
      (a + 'general_v_pre_bias', s + 'v_key_up.bias', flat),
      (a + 'W_R_up/kernel', s + 'o_key_up.weight', linear),          # [640, N, f=1, V]
      (a + 'general_vo_pre_bias', s + 'o_key_up.bias', flat),
      (a + 'W_lq_gate/kernel', s + 'q_gate.weight', linear),
      (a + 'W_lq_gate_b0', s + 'q_gate.bias', flat),
      (a + 'W_lk_gate/kernel', s + 'k_gate.weight', linear),
      (a + 'W_lk_gate_b0', s + 'k_gate.bias', flat),
      (a + 'W_lv_gate/kernel', s + 'v_gate.weight', linear),
      (a + 'W_lv_gate_b0', s + 'v_gate.bias', flat),
      (a + 'W_R_gate/kernel', s + 'o_gate.weight', linear),
      (a + 'W_R_gate_b0', s + 'o_gate.bias', flat),
      (a + 'common_qk_read_scale', s + 'qk_scale', None),
      (a + 'common_vo_read_scale', s + 'vo_scale', None),
      (a + 'qk_norm/q_norm/scale', s + 'q_norm.weight', lambda x: x + 1.0),     # (1 + scale) RMSNorm
      (a + 'qk_norm/k_norm/scale', s + 'k_norm.weight', lambda x: x + 1.0),
      (a + 'out/kernel', s + 'out.weight', lambda k: k.reshape(-1, k.shape[-1]).T),
      (a + 'out/bias', s + 'out.bias', None),
  ]
  for arm in 'qkvo':
    m += [(a + f'general_{arm}_static_gate/kernel', s + f'{arm}_static_gate.weight', linear),
          (a + f'general_{arm}_static_gate_bias', s + f'{arm}_static_gate.bias', None),
          (a + f'static_{arm}_key', s + f'{arm}_static_key', None)]
  if writer:
    m += [
        (a + 'P_loc_down/kernel', s + 'write_down.weight', linear),
        (a + 'P_loc_up/kernel', s + 'write_up.weight', linear),
        (a + 'P_loc_up/bias', s + 'write_up.bias', flat),
        (a + 'W_gw/kernel', s + 'write_gate.weight', linear),
        (a + 'gw_b0', s + 'write_gate.bias', None),
        ('memory_mlp/wi/kernel', 'memory_mlp.fc_in.weight', linear),
        ('memory_mlp/wo/kernel', 'memory_mlp.fc_out.weight', linear),
        ('mlp_write_gate/kernel', 'mlp_write_gate.weight', linear),
        ('mlp_write_gate_bias', 'mlp_write_gate.bias', None),
        ('mlp_address_down/kernel', 'mlp_address_down.weight', linear),
        ('mlp_address_up/kernel', 'mlp_address_up.weight', linear),
        ('mlp_address_up/bias', 'mlp_address_up.bias', flat),
    ]
  return m


def infer_config(params, **kwargs):
  scanned = params['decoder/layers/block/mlp/wi/kernel']
  d, n_scan, width = scanned.shape
  n, _, k, _ = params['decoder/layers/block/self_attention/out/kernel'].shape
  layers = n_scan + 1
  return BamPythiaConfig(
      vocab_size=params['token_embedder/embedding'].shape[0], hidden_size=d, num_hidden_layers=layers,
      num_attention_heads=n, head_dim=k, bam_k=k,
      bam_v=params['decoder/layers/block/self_attention/M_shift_gate_b0'].shape[0],
      standard_qk_dim=params['decoder/layers/block/self_attention/query/bias'].shape[-1],
      read_bottleneck_dim=params['decoder/layers/block/self_attention/full_m_read_down/kernel'].shape[-1],
      write_bottleneck_dim=params['decoder/layers/block/self_attention/P_loc_down/kernel'].shape[-1],
      embedding_write_bottleneck_dim=params['decoder/embedding_bam_write/W_emb_v_down/kernel'].shape[-1],
      mlp_write_address_dim=params['decoder/layers/block/mlp_address_down/kernel'].shape[-1],
      intermediate_size=width,
      last_intermediate_size=params['decoder/final_local_layer/block/mlp/wi/kernel'].shape[-1],
      memory_mlp_dim=params['decoder/layers/block/memory_mlp/wi/kernel'].shape[-1],
      **kwargs)


def convert(params, config):
  out, used = {}, set()

  def get(key):
    used.add(key)
    return params[key]

  def put(name, value):
    assert name not in out, name
    out[name] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))

  put('model.embed_tokens.weight', get('token_embedder/embedding'))
  e = 'decoder/embedding_bam_write/'
  put('model.embedding_write.data.weight', linear(get(e + 'W_emb_u/kernel')))
  put('model.embedding_write.address_down.weight', linear(get(e + 'W_emb_v_down/kernel')))
  put('model.embedding_write.address_up.weight', linear(get(e + 'W_emb_v_up/kernel')))
  put('model.embedding_write.address_up.bias', flat(get(e + 'W_emb_v_up/bias')))
  put('model.embedding_write.gate.weight', linear(get(e + 'W_emb_g/kernel')))
  put('model.embedding_write.gate.bias', get(e + 'emb_gw_b0'))
  put('model.norm.weight', get('decoder/lm_head/decoder_norm/scale'))
  put('model.norm.bias', get('decoder/lm_head/decoder_norm/bias'))
  put('lm_head.weight', get('decoder/lm_head/logits_dense').T)
  last = config.num_hidden_layers - 1
  for i in range(config.num_hidden_layers):
    root, index = (('decoder/final_local_layer/block/', None) if i == last
                   else ('decoder/layers/block/', i))
    for src, dst, fn in layer_map(i in config.attention_write_layers):
      value = get(root + src)
      if index is not None:
        value = np.take(value, index, axis=1)
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
  config.auto_map = {'AutoConfig': 'configuration_bam_pythia.BamPythiaConfig',
                     'AutoModelForCausalLM': 'modeling_bam_pythia.BamPythiaForCausalLM'}
  config.architectures = ['BamPythiaForCausalLM']
  if a.source:
    config.maxtext_source = a.source
  state = convert(params, config)
  with torch.device('meta'):
    model = BamPythiaForCausalLM(config)
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
  for f in CODE:
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
