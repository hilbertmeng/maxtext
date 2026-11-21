"""Convert flat MaxText parameters (export.py) to a HuggingFace directory for BamV5ForCausalLM.

  python bam_hf_v5/convert_maxtext_to_hf.py params.npz OUT_DIR [--dtype bfloat16|float32] [--tokenizer DIR]

`params.npz` carries the flat parameters and, under `__config__`, the resolved MaxText settings the
model needs (export.py). Scanned layers store layer i of `decoder/layers/block` at axis 1; a BAM model's
`decoder/final_local_layer/block` is its last layer. Decoder RMSNorm scales are stored minus one.
Every MaxText leaf must be used once and every PyTorch parameter assigned once.
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
from configuration_bam_v5 import BamV5Config  # noqa: E402
from modeling_bam_v5 import BamV5ForCausalLM  # noqa: E402

CODE = ('configuration_bam_v5.py', 'modeling_bam_v5.py')


def linear(k):
  """DenseGeneral kernel [in, *out] (or [*in, out] for the O projection) -> nn.Linear weight [out, in]."""
  return k.reshape(k.shape[0], -1).T


def flat(x):
  return x.reshape(-1)


def plus_one(x):
  return x + 1.0


def out_proj(k):
  return k.reshape(-1, k.shape[-1]).T


def layer_map(bam, writer):
  """(MaxText suffix under .../block/, PyTorch suffix under model.layers.i., transform)."""
  a, s = 'self_attention/', 'self_attn.'
  m = [('pre_self_attention_layer_norm/scale', 'input_layernorm.weight', plus_one),
       ('post_self_attention_layer_norm/scale', 'post_attention_layernorm.weight', plus_one),
       ('mlp/wi_0/kernel', 'mlp.gate.weight', linear),
       ('mlp/wi_1/kernel', 'mlp.up.weight', linear),
       ('mlp/wo/kernel', 'mlp.down.weight', linear),
       (a + 'query/kernel', s + 'query.weight', linear),
       (a + 'key/kernel', s + 'key.weight', linear),
       (a + 'out/kernel', s + 'out.weight', out_proj)]
  if not bam:
    return m + [(a + 'value/kernel', s + 'value.weight', linear)]
  m += [
      (a + 'M_shift_gate/kernel', s + 'm_shift.weight', linear),
      (a + 'M_shift_gate_b0', s + 'm_shift.bias', None),
      (a + 'full_m_read_down/kernel', s + 'read_down.weight', linear),
      (a + 'W_lq_c8_up/kernel', s + 'q_key_up.weight', linear),
      (a + 'general_q_pre_bias', s + 'q_key_up.bias', flat),
      (a + 'W_lk_c8_up/kernel', s + 'k_key_up.weight', linear),
      (a + 'general_k_pre_bias', s + 'k_key_up.bias', flat),
      (a + 'W_lv_c8_up/kernel', s + 'v_key_up.weight', linear),
      (a + 'general_v_pre_bias', s + 'v_key_up.bias', flat),
      (a + 'W_R_up/kernel', s + 'o_key_up.weight', linear),          # [R, N, f=1, V]
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
        ('memory_mlp/wi_0/kernel', 'memory_mlp.gate.weight', linear),
        ('memory_mlp/wi_1/kernel', 'memory_mlp.up.weight', linear),
        ('memory_mlp/wo/kernel', 'memory_mlp.down.weight', linear),
        ('mlp_write_gate/kernel', 'mlp_write_gate.weight', linear),
        ('mlp_write_gate_bias', 'mlp_write_gate.bias', None),
        ('mlp_address_down/kernel', 'mlp_address_down.weight', linear),
        ('mlp_address_up/kernel', 'mlp_address_up.weight', linear),
        ('mlp_address_up/bias', 'mlp_address_up.bias', flat),
    ]
  return m


def infer_config(params, cfg, **kwargs):
  """Shapes from the parameters, scalar settings from the exported MaxText config."""
  bam = 'decoder/embedding_bam_write/W_emb_u/kernel' in params
  scanned = params['decoder/layers/block/mlp/wi_0/kernel']
  d, n_scan, width = scanned.shape
  n, _, h, _ = params['decoder/layers/block/self_attention/out/kernel'].shape
  common = dict(vocab_size=params['token_embedder/embedding'].shape[0], hidden_size=d,
                num_attention_heads=n, head_dim=h, intermediate_size=width,
                rms_norm_eps=cfg['normalization_layer_epsilon'], rope_min_timescale=cfg['rope_min_timescale'],
                rope_max_timescale=cfg['rope_max_timescale'], max_position_embeddings=cfg['max_target_length'],
                bam_enabled=bam)
  if not bam:
    nope = cfg['bam_partial_rope_nope_dim'] if cfg.get('bam_partial_rope') else 0
    return BamV5Config(num_hidden_layers=n_scan, mha_nope_dim=int(nope or 0), **common, **kwargs)
  sa = 'decoder/layers/block/self_attention/'
  return BamV5Config(
      num_hidden_layers=n_scan + 1,
      bam_k=h, bam_v=params[sa + 'M_shift_gate_b0'].shape[0],
      standard_qk_dim=params[sa + 'query/kernel'].shape[-1],
      read_bottleneck_dim=params[sa + 'full_m_read_down/kernel'].shape[-1],
      write_bottleneck_dim=params[sa + 'P_loc_down/kernel'].shape[-1],
      embedding_write_bottleneck_dim=params['decoder/embedding_bam_write/W_emb_v_down/kernel'].shape[-1],
      mlp_write_address_dim=params['decoder/layers/block/mlp_address_down/kernel'].shape[-1],
      last_intermediate_size=params['decoder/final_local_layer/block/mlp/wi_0/kernel'].shape[-1],
      memory_mlp_dim=params['decoder/layers/block/memory_mlp/wi_0/kernel'].shape[-1],
      # The scanned block is homogeneous: all of its layers write, the terminal layer does not.
      attention_write_layers=list(range(n_scan)) if sa + 'P_loc_down/kernel' in params else [],
      mlp_write_layers=list(range(n_scan)) if 'decoder/layers/block/memory_mlp/wi_0/kernel' in params else [],
      read_key_eps=cfg['bam_read_key_epsilon'], read_key_scale=cfg['bam_read_key_scale'],
      write_eps=cfg['bam_write_rms_epsilon_resolved'], static_read_gate_init=cfg['bam_static_read_gate_init'],
      embedding_write_content_divisor=cfg['bam_embedding_write_content_divisor'],
      mlp_write_content_divisor=cfg['bam_mlp_write_content_divisor'],
      **common, **kwargs)


def convert(params, config):
  out, used = {}, set()

  def get(key):
    used.add(key)
    return params[key]

  def put(name, value):
    assert name not in out, name
    out[name] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))

  put('model.embed_tokens.weight', get('token_embedder/embedding'))
  if config.bam_enabled:
    e = 'decoder/embedding_bam_write/'
    put('model.embedding_write.data.weight', linear(get(e + 'W_emb_u/kernel')))
    put('model.embedding_write.address_down.weight', linear(get(e + 'W_emb_v_down/kernel')))
    put('model.embedding_write.address_up.weight', linear(get(e + 'W_emb_v_up/kernel')))
    put('model.embedding_write.address_up.bias', flat(get(e + 'W_emb_v_up/bias')))
    put('model.embedding_write.gate.weight', linear(get(e + 'W_emb_g/kernel')))
    put('model.embedding_write.gate.bias', get(e + 'emb_gw_b0'))
  put('model.norm.weight', plus_one(get('decoder/lm_head/decoder_norm/scale')))
  put('lm_head.weight', get('decoder/lm_head/logits_dense').T)
  last = config.num_hidden_layers - 1
  for i in range(config.num_hidden_layers):
    terminal = config.bam_enabled and i == last
    root, index = (('decoder/final_local_layer/block/', None) if terminal else ('decoder/layers/block/', i))
    for src, dst, fn in layer_map(config.bam_enabled, i in config.attention_write_layers):
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
  p.add_argument('--tokenizer', help='tokenizer directory to copy')
  p.add_argument('--source', default='', help='provenance note stored in config.json')
  a = p.parse_args()
  data = np.load(a.params)
  cfg = json.loads(str(data['__config__']))
  params = {k: data[k] for k in data.files if k != '__config__'}
  config = infer_config(params, cfg, torch_dtype=a.dtype)
  config.auto_map = {'AutoConfig': 'configuration_bam_v5.BamV5Config',
                     'AutoModelForCausalLM': 'modeling_bam_v5.BamV5ForCausalLM'}
  config.architectures = ['BamV5ForCausalLM']
  config.maxtext_source = a.source or cfg.get('exp', '')
  state = convert(params, config)
  with torch.device('meta'):
    model = BamV5ForCausalLM(config)
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
