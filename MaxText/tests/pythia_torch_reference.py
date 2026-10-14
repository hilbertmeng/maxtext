"""Independent Hugging Face GPT-NeoX oracle, run in a torch interpreter.

Usage: python pythia_torch_reference.py input.npz output.npz
The input contains MaxText parameters, tokens and targets. No MaxText imports.
"""
import sys
import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM


def main(source, destination):
  arrays = np.load(source)
  width = arrays['token_embedder/embedding'].shape[1]
  qshape = arrays['decoder/layers_0/self_attention/query/kernel'].shape
  heads, head_dim = qshape[1:]
  depth = len({key.split('/')[1] for key in arrays if key.startswith('decoder/layers_')})
  mlp_width = arrays['decoder/layers_0/mlp/wi/kernel'].shape[-1]
  config = GPTNeoXConfig(vocab_size=arrays['token_embedder/embedding'].shape[0],
      hidden_size=width, num_hidden_layers=depth, num_attention_heads=heads,
      intermediate_size=mlp_width, hidden_act='gelu', rotary_pct=.25,
      rotary_emb_base=10000, max_position_embeddings=32, layer_norm_eps=1e-5,
      use_parallel_residual=True, tie_word_embeddings=False,
      attention_dropout=0., hidden_dropout=0., use_cache=False)
  config._attn_implementation = 'eager'
  model = GPTNeoXForCausalLM(config).float().eval()
  mapping = {}

  def bind(key, param, transform=lambda a:a, inverse=lambda a:a):
    with torch.no_grad():
      param.copy_(torch.from_numpy(np.array(transform(arrays[key]), copy=True)))
    mapping[key] = (param, inverse)

  bind('token_embedder/embedding', model.gpt_neox.embed_in.weight)
  for i, layer in enumerate(model.gpt_neox.layers):
    root = f'decoder/layers_{i}/'
    for name in ('input_layernorm', 'post_attention_layernorm'):
      norm = getattr(layer, name)
      bind(root+name+'/scale', norm.weight)
      bind(root+name+'/bias', norm.bias)
    qkv_keys = [root+f'self_attention/{name}/' for name in ('query','key','value')]
    # NeoX fused QKV is [head, qkv, head_dim], not [qkv, head, head_dim].
    with torch.no_grad():
      weight = np.stack([arrays[k+'kernel'] for k in qkv_keys], axis=2)
      layer.attention.query_key_value.weight.copy_(torch.from_numpy(weight.reshape(width,-1).T.copy()))
      bias = np.stack([arrays[k+'bias'] for k in qkv_keys], axis=1)
      layer.attention.query_key_value.bias.copy_(torch.from_numpy(bias.reshape(-1).copy()))
    for j,k in enumerate(qkv_keys):
      mapping[k+'kernel'] = (layer.attention.query_key_value.weight,
          lambda a,j=j:a.T.reshape(width,heads,3,head_dim)[:,:,j,:])
      mapping[k+'bias'] = (layer.attention.query_key_value.bias,
          lambda a,j=j:a.reshape(heads,3,head_dim)[:,j,:])
    bind(root+'self_attention/out/kernel',layer.attention.dense.weight,
         lambda a:a.reshape(width,width).T,lambda a:a.T.reshape(heads,head_dim,width))
    bind(root+'self_attention/out/bias',layer.attention.dense.bias)
    for name, module in [('wi',layer.mlp.dense_h_to_4h),('wo',layer.mlp.dense_4h_to_h)]:
      bind(root+f'mlp/{name}/kernel',module.weight,lambda a:a.T,lambda a:a.T)
      bind(root+f'mlp/{name}/bias',module.bias)
  root = 'decoder/lm_head/'
  bind(root+'decoder_norm/scale',model.gpt_neox.final_layer_norm.weight)
  bind(root+'decoder_norm/bias',model.gpt_neox.final_layer_norm.bias)
  bind(root+'logits_dense',model.embed_out.weight,lambda a:a.T,lambda a:a.T)
  expected = set(arrays.files) - {'tokens','targets'}
  assert set(mapping) == expected, (expected-set(mapping),set(mapping)-expected)
  logits = model(torch.from_numpy(arrays['tokens']).long()).logits
  targets = torch.from_numpy(arrays['targets']).long()
  xent = torch.nn.functional.cross_entropy(logits.reshape(-1,config.vocab_size),targets.reshape(-1),reduction='none').reshape(targets.shape)
  xent.mean().backward()
  result = {'logits':logits.detach().numpy(), 'xent':xent.detach().numpy()}
  for key,(param,inverse) in mapping.items():
    result['grad/'+key] = inverse(param.grad.numpy())
  np.savez(destination,**result)


if __name__ == '__main__':
  main(*sys.argv[1:])
