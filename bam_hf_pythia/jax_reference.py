"""MaxText side of the PyTorch port: export parameters and reference logits (pure-JAX path).

Run in the pinned MaxText CPU environment from a worktree at the model's runtime commit (5a48faa):

  JAX_PLATFORMS=cpu PYTHONPATH=MaxText python /path/to/bam_hf_pythia/jax_reference.py \
      --params-out params.npz [--ckpt .../checkpoints/43000/items | --random-seed 0 --layers 3] \
      [--reference-out ref.npz --tokens tokens.npy --dtype float32]

The model runs the executed pure-JAX equations: the fused Pallas core, the Pallas write and Splash
are switched off (their parameters and math are the same; MaxText's tests check native vs Pallas),
and attention uses dense einsums. --ckpt restores only the `params` subtree on CPU. --random-seed
replaces every leaf by a nonzero random tensor (zero-initialized gates/keys would hide port bugs);
--layers builds a shallow fixture (scanned writers + the terminal layer).
"""
import argparse
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

import max_utils
import pyconfig
from layers import quantizations
from layers.models import Transformer

EXP = 'BamPythia2p8BHUnit300BFastKernelMPreNormQKScale'


def build(exp, t, b, dtype, layers=None):
  tmp = tempfile.mkdtemp()
  os.makedirs(os.path.join(tmp, 'hfref'), exist_ok=True)
  extra = {}
  if dtype == 'float32':   # BF16-operand/FP32-output logits; an FP32 model needs the plain tensordot
    extra['logits_fp32_output_bf16_backward'] = False
  c = pyconfig.initialize(
      [None, 'MaxText/configs/base.yml'], exp_class=exp, run_name='hfref', enable_checkpointing=False,
      base_output_directory=tmp + '/', jax_cache_dir='', log_config=False, dataset_type='synthetic',
      max_target_length=t, max_prefill_predict_length=t, per_device_batch_size=float(b),
      attention='dot_product', dtype=dtype, **extra)
  update = dict(bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False)
  if layers:   # shallow fixture for strict alignment (deep random networks are chaotic)
    update.update(base_num_decoder_layers=layers, num_decoder_layers=layers,
                  bam_layer_modes=[c.bam_layer_modes[0]] * layers,
                  bam_attention_write_layers=list(range(layers - 1)))
  c.get_keys().update(update)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
  return c, mesh, Transformer(c, mesh, quantizations.configure_quantization(c))


def call_args(tokens):
  tokens = jnp.asarray(tokens, jnp.int32)
  pos = jnp.broadcast_to(jnp.arange(tokens.shape[1], dtype=jnp.int32)[None], tokens.shape)
  targets = jnp.concatenate((tokens[:, 1:], tokens[:, :1]), axis=1)     # next token (last row unused)
  ones = jnp.ones_like(tokens)
  return tokens, pos, targets, ones, ones


def abstract_params(c, mesh, model, t, b):
  tok = np.zeros((b, t), np.int32)
  with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
    shapes = jax.eval_shape(lambda: model.init(
        {'params': jax.random.key(0), 'dropout': jax.random.key(1), 'aqt': jax.random.key(2)},
        *call_args(tok), enable_dropout=False))
  return nn.unbox(shapes['params'])


def restore_params(path, abstract):
  import orbax.checkpoint as ocp
  cpu = jax.sharding.SingleDeviceSharding(jax.devices('cpu')[0])
  target = jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=cpu), abstract)
  ckptr = ocp.PyTreeCheckpointer()
  meta = ckptr.metadata(path)
  tree = getattr(meta, 'item_metadata', meta)        # orbax >= 0.11 returns StepMetadata
  tree = getattr(tree, 'tree', tree)
  assert 'params' in tree, list(tree)
  item = {'params': {'params': target}} if 'params' in tree['params'] else {'params': target}
  restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=cpu, global_shape=s.shape, dtype=s.dtype),
                              item)
  out = ckptr.restore(path, args=ocp.args.PyTreeRestore(item=item, restore_args=restore_args, partial_restore=True))
  params = out['params']
  return params['params'] if 'params' in params and isinstance(params['params'], dict) and 'decoder' in params['params'] else params


def random_params(abstract, seed):
  flat = flatten_dict(abstract)
  rng = np.random.default_rng(seed)
  out = {}
  for k, s in flat.items():
    name, path = k[-1], '/'.join(k)
    if 'layer_norm' in path or 'decoder_norm' in path:      # direct-scale LayerNorm
      v = (1.0 + rng.normal(0, 0.1, s.shape)) if name == 'scale' else rng.normal(0, 0.1, s.shape)
    elif name == 'scale':                                    # QKNorm stores multiplier - 1
      v = rng.normal(0, 0.2, s.shape)
    elif name.startswith('common_'):
      v = 1.0 + rng.normal(0, 0.3, s.shape)
    elif name.endswith('b0') or name.endswith('_bias') or name == 'bias':
      v = rng.normal(0, 0.5, s.shape)
    elif name.startswith('static_'):
      v = rng.normal(0, 0.3, s.shape)
    elif name == 'embedding':
      v = rng.normal(0, 1.0, s.shape)
    else:    # kernels: fan-in over the leading (input) axis; scanned kernels keep the layer at axis 1
      fan_in = s.shape[0] * (s.shape[2] if k[-2] == 'out' else 1)
      v = rng.normal(0, fan_in ** -0.5, s.shape)
    out[k] = v.astype(np.float32)
  return unflatten_dict(out)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--exp', default=EXP)
  p.add_argument('--ckpt')
  p.add_argument('--random-seed', type=int)
  p.add_argument('--params-in', help='flat npz from an earlier --params-out (skips restore/init)')
  p.add_argument('--params-out')
  p.add_argument('--tokens', help='int .npy [B, T] for the reference forward')
  p.add_argument('--reference-out')
  p.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
  p.add_argument('--layers', type=int, help='shallow fixture: number of decoder layers')
  a = p.parse_args()

  tokens = np.load(a.tokens) if a.tokens else np.zeros((1, 8), np.int32)
  b, t = tokens.shape
  c, mesh, model = build(a.exp, t, b, a.dtype, a.layers)
  abstract = abstract_params(c, mesh, model, t, b)
  if a.params_in:
    data = np.load(a.params_in)
    params = unflatten_dict({tuple(k.split('/')): data[k] for k in data.files})
  elif a.ckpt:
    params = jax.tree.map(np.asarray, restore_params(a.ckpt, abstract))
  else:
    params = random_params(abstract, a.random_seed or 0)
  flat_a = {'/'.join(k): v for k, v in flatten_dict(abstract).items()}
  flat_p = {'/'.join(k): np.asarray(v) for k, v in flatten_dict(params).items()}
  assert set(flat_a) == set(flat_p), sorted(set(flat_a) ^ set(flat_p))[:10]
  for k, v in flat_p.items():
    assert v.shape == flat_a[k].shape, (k, v.shape, flat_a[k].shape)
  print('PARAMS', sum(v.size for v in flat_p.values()), flush=True)
  if a.params_out:
    np.savez(a.params_out, **{k: v.astype(np.float32) for k, v in flat_p.items()})
    print('PARAMS_OUT', a.params_out, flush=True)

  if a.reference_out:
    def forward(params, tokens):
      chunks = []

      def grab(next_fun, args, kwargs, context):
        out = next_fun(*args, **kwargs)
        if context.method_name == 'project_logits':
          chunks.append(out)
        return out

      with nn.intercept_methods(grab):
        xent = model.apply({'params': params}, *call_args(tokens), enable_dropout=False,
                           rngs={'aqt': jax.random.key(5)})[0]
      return jnp.concatenate(chunks, axis=1), xent

    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      logits, xent = jax.jit(forward)(jax.tree.map(jnp.asarray, params), tokens)
    np.savez(a.reference_out, logits=np.asarray(logits, np.float32), xent=np.asarray(xent, np.float32),
             tokens=tokens)
    print('REFERENCE_OUT', a.reference_out, np.asarray(logits).shape,
          'next-token loss', float(np.asarray(xent)[:, :-1].mean()), flush=True)


if __name__ == '__main__':
  main()
