"""Export deterministic training-state controls across a source-only V5 refactor.

Run before/after with the pinned CPU environment and compare the NPZ files. The
nonzero, perturbed parameters exercise learned read/write paths, not just their
zero-initialized controls. Files live outside the repository.
"""
import argparse
import json
from pathlib import Path

from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

import max_utils
import pyconfig
from layers.models import Transformer


RUNS = ('BamMediumPropV5', 'BamXLPropV5',
        'BamPythia2p8BHUnit300BFastKernelMPreNormQKScale',
        'Pythia2p8B', 'MuddPythia3BSerialMLPShortTrain')


def export(directory, run, dtype, shapes_only=False):
  directory.mkdir(parents=True, exist_ok=True)
  (directory / 'equivalence').mkdir(exist_ok=True)
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=run,
      run_name='equivalence', enable_checkpointing=False,
      base_output_directory=str(directory) + '/', jax_cache_dir='', log_config=False,
      dataset_type='synthetic', max_target_length=8, max_prefill_predict_length=8,
      query_chunk_size=4, per_device_batch_size=1., dtype=dtype, weight_dtype='float32',
      **({'logits_fp32_output_bf16_backward': False} if dtype == 'float32' else {}))
  if shapes_only:
    cfg.get_keys().update(bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False)
    if not cfg.bam_enabled:
      cfg.get_keys()['attention'] = 'dot_product'
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    tokens = jnp.ones((1, 8), jnp.int32)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params = jax.eval_shape(lambda key: Transformer(cfg, mesh, quant=None).init(
          key, tokens, jnp.arange(8)[None], tokens, tokens, enable_dropout=False)['params'], jax.random.key(3))
      specs = flatten_dict(nn.get_partition_spec(params))
    shapes = {'/'.join(p): [list(v.shape), str(v.dtype), str(specs[p])]
              for p, v in flatten_dict(nn.unbox(params)).items()}
    (directory / f'{run}-shapes.json').write_text(json.dumps(shapes, indent=2) + '\n')
    print('V5_FULL_TREE', run, len(shapes), flush=True)
    return
  n = 2
  cfg.get_keys().update(emb_dim=n*cfg.head_dim, num_query_heads=n, num_kv_heads=n,
      num_decoder_layers=3, mlp_dim=24, vocab_size=32, emb_bam_num_head=n,
      bam_layer_modes=['local_qk+local_v+local_o']*3,
      bam_attention_write_layers=[0, 1], bam_mlp_write_layers=[0, 1],
      bam_memory_mlp_dim=12, bam_last_mlp_dim=24,
      bam_local_full_m_read_bottleneck_dim=8, bam_write_v_bottleneck_dim=8,
      bam_mlp_write_address_rank=8, emb_bam_v_bottleneck_dim=8,
      bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False,
      record_training_health_metrics=False, record_logits_health_metrics=False,
      record_internal_nn_metrics=False)
  if not cfg.bam_enabled:
    cfg.get_keys().update(attention='dot_product', bam_memory_mlp_dim=0)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  tokens = jax.random.randint(jax.random.key(71), (1, 8), 0, cfg.vocab_size)
  positions = (jnp.arange(8) % 4)[None]
  segments = jnp.where(jnp.arange(8) < 4, 1, 2)[None]
  targets, mask = (tokens + 1) % cfg.vocab_size, jnp.ones_like(tokens)
  model = Transformer(cfg, mesh, quant=None)
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    variables = model.init(jax.random.key(72), tokens, positions, targets, mask,
                           segments, enable_dropout=False)
    flat = flatten_dict(nn.unbox(variables['params']))
    # Keep initialized values too: parameter creation order is part of the contract.
    arrays = {'init/'+'/'.join(p): np.asarray(v, dtype=np.float32) for p, v in flat.items()}
    def perturb(path, x):
      name = path[-1].key
      if name == 'common_qk_read_scale': return x * -.65
      if name == 'common_vo_read_scale': return x * .8
      delta = jnp.sin(jnp.arange(x.size, dtype=jnp.float32).reshape(x.shape) + .3) * .007
      return x + delta.astype(x.dtype)
    params = jax.tree.map_with_path(perturb, nn.unbox(variables['params']))
    def objective(p):
      # Same actual training CE/output-head computation as model.__call__.
      return jnp.mean(model.apply({'params': p}, tokens, positions, targets, mask,
                                  segments, enable_dropout=False)[0].astype(jnp.float32))
    loss, gradients = jax.jit(jax.value_and_grad(objective))(params)
    arrays['loss'] = np.asarray(loss)
    arrays.update({'grad/'+'/'.join(p): np.asarray(v, dtype=np.float32)
                   for p, v in flatten_dict(gradients).items()})
    # A second state catches changed gradients feeding back into the next forward.
    updated = jax.tree.map(lambda p, g: p - .001*g, params, gradients)
    arrays['updated_loss'] = np.asarray(jax.jit(objective)(updated))
  name = f'{run}-{dtype}.npz'
  np.savez_compressed(directory / name, **arrays)
  print('V5_CONTROL', name, float(loss), len(flat), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('directory', type=Path)
  parser.add_argument('--run', action='append')
  parser.add_argument('--dtype', default='float32')
  parser.add_argument('--compare', type=Path)
  parser.add_argument('--shapes-only', action='store_true')
  args = parser.parse_args()
  if args.compare:
    for p in sorted(args.compare.glob('*.npz')):
      before, after = np.load(p), np.load(args.directory/p.name)
      assert set(before.files) == set(after.files), p.name
      for key in before.files:
        np.testing.assert_array_equal(before[key], after[key], err_msg=p.name+': '+key)
      print('V5_BITWISE_EQUAL', p.name, len(before.files), flush=True)
    for p in sorted(args.compare.glob('*-shapes.json')):
      assert json.loads(p.read_text()) == json.loads((args.directory/p.name).read_text()), p.name
      print('V5_FULL_TREE_EQUAL', p.name, flush=True)
  else:
    for run in args.run or RUNS:
      export(args.directory, run, args.dtype, args.shapes_only)


if __name__ == '__main__':
  main()
