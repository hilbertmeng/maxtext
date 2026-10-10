"""Targeted gate for standard-head DirectC address-space expansions."""
import math
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import pyconfig
import train_compile
from layers.models import Transformer

ARMS = [('BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile', 432111616), ('BamMediumPropK75V64C16EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdDirectCTruePile', 432128512)]


def config(name, directory):
    Path(directory, 'audit').mkdir(exist_ok=True)
    return pyconfig.initialize(
        [None, 'MaxText/configs/base.yml'], exp_class=name, run_name='audit',
        enable_checkpointing=False, base_output_directory=directory + '/',
        jax_cache_dir='', log_config=False, dataset_type='synthetic',
        max_target_length=4, max_prefill_predict_length=4, query_chunk_size=2,
        per_device_batch_size=1., bam_splash_attention=False)


def check(name, expected):
    with tempfile.TemporaryDirectory() as directory:
        cfg = config(name, directory)
        assert cfg.bam_local_qk_direct_c8 and not cfg.bam_local_qk_share_basis
        assert cfg.bam_concat_static_qk and cfg.scan_layers and not cfg.qk_norm
        assert cfg.DATASET_VARIANT == 'truepile4096'
        assert cfg.bam_k == 75 and cfg.num_query_heads == 16 and cfg.emb_dim == 1200
        assert cfg.bam_local_qk_col_output_dim == 57 and cfg.bam_standard_qk_dim == 18
        assert cfg.bam_write_v_bottleneck_dim == cfg.bam_v * 8
        assert cfg.emb_bam_v_bottleneck_dim == cfg.bam_mlp_write_address_rank == cfg.bam_v * 8
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        args, _, _, _ = train_compile.get_shaped_inputs(mesh, cfg)
        flat = flatten_dict(nn.unbox(args[0].params))
        count = sum(math.prod(v.shape) for v in flat.values())
        assert count == expected, (name, count, expected)
        assert not any('W_local_packed' in path or 'W_lq_bias' in path for path in flat)
        for arm in ('q', 'k'):
            keys = [v for path, v in flat.items() if f'W_l{arm}_c8' in path]
            # The layer scan axis is axis1: kernel is [D, layers, heads, C].
            assert keys and all(v.shape[0] == 1200 and v.shape[-2:] == (cfg.num_query_heads, cfg.bam_abs_v_compression_dim) for v in keys), [(v.shape) for v in keys]
        print('DIRECT_ADDRESS_PARAMETER_TREE_OK', name, count, flush=True)

        # Keep each actual K/V/C shape and quarter-RoPE split; reduce only width,
        # head count, depth and MLP for the consumed-gradient check.
        k = cfg.bam_k
        cfg.get_keys().update(
            base_emb_dim=2*k, emb_dim=2*k, base_num_query_heads=2,
            base_num_kv_heads=2, num_query_heads=2, num_kv_heads=2,
            emb_bam_num_head=2, bam_mlp_write_num_heads=0,
            bam_write_v_bottleneck_dim=16, emb_bam_v_bottleneck_dim=16,
            bam_mlp_write_address_rank=16, base_num_decoder_layers=3,
            num_decoder_layers=3, bam_layer_modes=['local_qk+local_v+local_o']*3,
            base_mlp_dim=32, mlp_dim=32, mlp_dim_by_block=[32,24,32],
            vocab_size=128, dtype='float32', weight_dtype='float32')
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        model = Transformer(cfg, mesh, quant=None)
        tokens = jnp.array([[1,4,8,2]], jnp.int32)
        pos = jnp.arange(4, dtype=jnp.int32)[None]
        seg = jnp.ones_like(tokens)
        rng = {n: jax.random.key(i) for i, n in enumerate(('params','dropout','aqt'))}
        with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
            params = model.init(rng, tokens, pos, seg, tokens)['params']
            def loss(p):
                result = model.apply({'params': p}, tokens, pos, seg, tokens,
                                     enable_dropout=False, rngs=rng)
                if isinstance(result, tuple):
                    result = result[0]
                return jnp.mean(result.astype(jnp.float32)**2)
            value, grad = jax.jit(jax.value_and_grad(loss))(params)
        assert np.isfinite(float(value))
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad))
        flat = flatten_dict(nn.unbox(grad))
        for key in ('W_lq_c8','W_lk_c8','static_q_key','static_k_key',
                    'mlp_address_down','mlp_address_up'):
            leaves = [v for path, v in flat.items() if key in path]
            assert leaves and sum(float(jnp.sum(v*v)) for v in leaves) > 0, key
        print('DIRECT_ADDRESS_CONSUMED_GRADIENTS_OK', name, float(value), flush=True)


if __name__ == '__main__':
    for name, expected in ARMS:
        check(name, expected)
