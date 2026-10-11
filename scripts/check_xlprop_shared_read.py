"""Config-only shared-R400 arm; reuse checked full-M source, verify its active gradients."""
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

EXP = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu400TruePile'
with tempfile.TemporaryDirectory() as directory:
    Path(directory, 'audit').mkdir()
    cfg = pyconfig.initialize(
        [None, 'MaxText/configs/base.yml'], exp_class=EXP, run_name='audit',
        enable_checkpointing=False, base_output_directory=directory+'/', jax_cache_dir='',
        log_config=False, dataset_type='synthetic', max_target_length=4,
        max_prefill_predict_length=4, query_chunk_size=2, per_device_batch_size=1.,
        bam_splash_attention=False)
    assert (cfg.emb_dim, cfg.head_dim, cfg.num_query_heads, cfg.num_decoder_layers) == (1920, 96, 20, 28)
    assert cfg.DATASET_VARIANT == 'truepile4096' and cfg.bam_k == 96 and cfg.bam_v == 40
    assert cfg.bam_local_full_m_read_bottleneck_dim == 400 and cfg.bam_local_full_m_read_share_down
    assert cfg.bam_abs_v_compression_dim is None and not cfg.bam_local_o_compress_v
    assert not cfg.bam_general_column_read and not cfg.bam_general_matrix_write
    assert cfg.bam_local_vo_independent_gates and cfg.bam_read_key_scale == .1 and cfg.bam_read_gate_init == .05
    assert cfg.mlp_dim_by_block == [6167, 5979, 6167] and cfg.bam_final_local_mlp_dim == 6167
    assert cfg.bam_record_concat_health and cfg.record_training_health_metrics and not cfg.qk_norm
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    shaped, _, partitions, _ = train_compile.get_shaped_inputs(mesh, cfg)
    flat = flatten_dict(nn.unbox(shaped[0].params))
    count = sum(math.prod(value.shape) for value in flat.values())
    assert count == 1432370680, count
    assert not any('general_' in part for path in flat for part in path)
    specs = flatten_dict(nn.unbox(partitions.params))
    factor = math.prod(mesh.shape[a] for a in (
        'fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'tensor_transpose',
        'tensor_sequence', 'stage', 'expert'))
    overhead = sum(math.prod(specs[path].shard_shape(value.shape))
                   for path, value in flat.items()) / (count/factor) - 1
    assert overhead < cfg.sharding_tolerance, overhead
    print('XL_SHARED_READ_TREE_LAYOUT_OK', count, overhead, flush=True)
    # Three layers suffice: the separate generalized-arm gate already checks the identical tail path.
    cfg.get_keys().update(
        base_emb_dim=768, emb_dim=768, base_num_query_heads=8, base_num_kv_heads=8,
        num_query_heads=8, num_kv_heads=8, emb_bam_num_head=8, bam_mlp_write_num_heads=0,
        bam_write_v_bottleneck_dim=16, emb_bam_v_bottleneck_dim=16,
        bam_mlp_write_address_rank=16, base_num_decoder_layers=3, num_decoder_layers=3,
        bam_layer_modes=['local_qk+local_v+local_o']*3, bam_extra_final_local_layer=False,
        base_mlp_dim=32, mlp_dim=32, mlp_dim_by_block=[32, 24, 32],
        vocab_size=128, dtype='float32', weight_dtype='float32')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.array([[1, 4, 8, 2]], jnp.int32)
    positions = jnp.arange(4)[None]
    segments = jnp.ones_like(tokens)
    rng = {name: jax.random.key(i) for i, name in enumerate(('params', 'dropout', 'aqt'))}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params = nn.unbox(model.init(rng, tokens, positions, segments, tokens)['params'])
        def loss(p):
            result = model.apply({'params': p}, tokens, positions, segments, tokens,
                                 enable_dropout=False, rngs=rng)
            output = result[0] if isinstance(result, tuple) else result
            return jnp.mean(output.astype(jnp.float32)**2)
        value, gradients = jax.jit(jax.value_and_grad(loss))(params)
        assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(gradients))
        gradient_flat = flatten_dict(gradients)
        for name in ('full_m_read_down', 'W_lq_c8_up', 'W_lk_c8_up', 'W_R_up'):
            leaves = [v for path, v in gradient_flat.items() if name in path]
            assert leaves and sum(float(jnp.sum(v*v)) for v in leaves) > 0, name
        print('XL_SHARED_READ_FORWARD_GRAD_OK', float(value), flush=True)
