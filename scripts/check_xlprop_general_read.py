"""XL transfer: exact full tree/sharding, 27+1 tail, initial reads and active health."""
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
import train
import train_compile
from layers.models import Transformer

EXP = 'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
with tempfile.TemporaryDirectory() as directory:
    Path(directory, 'audit').mkdir()
    cfg = pyconfig.initialize(
        [None, 'MaxText/configs/base.yml'], exp_class=EXP, run_name='audit',
        enable_checkpointing=False, base_output_directory=directory+'/', jax_cache_dir='',
        log_config=False, dataset_type='synthetic', max_target_length=4,
        max_prefill_predict_length=4, query_chunk_size=2, per_device_batch_size=1.,
        bam_splash_attention=False)
    assert cfg.DATASET_VARIANT == 'truepile4096'
    assert (cfg.emb_dim, cfg.head_dim, cfg.num_query_heads, cfg.num_decoder_layers) == (1920, 96, 20, 28)
    assert (cfg.bam_k, cfg.bam_v, cfg.bam_local_full_m_read_bottleneck_dim) == (96, 40, 400)
    assert cfg.bam_abs_v_compression_dim is None and not cfg.bam_local_o_compress_v
    assert cfg.bam_general_column_read and not cfg.bam_general_matrix_write
    assert cfg.bam_local_full_m_read_share_down and cfg.bam_local_vo_independent_gates
    assert cfg.bam_read_gate_init == .05 and cfg.bam_read_key_scale == .1
    assert cfg.bam_static_read_gate_init == .99 and not cfg.qk_norm
    assert cfg.bam_local_qk_col_output_dim == 72 and cfg.bam_standard_qk_dim == 24
    assert cfg.bam_write_v_bottleneck_dim == 400 and cfg.bam_mlp_write_address_rank == 384
    assert cfg.mlp_dim_by_block == [6140, 5952, 6140] and cfg.bam_final_local_mlp_dim == 6140
    assert cfg.bam_record_concat_health and cfg.record_training_health_metrics
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    shaped, _, partitions, _ = train_compile.get_shaped_inputs(mesh, cfg)
    flat = flatten_dict(nn.unbox(shaped[0].params))
    count = sum(math.prod(value.shape) for value in flat.values())
    assert count == 1432386360, count
    specs = flatten_dict(nn.unbox(partitions.params))
    factor = math.prod(mesh.shape[a] for a in (
        'fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'tensor_transpose',
        'tensor_sequence', 'stage', 'expert'))
    overhead = sum(math.prod(specs[path].shard_shape(value.shape))
                   for path, value in flat.items()) / (count/factor) - 1
    assert overhead < cfg.sharding_tolerance, overhead
    assert any('final_local_layer' in path for path in flat)
    assert not any('AbsVCompression' in part for path in flat for part in path)
    for path, value in flat.items():
        block_axis = () if 'final_local_layer' in path else (9,)
        if 'full_m_read_down' in path:
            assert value.shape == (1920,) + block_axis + (400,), (path, value.shape)
        if any(name in path for name in ('W_lq_c8_up', 'W_lk_c8_up', 'W_R_up')):
            extra = (1,) if 'W_R_up' in path else ()
            assert value.shape == (400,) + block_axis + (20,) + extra + (40,), (path, value.shape)
    print('XL_GENERAL_READ_TREE_LAYOUT_OK', count, overhead, flush=True)

    # One complete block plus its terminal local layer exercises XL's 27+1 path.
    cfg.get_keys().update(
        base_emb_dim=768, emb_dim=768, base_num_query_heads=8, base_num_kv_heads=8,
        num_query_heads=8, num_kv_heads=8, emb_bam_num_head=8, bam_mlp_write_num_heads=0,
        bam_write_v_bottleneck_dim=16, emb_bam_v_bottleneck_dim=16,
        bam_mlp_write_address_rank=16, base_num_decoder_layers=4, num_decoder_layers=4,
        bam_layer_modes=['local_qk+local_v+local_o']*4,
        base_mlp_dim=32, mlp_dim=32, mlp_dim_by_block=[32, 24, 32],
        bam_final_local_mlp_dim=32, vocab_size=128, dtype='float32', weight_dtype='float32')
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.array([[1, 4, 8, 2]], jnp.int32)
    positions = jnp.arange(4)[None]
    segments = jnp.ones_like(tokens)
    rng = {name: jax.random.key(i) for i, name in enumerate(('params', 'dropout', 'aqt'))}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params = nn.unbox(model.init(rng, tokens, positions, segments, tokens)['params'])

        def forward(p, health=False):
            result = model.apply({'params': p}, tokens, positions, segments, tokens,
                                 enable_dropout=False, rngs=rng,
                                 mutable=['intermediates'] if health else False)
            output, stats = result if health else (result, None)
            output = output[0] if isinstance(output, tuple) else output
            return (output, stats) if health else output

        initial, stats = forward(params, True)
        cfg.get_keys().update(bam_general_column_read=False)
        parent_params = nn.unbox(model.init(rng, tokens, positions, segments, tokens)['params'])
        child_flat = flatten_dict(params)
        for path, value in flatten_dict(parent_params).items():
            np.testing.assert_array_equal(np.asarray(value), np.asarray(child_flat[path]), err_msg='/'.join(path))
        np.testing.assert_allclose(np.asarray(initial), np.asarray(forward(params)), rtol=3e-5, atol=3e-6)
        cfg.get_keys().update(bam_general_column_read=True)
        metrics = {'scalar': {}}
        train.record_bam_concat_health_metrics(metrics, stats, cfg)
        health = {key: value for key, value in metrics['scalar'].items() if '/general_' in key}
        assert health and all(np.isfinite(np.asarray(value)).all() for value in health.values())
        for key, value in health.items():
            if key.endswith('/mean') and '_static_gate/' in key:
                np.testing.assert_allclose(float(value), .99, rtol=2e-6)
        value, gradients = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p).astype(jnp.float32)**2)))(params)
        assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(gradients))
        gradient_flat = flatten_dict(gradients)
        for name in ('full_m_read_down', 'W_lq_c8_up', 'W_lk_c8_up', 'W_R_up',
                     'general_q_pre_bias', 'general_k_pre_bias', 'general_vo_pre_bias', 'general_v_static_gate'):
            leaves = [v for path, v in gradient_flat.items() if name in path]
            assert leaves and sum(float(jnp.sum(v*v)) for v in leaves) > 0, name
        assert any('final_local_layer' in path and 'general_v_static_gate' in path
                   for path in gradient_flat)
        print('XL_GENERAL_READ_INIT_GRAD_HEALTH_OK', len(health), float(value), flush=True)
