"""Parameter, full-address read, and consumed-gradient gates for R128 keys."""
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

NAME = "BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGelu128TruePile"
def check(NAME, expected, rank, shared):
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, 'audit').mkdir()
        cfg = pyconfig.initialize([None,'MaxText/configs/base.yml'], exp_class=NAME,
            run_name='audit', enable_checkpointing=False, base_output_directory=directory+'/',
            jax_cache_dir='', log_config=False, dataset_type='synthetic',
            max_target_length=4, max_prefill_predict_length=4, query_chunk_size=2,
            per_device_batch_size=1., bam_splash_attention=False)
        assert cfg.DATASET_VARIANT == 'truepile4096'
        assert cfg.bam_abs_v_compression_dim is None and not cfg.bam_local_o_compress_v
        assert cfg.bam_local_full_m_read_bottleneck_dim == rank
        assert cfg.bam_read_gate_init == .05 and cfg.bam_read_key_scale == .1
        assert cfg.bam_k == 75 and cfg.bam_v == 32 and not cfg.qk_norm
        assert cfg.bam_local_vo_independent_gates and cfg.scan_layers
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        args, _, _, _ = train_compile.get_shaped_inputs(mesh,cfg)
        flat = flatten_dict(nn.unbox(args[0].params))
        count = sum(math.prod(v.shape) for v in flat.values())
        assert count == expected, count
        assert not any('abs_v_cache_projection' in p or 'W_local_packed' in p for p in flat)
        for name in ('W_lq_c8','W_lk_c8','W_R'):
            down_name = 'full_m_read_down' if shared else name+'_down'
            down = [v for p,v in flat.items() if down_name in p]
            up = [v for p,v in flat.items() if name+'_up' in p]
            assert down and all(v.shape[0] == 1200 and v.shape[-1] == rank for v in down)
            assert up and all(v.shape[0] == rank and v.shape[-1] == 32 for v in up)
            assert not any(name in p for p in flat), name
        print('FULL_M_PARAMETER_TREE_OK', count, flush=True)
        cfg.get_keys().update(base_emb_dim=300,emb_dim=300,base_num_query_heads=4,
            base_num_kv_heads=4,num_query_heads=4,num_kv_heads=4,emb_bam_num_head=4,
            bam_mlp_write_num_heads=0,bam_write_v_bottleneck_dim=16,
            emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16,
            base_num_decoder_layers=3,num_decoder_layers=3,
            bam_layer_modes=['local_qk+local_v+local_o']*3,
            base_mlp_dim=32,mlp_dim=32,mlp_dim_by_block=[32,24,32],
            vocab_size=128,dtype='float32',weight_dtype='float32')
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
        model=Transformer(cfg,mesh,quant=None)
        tokens=jnp.array([[1,4,8,2]],jnp.int32)
        pos=jnp.arange(4,dtype=jnp.int32)[None]
        seg=jnp.ones_like(tokens)
        rng={n:jax.random.key(i) for i,n in enumerate(('params','dropout','aqt'))}
        with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
            params=model.init(rng,tokens,pos,seg,tokens)['params']
            def loss(p):
                out=model.apply({'params':p},tokens,pos,seg,tokens,enable_dropout=False,rngs=rng)
                if isinstance(out,tuple): out=out[0]
                return jnp.mean(out.astype(jnp.float32)**2)
            value,grad=jax.jit(jax.value_and_grad(loss))(params)
        assert np.isfinite(float(value))
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad))
        flat=flatten_dict(nn.unbox(grad))
        for name in (('full_m_read_down',) if shared else ('W_lq_c8_down','W_lk_c8_down','W_R_down')) + ('W_lq_c8_up','W_lk_c8_up','W_R_up',
                     'W_lq_gate','W_lk_gate','W_R_gate','W_lv_gate','mlp_address_down','mlp_address_up'):
            leaves=[v for p,v in flat.items() if name in p]
            assert leaves and sum(float(jnp.sum(v*v)) for v in leaves)>0,name
        print('FULL_M_CONSUMED_GRADIENTS_OK',float(value),flush=True)

check(NAME,432128960,128,False)
check("BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadSharedGelu256TruePile",432125504,256,True)
