"""XLProp address-space expansion: exact budget and consumed-write gradients."""
import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import train
import train_compile
from layers import quantizations
from layers.models import Transformer
from tests.bam_mlp_write_test import MLPWriteTest

EXP = 'BamXLPropK96V60C15EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdTruePile'


class XLV60TransferTest(unittest.TestCase):
    setUp = MLPWriteTest.setUp
    tearDown = MLPWriteTest.tearDown
    config = MLPWriteTest.config

    def test_full_budget_scope_and_writer_health(self):
        c = self.config(EXP)
        self.assertEqual((c.emb_dim, c.num_decoder_layers, c.num_query_heads, c.head_dim),
                         (1920, 28, 20, 96))
        self.assertEqual((c.bam_k, c.bam_v, c.bam_abs_v_compression_dim), (96, 60, 15))
        self.assertEqual((c.bam_local_qk_col_output_dim, c.bam_standard_qk_dim), (72, 24))
        self.assertEqual(c.bam_mlp_write_address_rank, 576)
        self.assertEqual(c.bam_write_v_bottleneck_dim, 600)
        self.assertEqual(c.emb_bam_v_bottleneck_dim, 600)
        self.assertEqual(c.mlp_dim_by_block, [6092, 5773, 6093])
        self.assertEqual(c.DATASET_VARIANT, 'truepile4096')
        self.assertEqual((c.steps, c.learning_rate_schedule_steps, c.learning_rate), (50000, 50000, 2e-4))
        self.assertEqual(c.bam_final_local_mlp_dim, 6094)
        self.assertEqual(c.wd_mults, [('.*scale$', 0.0), ('.*bias$', 0.0)])
        self.assertTrue(c.bam_extra_final_local_layer)
        self.assertTrue(c.bam_local_v_replace)
        self.assertTrue(c.bam_local_vo_static)
        self.assertTrue(c.bam_local_vo_independent_gates)
        self.assertTrue(c.bam_mlp_write_content_rms)
        self.assertTrue(c.bam_write_data_rms)
        self.assertTrue(all('full' not in m for m in c.bam_layer_modes))
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
        args, kw, sharding, model = train_compile.get_shaped_inputs(mesh, c)
        flat = flatten_dict(args[0].params)
        up_key = ('params', 'decoder', 'layers', 'local_1', 'block', 'mlp_address_up', 'kernel')
        up_sharding = flatten_dict(sharding.params)[up_key]
        self.assertEqual(flat[up_key].shape[0], 576)
        total = sum(int(np.prod(v.shape)) for v in flat.values())
        self.assertEqual(total, 1432396680)
        self.assertLess(abs(total - 1432398720), 5760 / 2)
        self.assertFalse(any('value' in p for p in flat))
        for block in ('local_0', 'local_1', 'fetch_2'):
            prefix = ('params', 'decoder', 'layers', block, 'block', 'self_attention')
            for name in ('static_q_key', 'static_k_key', 'static_v_key', 'static_o_key'):
                self.assertEqual(flat[prefix + (name,)].shape, (60, 9, 20))
            self.assertEqual(flat[prefix + ('abs_v_cache_projection',)].shape, (60, 9, 15))
            self.assertEqual(flat[prefix + ('P_loc_up', 'kernel')].shape, (600, 9, 20, 60))
            self.assertEqual(flat[prefix + ('W_R', 'kernel')].shape, (1920, 9, 20, 1, 15))
        self.assertEqual(flat[('params', 'decoder', 'embedding_bam_write', 'W_emb_v_up', 'kernel')].shape, (600, 20, 60))
        self.assertEqual(c.bam_write_outer_implementation, 'dot')
        self.assertEqual(c.bam_read_implementation, 'dot_btn')
        for block in ('local_0', 'local_1', 'fetch_2'):
            prefix = ('params', 'decoder', 'layers', block, 'block')
            self.assertEqual(flat[prefix + ('self_attention', 'query', 'kernel')].shape,
                             (1920, 9, 20, 24))
            self.assertEqual(prefix + ('mlp_address_down', 'kernel') in flat,
                             block == 'local_1')
        with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
            metrics = jax.eval_shape(functools.partial(train.train_step, model, c, sharding),
                                     *args, **kw)[1]
        for l in range(28):
            self.assertEqual(f'bam/concat/mlp_write_gate/layer_{l:03d}/mean'
                             in metrics['scalar'], l in range(1, 28, 3))
            self.assertEqual(f'bam/concat/mlp_address_alignment/layer_{l:03d}/mean_cosine'
                             in metrics['scalar'], l in range(1, 28, 3))
        print('XLPROP_V60_EXACT_BUDGET_HEALTH', total, total - 1432398720, flush=True)

    def test_two_blocks_finite_consumed_write_gradients(self):
        c = self.config(EXP, dtype='float32', weight_dtype='float32')
        c.get_keys().update(emb_dim=192, base_emb_dim=192, num_query_heads=2,
                           num_kv_heads=2, base_num_query_heads=2, base_num_kv_heads=2,
                           num_decoder_layers=7, base_num_decoder_layers=7,
                           mlp_dim=128, base_mlp_dim=128, mlp_dim_by_block=[128, 112, 128],
                           vocab_size=64, bam_layer_modes=['local_qk+local_v+local_o'] * 7,
                           bam_write_v_bottleneck_dim=16, emb_bam_num_head=2,
                           emb_bam_v_bottleneck_dim=16, bam_mlp_write_address_rank=16, bam_final_local_mlp_dim=128)
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
        model = Transformer(c, mesh, quantizations.configure_quantization(c))
        tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
        mask = jnp.ones_like(tokens)
        args = (tokens, jnp.arange(4)[None], tokens, mask, mask)
        rngs = {'params': jax.random.key(3), 'dropout': jax.random.key(4), 'aqt': jax.random.key(5)}
        with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
            params = model.init(rngs, *args, enable_dropout=False)['params']

            def loss(p):
                logits = model.apply({'params': p}, *args, enable_dropout=False,
                                     rngs={'aqt': jax.random.key(5)})[0]
                return jnp.mean(logits ** 2)

            value, grads = jax.jit(jax.value_and_grad(loss))(params)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grads)))
        flat = flatten_dict(nn.unbox(grads))
        for name in ('mlp_address_down', 'mlp_address_up', 'mlp_write_gate', 'W_emb_u'):
            energy = sum(float(jnp.sum(v * v)) for p, v in flat.items() if name in p)
            self.assertGreater(energy, 0., name)
        print('XLPROP_V60_TWO_BLOCK_GRADIENTS_OK', float(value), flush=True)


if __name__ == '__main__':
    unittest.main()
