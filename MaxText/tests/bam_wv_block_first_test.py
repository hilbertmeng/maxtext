"""Sparse standard V restoration: scoped parameters, scan schedule and consumed gradients."""
import functools
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers import quantizations
from layers.models import Transformer
from tests.bam_mlp_write_test import MLPWriteTest

PARENT = 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
STEM = PARENT.removesuffix('TruePile')
KEEP = STEM + 'WVBlockFirstKeepVOTruePile'
DROP = STEM + 'WVBlockFirstNoVOTruePile'

class SparseWVTest(unittest.TestCase):
    setUp = MLPWriteTest.setUp
    tearDown = MLPWriteTest.tearDown
    config = MLPWriteTest.config

    def shaped(self, c):
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
        args, kw, sharding, model = train_compile.get_shaped_inputs(mesh, c)
        return mesh, args, kw, sharding, model, flatten_dict(args[0].params)

    def test_budget_schedule_and_health(self):
        for exp, total, first_width in [(KEEP,432115072,3365),(DROP,432109408,3440)]:
            c = self.config(exp)
            self.assertEqual((c.emb_dim,c.num_decoder_layers,c.num_query_heads,c.head_dim),(1200,18,16,75))
            self.assertEqual((c.bam_k,c.bam_v,c.bam_abs_v_compression_dim),(75,48,12))
            self.assertEqual(c.bam_local_v_replace,[False,True,True]*6)
            self.assertEqual(c.mlp_dim_by_block,[first_width,3550,3765])
            self.assertEqual((c.bam_mlp_write_every,c.bam_mlp_write_offset),(3,2))
            self.assertEqual(c.DATASET_VARIANT,'truepile4096')
            mesh,args,kw,sharding,model,flat = self.shaped(c)
            self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),total)
            for stage in ('local_0','local_1','fetch_2'):
                prefix=('params','decoder','layers',stage,'block')
                attn=prefix+('self_attention',)
                self.assertEqual(attn+('value','kernel') in flat,stage=='local_0')
                if stage=='local_0':
                    self.assertEqual(flat[attn+('value','kernel')].shape,(1200,6,16,75))
                vo = exp==KEEP or stage!='local_0'
                for name in ('W_R','W_lv_gate'):
                    self.assertEqual(any(p[:len(attn)]==attn and name in p for p in flat),vo)
                for name in ('static_v_key','static_o_key','abs_v_cache_projection'):
                    self.assertEqual(attn+(name,) in flat,vo)
                self.assertIn(attn+('out','kernel'),flat)
                self.assertEqual(prefix+('mlp_address_down','kernel') in flat,stage=='local_1')
            with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
                scalar=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
            for l in range(18):
                self.assertEqual(f'bam/concat/mlp_write_gate/layer_{l:03d}/mean' in scalar,l%3==1)
            print('SPARSE_WV_BUDGET_HEALTH',exp,total,flush=True)

    def test_bool_parent_equivalence_and_periodicity_guard(self):
        c = self.config(PARENT)
        original = self.shaped(c)[-1]
        c.get_keys()['bam_local_v_replace']=[True]*18
        self.assertEqual({p:v.shape for p,v in original.items()},
                         {p:v.shape for p,v in self.shaped(c)[-1].items()})
        c.get_keys()['bam_local_v_replace']=[False]+[True]*17
        with self.assertRaises(AssertionError):
            self.shaped(c)

    def test_two_block_gradients(self):
        for exp in (KEEP,DROP):
            c=self.config(exp,dtype='float32',weight_dtype='float32')
            modes=['local_qk' if exp==DROP and l%3==0 else 'local_qk+local_v+local_o' for l in range(6)]
            c.get_keys().update(emb_dim=150,base_emb_dim=150,num_query_heads=2,num_kv_heads=2,
                base_num_query_heads=2,base_num_kv_heads=2,num_decoder_layers=6,base_num_decoder_layers=6,
                mlp_dim=128,base_mlp_dim=128,mlp_dim_by_block=[128,112,128],vocab_size=64,
                bam_layer_modes=modes,bam_local_v_replace=[False,True,True]*2,
                bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,
                bam_mlp_write_address_rank=16)
            mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
            model=Transformer(c,mesh,quantizations.configure_quantization(c))
            tokens=jnp.array([[1,2,3,4]],jnp.int32);mask=jnp.ones_like(tokens)
            args=(tokens,jnp.arange(4)[None],tokens,mask,mask)
            with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
                params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*args,enable_dropout=False)['params']
                def loss(p):
                    logits=model.apply({'params':p},*args,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]
                    return jnp.mean(logits**2)
                value,grads=jax.jit(jax.value_and_grad(loss))(params)
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grads)))
            flat=flatten_dict(nn.unbox(grads))
            for name in ('value','mlp_address_down','mlp_address_up','mlp_write_gate','W_emb_u'):
                self.assertGreater(sum(float(jnp.sum(v*v)) for p,v in flat.items() if name in p),0.,name)
            print('SPARSE_WV_TWO_BLOCK_GRADIENTS_OK',exp,float(value),flush=True)

if __name__=='__main__':unittest.main()
