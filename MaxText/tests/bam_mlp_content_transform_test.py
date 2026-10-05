"""Write-only head transforms: exact budget, identity equivalence, usable gradients and health."""
import functools, unittest
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers.models import Transformer
from layers import quantizations
from tests.bam_mlp_write_test import MLPWriteTest
PARENT='BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
EXP=PARENT.removesuffix('TruePile')+'ContentTransformTruePile'

class ContentTransformTest(unittest.TestCase):
    setUp=MLPWriteTest.setUp
    tearDown=MLPWriteTest.tearDown
    config=MLPWriteTest.config

    def test_exact_budget_scope_and_health(self):
        c=self.config(EXP)
        self.assertEqual(c.mlp_dim_by_block,[3765,3525,3765])
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
        flat=flatten_dict(args[0].params)
        self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432115072)
        maps=[(p,v) for p,v in flat.items() if 'mlp_write_content_transform' in p]
        self.assertEqual(len(maps),1)
        self.assertIn('local_1',maps[0][0])
        self.assertEqual(maps[0][1].shape,(16,6,75,75))
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
            scalar=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
        for l in range(18):
            for suffix in ('mlp_write_transform/bam_over_standard','mlp_content_alignment/mean_cosine'):
                metric,stat=suffix.split('/')
                self.assertEqual(f'bam/concat/{metric}/layer_{l:03d}/{stat}' in scalar,l%3==1)

    def test_identity_and_nonzero_map_gradients(self):
        c=self.config(EXP,dtype='float32',weight_dtype='float32')
        c.get_keys().update(emb_dim=150,base_emb_dim=150,num_query_heads=2,num_kv_heads=2,
            base_num_query_heads=2,base_num_kv_heads=2,num_decoder_layers=6,base_num_decoder_layers=6,
            mlp_dim=128,base_mlp_dim=128,mlp_dim_by_block=[128]*3,vocab_size=64,
            bam_layer_modes=['local_qk+local_v+local_o']*6,bam_write_v_bottleneck_dim=16,
            emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        tokens=jnp.array([[1,2,3,4]],jnp.int32);mask=jnp.ones_like(tokens)
        args=(tokens,jnp.arange(4)[None],tokens,mask,mask)
        rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)}
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
            model=Transformer(c,mesh,quantizations.configure_quantization(c))
            params=model.init(rng,*args,enable_dropout=False)['params']
            flat=flatten_dict(nn.unbox(params))
            maps=[v for p,v in flat.items() if 'mlp_write_content_transform' in p]
            self.assertEqual(len(maps),1)
            np.testing.assert_array_equal(np.asarray(maps[0]),np.broadcast_to(np.eye(75),maps[0].shape))
            def loss(p):
                logits=model.apply({'params':p},*args,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]
                return jnp.mean(logits**2)
            value,grad=jax.jit(jax.value_and_grad(loss))(params)
            logits=model.apply({'params':params},*args,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
        g=[v for p,v in flatten_dict(nn.unbox(grad)).items() if 'mlp_write_content_transform' in p][0]
        self.assertGreater(float(jnp.sum(g*g)),0)
        c.get_keys()['bam_mlp_write_content_transform']=False
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
            parent=Transformer(c,mesh,quantizations.configure_quantization(c))
            pv=parent.init(rng,*args,enable_dropout=False)['params']
            pflat=flatten_dict(nn.unbox(pv))
            self.assertEqual(set(pflat),set(flat)-{p for p in flat if 'mlp_write_content_transform' in p})
            for p,v in pflat.items():np.testing.assert_array_equal(np.asarray(v),np.asarray(flat[p]))
            plogits=parent.apply({'params':pv},*args,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]
        np.testing.assert_allclose(np.asarray(logits),np.asarray(plogits),rtol=2e-6,atol=2e-6)
        print('IDENTITY_EQUIVALENCE_AND_WRITE_MAP_GRADIENT_OK',float(value),flush=True)

if __name__=='__main__':unittest.main()
