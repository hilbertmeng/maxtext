"""Full deep configurations and both new recipe paths."""
import unittest
import jax
import jax.numpy as jnp
from flax import linen as nn
import max_utils
from layers.models import Transformer
from residual_connections_test import ResidualModelTest
from tests.bam_prop_test_utils import MLPWriteTest
BASE='BamMHAMediumPropL27C256TruePile'
class DeepHC(unittest.TestCase):
    setUp=ResidualModelTest.setUp
    tearDown=ResidualModelTest.tearDown
    config=MLPWriteTest.config
    tiny=ResidualModelTest.tiny
    def test_full_dimensions_and_parameter_budget(self):
        parent=self.config(BASE);counts={}
        for run in [BASE,BASE+'mHC',BASE+'DHC4Tanh']:
            cfg=self.config(run)
            for key in ['emb_dim','num_decoder_layers','head_dim','num_query_heads','steps','learning_rate','init_method','adam_weight_decay','init_weights_seed','bam_partial_rope_nope_dim','remat_policy','data_shuffle_seed']:
                self.assertEqual(getattr(cfg,key),getattr(parent,key),key)
            self.assertEqual(cfg.num_decoder_layers,27);self.assertEqual(cfg.emb_dim,1200)
            self.assertTrue(cfg.record_training_health_metrics)
            self.assertFalse(cfg.record_internal_nn_metrics)
            self.assertEqual(cfg.tensorboard_gcs_rotate_period,500)
            mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
            t=jax.ShapeDtypeStruct((1,4),jnp.int32)
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
                model=Transformer(cfg,mesh,quant=None)
                p=jax.eval_shape(lambda t:nn.unbox(model.init({'params':jax.random.key(7),'dropout':jax.random.key(7)},t,t,t,t,decoder_segment_ids=t,enable_dropout=False)['params']),t)
            counts[run]=sum(x.size for x in jax.tree.leaves(p))
            print('DEEP_FULL_SHAPE',run,counts[run],cfg.mlp_dim,flush=True)
        self.assertEqual(counts[BASE],432142800)
        for run in [BASE+'mHC',BASE+'DHC4Tanh']:self.assertLess(abs(counts[run]-counts[BASE]),27*3*1200)
    def test_new_recipes_forward_backward(self):
        for run in [BASE+'mHC',BASE+'DHC4Tanh']:
            cfg=self.tiny(run);cfg.get_keys()['bam_splash_attention']=False;cfg.get_keys()['mhc_pallas_interpret']=True
            mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
            t=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None,:];mask=jnp.ones_like(t)
            with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
                model=Transformer(cfg,mesh,quant=None)
                p=nn.unbox(model.init({'params':jax.random.key(7),'dropout':jax.random.key(7)},t,pos,t,mask,decoder_segment_ids=mask,enable_dropout=False)['params'])
                def objective(p):return jnp.square(model.apply({'params':p},t,pos,t,mask,decoder_segment_ids=mask,enable_dropout=False)[0]).mean()
                val,g=jax.value_and_grad(objective)(p)
                self.assertTrue(bool(jnp.isfinite(val)))
                self.assertTrue(all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(g)))
                self.assertGreater(float(sum(jnp.square(x).sum() for x in jax.tree.leaves(g))),0.)
