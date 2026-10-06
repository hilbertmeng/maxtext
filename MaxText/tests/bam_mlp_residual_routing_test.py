"""Verify that vector routing preserves M writes, parameter budgets and M-only gradients."""
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
from layers import fusion, quantizations
from layers.models import Transformer
from tests.bam_mlp_write_test import MLPWriteTest

PARENT = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
EXPS = [PARENT.removesuffix('TruePile') + suffix + 'TruePile'
        for suffix in ('ResidualComplement', 'ResidualOff')]


class ResidualRoutingTest(unittest.TestCase):
  setUp = MLPWriteTest.setUp
  tearDown = MLPWriteTest.tearDown
  config = MLPWriteTest.config

  def small_config(self, exp):
    c = self.config(exp, dtype='float32', weight_dtype='float32')
    c.get_keys().update(emb_dim=150, base_emb_dim=150, num_query_heads=2, num_kv_heads=2,
        base_num_query_heads=2, base_num_kv_heads=2, num_decoder_layers=6,
        base_num_decoder_layers=6, mlp_dim=128, base_mlp_dim=128,
        mlp_dim_by_block=[128]*3, vocab_size=64,
        bam_layer_modes=['local_qk+local_v+local_o']*6,
        bam_write_v_bottleneck_dim=16, emb_bam_num_head=2,
        emb_bam_v_bottleneck_dim=16, bam_mlp_write_address_rank=16)
    return c

  def test_exact_budget_health_and_scanned_gradients(self):
    for exp in EXPS:
      c = self.config(exp)
      self.assertEqual(c.mlp_dim_by_block, [3901,3774,3901])
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, kw, sharding, model = train_compile.get_shaped_inputs(mesh,c)
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flatten_dict(args[0].params).values()),432096128)
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        scalars = jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
      for l in range(18):
        self.assertEqual(f'bam/concat/mlp_vector_residual_amplitude/layer_{l:03d}/bam_over_standard' in scalars,l%3==1)
      c = self.small_config(exp)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      tokens = jnp.array([[1,2,3,4]],jnp.int32)
      args = (tokens,jnp.arange(4)[None],tokens,jnp.ones_like(tokens),jnp.ones_like(tokens))
      with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
        model = Transformer(c,mesh,quantizations.configure_quantization(c))
        params = model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*args,enable_dropout=False)['params']
        def loss(p):
          return jnp.mean(model.apply({'params':p},*args,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]**2)
        value,grad = jax.jit(jax.value_and_grad(loss))(params)
      self.assertTrue(np.isfinite(float(value)))
      flat = flatten_dict(nn.unbox(grad))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
      for name in ('mlp_write_gate','mlp_address_up'):
        leaves=[v for p,v in flat.items() if name in p]
        self.assertTrue(leaves)
        self.assertGreater(sum(float(jnp.sum(v**2)) for v in leaves),0,name)
      # In the off arm this MLP has no vector output; it must still learn through M.
      mlp_grads=[v for p,v in flat.items() if 'local_1' in p and 'mlp' in p and p[-1]=='kernel']
      self.assertTrue(mlp_grads)
      self.assertGreater(sum(float(jnp.sum(v**2)) for v in mlp_grads),0)

  def test_same_matrix_write_and_headwise_vector_complement(self):
    for layer_index in (0,1):
      outputs={}
      params=None
      for mode in ('add','complement','off'):
        c=self.small_config(PARENT)
        c.get_keys()['bam_mlp_write_vector_residual_mode']=mode
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        x=jax.random.normal(jax.random.key(10),(1,4,150))
        M=jax.random.normal(jax.random.key(11),(1,4,75,32))
        tokens=jnp.array([[1,2,3,4]],jnp.int32)
        args=(x,jnp.ones_like(tokens),jnp.arange(4)[None],tokens,None,True,'train',None)
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
          layer=fusion.SubDecoderLayer(c,mesh,quantizations.configure_quantization(c),layer_inx=layer_index)
          initialized=layer.init({'params':jax.random.key(12),'aqt':jax.random.key(13)},*args,M_in=M)['params']
          if params is None:
            params=initialized
            if layer_index==1:
              # Two distinct gates make accidental broadcast across heads detectable.
              raw=nn.unbox(params)
              raw['mlp_write_gate']['kernel']=jnp.zeros_like(raw['mlp_write_gate']['kernel'])
              raw['mlp_write_gate_bias']=jnp.log(jnp.array([.2,.7])/(1-jnp.array([.2,.7])))
              params=raw
          else:
            self.assertEqual(set(flatten_dict(nn.unbox(params))),set(flatten_dict(nn.unbox(initialized))))
          outputs[mode]=layer.apply({'params':params},*args,M_in=M,rngs={'aqt':jax.random.key(13)})
      for mode in ('complement','off'):
        np.testing.assert_allclose(outputs[mode][1],outputs['add'][1],rtol=1e-6,atol=1e-6)
      if layer_index==0:
        for mode in ('complement','off'):
          np.testing.assert_allclose(outputs[mode][0],outputs['add'][0],rtol=1e-6,atol=1e-6)
      else:
        y=(outputs['add'][0]-outputs['off'][0]).reshape(1,4,2,75)
        expected=outputs['off'][0]+(y*jnp.array([.8,.3])[None,None,:,None]).reshape(x.shape)
        np.testing.assert_allclose(outputs['complement'][0],expected,rtol=2e-5,atol=2e-5)
        self.assertGreater(float(jnp.sum(y*y)),0)


if __name__=='__main__':
  unittest.main()
