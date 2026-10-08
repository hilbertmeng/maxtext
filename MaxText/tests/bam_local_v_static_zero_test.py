"""Focused LocalV initialization and consumed scanned-gradient checks."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train_compile
from layers.models import Transformer
from layers import quantizations
from bam_mlp_write_test import MLPWriteTest

PARENT = 'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
EXP = PARENT.replace('TruePile', 'LocalVStaticZeroTruePile')

class LocalVStaticZeroTest(unittest.TestCase):
  setUp = MLPWriteTest.setUp
  tearDown = MLPWriteTest.tearDown
  config = MLPWriteTest.config

  def test_full_budget_and_only_initialization_config_change(self):
    configs = [self.config(n) for n in (PARENT, EXP)]
    a,b = [c.get_keys() for c in configs]
    differences = {k for k in a.keys() | b.keys() if str(a.get(k)) != str(b.get(k))}
    self.assertEqual(differences - {'model_name','exp_class','compare_runs'}, {'bam_local_v_static_zero_init'})
    c = configs[1]
    self.assertEqual(c.mlp_dim_by_block, [3901,3774,3901])
    self.assertEqual(c.DATASET_VARIANT, 'truepile4096')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    args,_,_,_ = train_compile.get_shaped_inputs(mesh,c)
    flat = flatten_dict(args[0].params)
    self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432096128)
    self.assertTrue(any('static_v_key' in p for p in flat))
    print('LOCAL_V_ZERO_FULL_BUDGET_OK 432096128',flush=True)

  def test_same_other_parameters_finite_forward_and_live_static_v_gradient(self):
    tokens = jnp.array([[1,2,3,4]],jnp.int32)
    call = (tokens,jnp.arange(4)[None],tokens,jnp.ones_like(tokens),jnp.ones_like(tokens))
    trees = []
    for name in (PARENT,EXP):
      c = self.config(name,dtype='float32',weight_dtype='float32')
      c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,
          base_num_decoder_layers=3,num_decoder_layers=3,mlp_dim=64,
          mlp_dim_by_block=[64]*3,vocab_size=32,
          bam_layer_modes=['local_qk+local_v+local_o']*3,
          bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,
          emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quantizations.configure_quantization(c))
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
        if name==EXP:
          def loss(p):
            return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]**2)
          value,grad=jax.jit(jax.value_and_grad(loss))(params)
      trees.append(flatten_dict(nn.unbox(params)))
    self.assertEqual(trees[0].keys(),trees[1].keys())
    changed=[]
    for p,v in trees[0].items():
      if 'static_v_key' in p:
        self.assertGreater(float(jnp.linalg.norm(v)),0.)
        np.testing.assert_array_equal(trees[1][p],0.)
        changed.append(p)
      else: np.testing.assert_array_equal(v,trees[1][p])
    self.assertTrue(changed)
    self.assertTrue(np.isfinite(float(value)))
    self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad)))
    flat=flatten_dict(nn.unbox(grad))
    self.assertGreater(sum(float(jnp.sum(g*g)) for p,g in flat.items() if 'static_v_key' in p),0.)
    # Both static V and the dynamic C8 key start at zero. Downstream matrix-address
    # gradients must be checked after a nonzero optimizer update, not at step0.
    updated=jax.tree.map(lambda p,g:p-.001*g,params,grad)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      next_value,next_grad=jax.jit(jax.value_and_grad(loss))(updated)
    self.assertTrue(np.isfinite(float(next_value)))
    self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(next_grad)))
    next_flat=flatten_dict(nn.unbox(next_grad))
    for needle in ('static_v_key','mlp_address_up','static_q_key','static_k_key'):
      energy=sum(float(jnp.sum(g*g)) for p,g in next_flat.items() if needle in p)
      self.assertGreater(energy,0.,needle)
      print('AFTER_EFFECTIVE_UPDATE_GRAD_ENERGY',needle,energy,flush=True)
    print('LOCAL_V_ZERO_ONLY_LEAF_CHANGED_SCANNED_LIVE_GRADIENT_OK',float(value),float(next_value),flush=True)

if __name__=='__main__': unittest.main()
