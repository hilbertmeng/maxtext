"""No-W_O shape/budget and identity-W_O equivalence on both write schedules."""
import functools
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils, train, train_compile
from layers.models import Transformer
from layers import quantizations
from bam_mlp_write_test import MLPWriteTest, PREFIX

class NoWOTest(MLPWriteTest):
  def test_exact_budgets_and_health(self):
    for suffix, expected in [('StaticEveryThirdNoWOTruePile',432101696),
                             ('IndependentEveryThirdNoWOTruePile',432096128)]:
      c=self.config(PREFIX+suffix)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params)
      self.assertEqual(sum(int(np.prod(x.shape)) for x in flat.values()),expected)
      self.assertFalse(any('self_attention' in p and 'out' in p for p in flat))
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
      for l in range(18):
        self.assertEqual(f'bam/concat/mlp_write_gate/layer_{l:03d}/mean' in metrics['scalar'],l%3==1)
      print('NO_WO_BUDGET_HEALTH_OK',suffix,expected,flush=True)

  def test_identity_wo_equivalence_and_gradients(self):
    # Three layers exercise ordinary returns at0/2 and deferred writes at1.
    for suffix in ('StaticEveryThirdTruePile','IndependentEveryThirdTruePile'):
      c=self.config(PREFIX+suffix,dtype='float32',weight_dtype='float32')
      c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,
          base_num_decoder_layers=3,num_decoder_layers=3,mlp_dim=64,
          mlp_dim_by_block=[64]*3,vocab_size=32,
          bam_layer_modes=['local_qk+local_v+local_o']*3,
          bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,
          emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens)
      call=(tokens,pos,tokens,mask,mask)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        parent=Transformer(c,mesh,quantizations.configure_quantization(c))
        params=parent.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
        flat=flatten_dict(nn.unbox(params));out_paths=[]
        for p,x in flat.items():
          if 'self_attention' in p and p[-2:]==('out','kernel'):
            self.assertEqual(x.shape[-3:],(2,75,150))
            flat[p]=jnp.broadcast_to(jnp.eye(150,dtype=x.dtype).reshape(2,75,150),x.shape)
            out_paths.append(p)
        self.assertEqual(len(out_paths),3)
        identity_params=unflatten_dict(flat)
        def objective(model,p):
          return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])
        parent_value,parent_grad=jax.jit(jax.value_and_grad(lambda p:objective(parent,p)))(identity_params)
        c.get_keys()['bam_no_output_projection']=True
        child=Transformer(c,mesh,quantizations.configure_quantization(c))
        child_params=unflatten_dict({p:x for p,x in flat.items() if p not in out_paths})
        value,grad=jax.jit(jax.value_and_grad(lambda p:objective(child,p)))(child_params)
      self.assertTrue(np.isfinite(float(value)))
      np.testing.assert_allclose(value,parent_value,rtol=2e-5,atol=2e-5)
      child_flat=flatten_dict(grad);parent_flat=flatten_dict(parent_grad)
      for p,x in child_flat.items():
        self.assertTrue(np.all(np.isfinite(np.asarray(x))))
        np.testing.assert_allclose(x,parent_flat[p],rtol=2e-4,atol=2e-5,err_msg=str(p))
      print('NO_WO_IDENTITY_FORWARD_GRAD_OK',suffix,float(value),flush=True)

if __name__=='__main__':unittest.main()
