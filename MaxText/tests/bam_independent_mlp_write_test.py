"""Focused checks for sparse MLP-private dynamic write addresses."""
import functools
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers import attentions, quantizations
from layers.models import Transformer
from bam_mlp_write_test import MLPWriteTest

EXP='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'

class IndependentAddressTest(MLPWriteTest):
  def test_independent_parameters_health(self):
    c=self.config(EXP);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
    flat=flatten_dict(args[0].params)
    total=sum(int(np.prod(v.shape)) for v in flat.values())
    address=sum(int(np.prod(v.shape)) for p,v in flat.items() if any(n in p for n in ('mlp_address_down','mlp_address_up')))
    self.assertEqual(total,432096128);self.assertEqual(address,2632704)
    self.assertFalse(c.bam_dynamic_unembedding_read)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
    for l in range(18):
      tag=f'bam/concat/mlp_address_alignment/layer_{l:03d}/mean_cosine'
      self.assertEqual(tag in metrics['scalar'],(l+1)%3==2)
    print('INDEPENDENT_PARAMS_HEALTH_OK',total,address,flush=True)
  def test_independent_forward_gradient_and_outer(self):
    c=self.config(EXP,dtype='float32',weight_dtype='float32')
    c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,
        base_num_decoder_layers=3,num_decoder_layers=3,mlp_dim=64,
        mlp_dim_by_block=[64]*3,vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,
        bam_mlp_write_address_rank=16)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    model=Transformer(c,mesh,quantizations.configure_quantization(c))
    tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens)
    call=(tokens,pos,tokens,mask,mask)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
      def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])
      val,grad=jax.jit(jax.value_and_grad(loss))(params)
      self.assertTrue(np.isfinite(float(val)))
      self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad)))
      flat=flatten_dict(nn.unbox(grad))
      for name in ('mlp_address_down','mlp_address_up','mlp_write_gate'):
        self.assertGreater(sum(float(jnp.sum(g*g)) for p,g in flat.items() if name in p),0,name)
      c.get_keys().update(bam_record_concat_health=False,bam_lambda_decay=.73)
      a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
      x=jax.random.normal(jax.random.key(10),(1,4,150));o=jax.random.normal(jax.random.key(11),(1,4,2,75));m=jax.random.normal(jax.random.key(12),(1,4,75,32));y=jax.random.normal(jax.random.key(13),o.shape);addr=jax.random.normal(jax.random.key(14),(1,4,2,32));g=jnp.full((1,4,2),.23)
      v=a.init(jax.random.key(15),o,x,m,method=a._write)
      old,_=a.apply(v,o,x,m,method=a._write);f=a.apply(v,o,x,method=a._deferred_write_factors)
      yn=a.apply(v,y,method=lambda mod,z:mod.write_data_norm(z));an=a.apply(v,addr,method=lambda mod,z:mod.write_address_norm(z))
      ref=old+jnp.einsum('btnk,btnv->btkv',g[...,None]*yn,an)
      for implementation in ('dot','mul_reduce'):
        c.get_keys()['bam_write_outer_implementation']=implementation
        result=a.apply(v,y,g,f,m,independent_address=addr,method=a.merge_mlp_write)
        np.testing.assert_allclose(result,ref,rtol=2e-5,atol=2e-5)
    print('INDEPENDENT_FORWARD_GRAD_OUTER_OK',float(val),flush=True)

if __name__=='__main__':
  import unittest
  unittest.main()
