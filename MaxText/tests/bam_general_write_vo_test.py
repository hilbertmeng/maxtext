"""Generalized writes with independent-VO reads: budget, math and mixed-backend parity."""
import copy
import math
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,pyconfig,train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import attentions
from layers.models import Transformer

RUN='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysGeneralWriteTruePile'
PARENT='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysTruePile'

class GeneralWriteVOTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,length=128):
    c=self.config(RUN,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,
        base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32',
        bam_splash_attention=False)
    c.get_keys().update(bam_pallas_core=False,bam_memory_mlp_dim=16,bam_last_mlp_dim=32,
        bam_layer_modes=['local_qk+local_v+local_o']*3,bam_attention_write_layers=[0,1],
        bam_mlp_write_layers=[0,1],bam_write_v_bottleneck_dim=8,emb_bam_num_head=2,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,
        max_target_length=length,max_prefill_predict_length=length,query_chunk_size=length)
    return c

  def test_full_budget_and_initialization(self):
    c=self.config(RUN);c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
    total=sum(math.prod(v.shape) for v in flat.values());self.assertEqual(total,432113344)
    names={'general_attention_static_address','general_attention_static_gate','general_attention_static_gate_bias',
           'general_mlp_static_address','general_mlp_static_gate','general_mlp_static_gate_bias'}
    added={p:v for p,v in flat.items() if set(p)&names}
    self.assertEqual(sum(math.prod(v.shape) for v in added.values()),670752)
    self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2507,1120,3905))
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_write)
    self.assertTrue(c.scan_layers and not c.bam_pair_scan)
    self.assertEqual(c.bam_attention_write_layers,list(range(17)))
    self.assertTrue(c.bam_skip_final_mlp_write)
    self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
    for path in added:self.assertNotIn('final_local_layer',path)
    c.get_keys()['bam_general_matrix_write']=False
    old,*_=train_compile.get_shaped_inputs(mesh,c)
    self.assertEqual(sum(math.prod(v.shape) for v in flatten_dict(nn.unbox(old[0].params)).values()),total-670752)
    print('GENERAL_WRITE_VO_BUDGET_OK',total,'extra',670752,'parent_delta',1152,flush=True)

  def test_reference_formula_and_all_gradients(self):
    c=self.tiny(length=4);keys=iter(jax.random.split(jax.random.key(8),20))
    rand=lambda sh:jax.random.normal(next(keys),sh,jnp.float32)
    m=rand((1,4,75,32));groups=tuple((rand((1,4,2,75)),rand((1,4,2)),rand((1,4,2,32)),
                                     rand((1,4,2)),rand((2,32))*.1) for _ in range(2))
    from layers import initializers
    eps=initializers.get_bam_write_epsilon(c)
    norm=lambda x:x*jax.lax.rsqrt(jnp.mean(x*x,axis=-1,keepdims=True)+eps)
    def ref(m,groups):
      out=m
      for u,ld,a,ls,s in groups:
        ud=norm(u);ad=norm(a)
        out=out+jnp.einsum('btnk,btnv->btkv',jax.nn.sigmoid(ld)[...,None]*ud,ad)
        out=out+jnp.einsum('btnk,btnv->btkv',1.1*jax.nn.sigmoid(ls)[...,None]*ud,
                          jnp.broadcast_to(s,a.shape))
      return out
    fn=lambda m,g:attentions.native_bam_write(m,g,c)
    np.testing.assert_allclose(fn(m,groups),ref(m,groups),rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(jnp.swapaxes(attentions.native_bam_write(jnp.swapaxes(m,-1,-2),groups,c,layout='btvk'),-1,-2),fn(m,groups),rtol=1e-6,atol=1e-6)
    ct=rand(m.shape)
    ga=jax.grad(lambda m,g:jnp.sum(fn(m,g)*ct),argnums=(0,1))(m,groups)
    gb=jax.grad(lambda m,g:jnp.sum(ref(m,g)*ct),argnums=(0,1))(m,groups)
    for a,b in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):
      np.testing.assert_allclose(a,b,rtol=2e-4,atol=3e-5)
    print('GENERAL_WRITE_POST_RMS_ALL_GRADIENTS_OK',flush=True)

  def test_read_only_pallas_matches_native_logits_and_all_gradients(self):
    c=self.tiny();pc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),
        bam_pallas_core=True,bam_pallas_interpret=True,bam_pallas_write=False)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    native=Transformer(c,mesh,quant=None);mixed=Transformer(pc,mesh,quant=None)
    tok=jax.random.randint(jax.random.key(20),(1,128),0,32)
    seg=jnp.where(jnp.arange(128)<64,1,2)[None];pos=(jnp.arange(128)%64)[None]
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(native.init(jax.random.key(21),tok,pos,seg,tok))
      pp=nn.unbox(mixed.init(jax.random.key(21),tok,pos,seg,tok))
      self.assertEqual(jax.tree.structure(p),jax.tree.structure(pp))
      for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(pp)):np.testing.assert_array_equal(a,b)
      initial=native.apply(p,tok,pos,seg,tok,enable_dropout=False)[0]
      c.get_keys()['bam_general_matrix_write']=False
      parent=nn.unbox(native.init(jax.random.key(21),tok,pos,seg,tok))
      for path,v in flatten_dict(parent).items():np.testing.assert_array_equal(v,flatten_dict(p)[path],err_msg=str(path))
      np.testing.assert_allclose(initial,native.apply(p,tok,pos,seg,tok,enable_dropout=False)[0],rtol=2e-5,atol=2e-6)
      for path,v in flatten_dict(p).items():
        if path[-1] in ('general_attention_static_address','general_mlp_static_address'):
          np.testing.assert_array_equal(v,0)
        if path[-1] in ('general_attention_static_gate_bias','general_mlp_static_gate_bias'):
          np.testing.assert_allclose(1.1*jax.nn.sigmoid(v),.01,rtol=2e-6,atol=1e-7)
      c.get_keys()['bam_general_matrix_write']=True
      leaves,tree=jax.tree.flatten(p);ks=jax.random.split(jax.random.key(22),len(leaves))
      p=jax.tree.unflatten(tree,[v+.01*jax.random.normal(k,v.shape,v.dtype) for v,k in zip(leaves,ks)])
      call=lambda mod,q:mod.apply(q,tok,pos,seg,tok,enable_dropout=False)[0].astype(jnp.float32)
      target=jax.random.normal(jax.random.key(23),call(native,p).shape)
      def loss(mod,q):
        y=call(mod,q);return jnp.mean(y*target),y
      (ln,yn),gn=jax.value_and_grad(lambda q:loss(native,q),has_aux=True)(p)
      (lp,yp),gp=jax.value_and_grad(lambda q:loss(mixed,q),has_aux=True)(p)
    np.testing.assert_allclose(yp,yn,rtol=1e-4,atol=1e-6)
    worst=[]
    for (path,a),b in zip(jax.tree_util.tree_flatten_with_path(gn)[0],jax.tree.leaves(gp)):
      worst.append((float(jnp.max(jnp.abs(a-b))/(jnp.max(jnp.abs(a))+1e-12)),jax.tree_util.keystr(path)))
    worst.sort(reverse=True);self.assertLess(worst[0][0],1e-3,worst[:5])
    for name in ('general_attention_static_address','general_mlp_static_address','general_attention_static_gate','general_mlp_static_gate'):
      selected=[v for path,v in flatten_dict(gp).items() if name in path]
      self.assertTrue(selected,name);self.assertGreater(sum(float(jnp.sum(v*v)) for v in selected),0,name)
    print('GENERAL_WRITE_READ_ONLY_PALLAS_ALL_GRADIENTS_OK',worst[:3],flush=True)

  def _write_kernel_case(self,b,seed):
    from layers import bam_pallas_v8, initializers
    c=self.tiny();eps=initializers.get_bam_write_epsilon(c)
    keys=iter(jax.random.split(jax.random.key(seed),40))
    rand=lambda sh,s=1.:s*jax.random.normal(next(keys),sh,jnp.float32)
    t,v,k=256,32,75
    general=lambda n:(rand((b,t,n,k)),rand((b,t,n)),rand((b,t,n,v)),rand((b,t,n)),rand((n,v),.5))
    plain=lambda n:(rand((b,t,n,k)),rand((b,t,n)),rand((b,t,n,v)))
    ker=lambda m,g:bam_pallas_v8.write_tm(m,g,epsilon=eps,forward_tile=64,reverse_tile=128,interpret=True)
    ref=lambda m,g:attentions.native_bam_write(m,g,c,layout='btvk')
    cases=(('general',(general(2),general(3))),('mixed',(plain(2),general(3))),('one',(general(1),)))
    return c,[(name,rand((b,t,v,k)),groups,rand((b,t,v,k))) for name,groups in cases],ker,ref

  def _check_write_kernel(self,cases,ker,ref,wrap=lambda f:f):
    for name,m,groups,ct in cases:
      y=wrap(ker)(m,groups);np.testing.assert_allclose(y,ref(m,groups),rtol=2e-5,atol=2e-5,err_msg=name)
      ga=wrap(jax.grad(lambda m,g:jnp.sum(ker(m,g)*ct),argnums=(0,1)))(m,groups)
      gb=jax.grad(lambda m,g:jnp.sum(ref(m,g)*ct),argnums=(0,1))(m,groups)
      for (path,a),bb in zip(jax.tree_util.tree_flatten_with_path(ga)[0],jax.tree.leaves(gb)):
        np.testing.assert_allclose(a,bb,rtol=2e-4,atol=2e-4*float(jnp.max(jnp.abs(bb))),
                                   err_msg=name+jax.tree_util.keystr(path))

  def test_v8t_write_kernel_matches_native_all_gradients(self):
    _,cases,ker,ref=self._write_kernel_case(2,31);self._check_write_kernel(cases,ker,ref)
    print('GENERAL_WRITE_V8T_KERNEL_ALL_GRADIENTS_OK',flush=True)

  def test_v8t_write_kernel_batch_sharded_static_gradients(self):
    # Replicated static addresses inside shard_map: their gradients must sum over every batch shard.
    if jax.device_count()<8:self.skipTest('needs XLA_FLAGS=--xla_force_host_platform_device_count=8')
    c,cases,ker,ref=self._write_kernel_case(8,32)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    def wrap(f):
      def run(*xs):
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):return jax.jit(f)(*xs)
      return run
    self._check_write_kernel(cases,ker,ref,wrap)
    print('GENERAL_WRITE_V8T_KERNEL_SHARDED_OK',flush=True)

  def test_full_pallas_matches_native_logits_and_all_gradients(self):
    c=self.tiny();pc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),
        bam_pallas_core=True,bam_pallas_interpret=True,bam_pallas_write=True)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    native=Transformer(c,mesh,quant=None);fused=Transformer(pc,mesh,quant=None)
    tok=jax.random.randint(jax.random.key(40),(2,128),0,32)
    seg=jnp.where(jnp.arange(128)<64,1,2)[None].repeat(2,0);pos=(jnp.arange(128)%64)[None].repeat(2,0)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(native.init(jax.random.key(41),tok,pos,seg,tok))
      pp=nn.unbox(fused.init(jax.random.key(41),tok,pos,seg,tok))
      for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(pp)):np.testing.assert_array_equal(a,b)
      leaves,tree=jax.tree.flatten(p);ks=jax.random.split(jax.random.key(42),len(leaves))
      p=jax.tree.unflatten(tree,[v+.01*jax.random.normal(k,v.shape,v.dtype) for v,k in zip(leaves,ks)])
      call=lambda mod,q:mod.apply(q,tok,pos,seg,tok,enable_dropout=False)[0].astype(jnp.float32)
      target=jax.random.normal(jax.random.key(43),call(native,p).shape)
      def loss(mod,q):
        y=call(mod,q);return jnp.mean(y*target),y
      (ln,yn),gn=jax.value_and_grad(lambda q:loss(native,q),has_aux=True)(p)
      (lp,yp),gp=jax.value_and_grad(lambda q:loss(fused,q),has_aux=True)(p)
    np.testing.assert_allclose(yp,yn,rtol=1e-4,atol=1e-6)
    worst=[]
    for (path,a),b in zip(jax.tree_util.tree_flatten_with_path(gn)[0],jax.tree.leaves(gp)):
      worst.append((float(jnp.max(jnp.abs(a-b))/(jnp.max(jnp.abs(a))+1e-12)),jax.tree_util.keystr(path)))
    worst.sort(reverse=True);self.assertLess(worst[0][0],1e-3,worst[:5])
    for name in ('general_attention_static_address','general_mlp_static_address','general_attention_static_gate','general_mlp_static_gate'):
      selected=[v for path,v in flatten_dict(gp).items() if name in path]
      self.assertTrue(selected,name);self.assertGreater(sum(float(jnp.sum(v*v)) for v in selected),0,name)
    print('GENERAL_WRITE_FULL_PALLAS_ALL_GRADIENTS_OK',worst[:3],flush=True)

if __name__=='__main__':unittest.main()
