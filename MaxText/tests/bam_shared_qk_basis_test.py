"""Shared Q/K basis cache: explicit tied reference and packed initialization parity."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import core
from flax.traverse_util import flatten_dict
import max_utils
import train
import train_compile
from flax.linen import partitioning
from bam_local_fetch_test import LocalFetchTest
from layers.attentions import (factorized_head_bam_read, _basis_gram,
    _contract_bam_read_sides, _LocalReadArm, _packed_local_layout, _packed_local_arms_init, BamAttention)


class SharedBasisTest(unittest.TestCase):
  config = LocalFetchTest.config

  def test_xl_module_and_scan_shape(self):
    exp='BamXLIndependentLLFLocalQKRank4CFp32AlignedRow'
    configs=[self.config(exp),self.config(exp+'SharedBasis')]
    for c in configs:
      c.get_keys().update(dtype=jnp.float32, head_dim=128, base_emb_dim=256, emb_dim=256,
          bam_layer_modes=['local_qk+local_o','local_qk+local_o','local_qk+full']*2,
          base_num_decoder_layers=6,num_decoder_layers=6,vocab_size=128)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(configs[0]),configs[0].mesh_axes)
    modules=[BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=128,bam_k=64,bam_v=32,
        max_target_length=8,max_prefill_predict_length=8,mesh=mesh,
        attention_kernel='dot_product_chunk',dtype=jnp.float32,
        layer_mode='local_qk+local_o',attention_type=c.attention_type) for c in configs]
    x=jax.random.normal(jax.random.key(12),(1,8,256))
    m=jax.random.normal(jax.random.key(13),(1,8,64,32))
    args=(x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
    kw=dict(M_in=m,deterministic=True,layer_index=2)
    variables=[mod.init({'params':jax.random.key(14),'aqt':jax.random.key(15)},*args,**kw) for mod in modules]
    p0,p1=(core.unfreeze(v['params']) for v in variables)
    self.assertNotIn('W_lk_bias',p1)
    self.assertIn('W_lq_gate_b0',p1); self.assertIn('W_lk_gate_b0',p1)
    packed_name=configs[0].bam_local_packed_parameter_name or 'W_local_packed'
    arms=[_LocalReadArm(n,4,64,32,'effective_key',2,True,'both') for n in ('q','k','v')]
    layouts=[_packed_local_layout(arms,share)[0] for share in (False,True)]
    # Make the shared basis active and copy it into both reference slots.
    a=.006*jax.random.normal(jax.random.key(16),(256,384))
    for params,layout in zip((p0,p1),layouts):
      kernel=params[packed_name]['kernel']
      val=kernel.value
      for i in (0,1): val=val.at[:,layout[i][0]].set(a)
      params[packed_name]['kernel']=kernel.replace(value=val)
    for slot0,slot1 in zip(*layouts):
      for s0,s1 in zip(slot0,slot1):
        np.testing.assert_array_equal(p0[packed_name]['kernel'].value[:,s0],p1[packed_name]['kernel'].value[:,s1])
    for path,value in flatten_dict(p1).items():
      if path[0]==packed_name: continue
      for a0,a1 in zip(jax.tree.leaves(flatten_dict(p0)[path]),jax.tree.leaves(value)):
        np.testing.assert_array_equal(a0,a1)
    for y0,y1 in zip(jax.tree.leaves(modules[0].apply({'params':p0},*args,**kw)),
                     jax.tree.leaves(modules[1].apply({'params':p1},*args,**kw))):
      np.testing.assert_allclose(y0,y1,rtol=2e-5,atol=2e-5)
    wd=train.get_wd_tree(configs[1],p1)
    self.assertEqual(configs[1].wd_mults,[])
    self.assertIsNone(wd)  # None selects ordinary all-parameter AdamW decay.
    shaped,_,sharding,model=train_compile.get_shaped_inputs(mesh,configs[1])
    with mesh,partitioning.axis_rules(configs[1].logical_axis_rules):
      _,metrics=jax.eval_shape(lambda s,d,r:train.train_step(model,configs[1],sharding,s,d,r),*shaped)
    self.assertIn('learning/raw_grad_norm',metrics['scalar'])

  def test_cached_pair_forward_and_gradient(self):
    keys = jax.random.split(jax.random.key(1), 5)
    M = jax.random.normal(keys[0], (1,3,6,5))
    A = jax.random.normal(keys[1], (1,3,4,11)) * .03
    H = jax.random.normal(keys[2], (2,1,3,2,2,4))
    gate = jax.random.normal(keys[3], (2,1,3,2,2))
    x = jnp.zeros((1,3,8))
    for impl in ('dot', 'mul_reduce'):
      for placement in ('mix', 'output'):
        def pair(m,a,h,g,cached):
          row,col=jnp.split(a,[6],-1)
          cache=(_contract_bam_read_sides(m,m,row,col,'mul_reduce_btn','both'),
                 tuple(_basis_gram(z,impl) for z in (row,col))) if cached else None
          return tuple(factorized_head_bam_read(m,x,lambda _:a,lambda _:h[i],
              key_mode='rms_gate',key_gate_logits=g[i],key_scale=2.,rms_epsilon=1e-4,
              rank=4,rank_routing='effective_key',gram_implementation=impl,
              scale_placement=placement,basis_cache=cache) for i in range(2))
        for a,b in zip(jax.tree.leaves(pair(M,A,H,gate,False)),jax.tree.leaves(pair(M,A,H,gate,True))):
          np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
        def loss(m,a,h,g,cached):
          return sum(jnp.mean(y*y) for y in jax.tree.leaves(pair(m,a,h,g,cached)))
        old=jax.grad(loss,(0,1,2,3))(M,A,H,gate,False)
        new=jax.grad(loss,(0,1,2,3))(M,A,H,gate,True)
        for a,b in zip(jax.tree.leaves(old),jax.tree.leaves(new)):
          np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-5)

  def test_packed_slices_keep_independent_mix_initialization(self):
    arms=[_LocalReadArm(n,4,6,5,'effective_key',2,True,'both') for n in ('q','k','v')]
    old,width0=_packed_local_layout(arms)
    new,width1=_packed_local_layout(arms,True)
    self.assertEqual(width0-width1,44)
    self.assertEqual(new[0][0],new[1][0])
    def init(key,shape,dtype,*axes):
      return jax.random.normal(key,shape,dtype)*.006
    p0=_packed_local_arms_init(init,arms)(jax.random.key(2),(8,width0),jnp.float32)
    p1=_packed_local_arms_init(init,arms,share_qk_basis=True)(jax.random.key(2),(8,width1),jnp.float32)
    for slices0,slices1 in zip(old,new):
      for s0,s1 in zip(slices0,slices1):
        np.testing.assert_array_equal(p0[:,s0],p1[:,s1])
    self.assertFalse(np.array_equal(p1[:,new[0][2]],p1[:,new[1][2]]))


if __name__ == '__main__':
  unittest.main()
