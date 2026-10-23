"""Dedicated memory FFNs: branch isolation, live gradients, scan and v8t writes."""
import copy
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import fusion, quantizations, bam_pallas, normalizations
from layers.models import Transformer
from common_types import MODEL_MODE_TRAIN

A='BamMediumPropMShiftReadG050DedicatedMemoryMLPMHAWidthTruePile'
B='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1TruePile'

class DedicatedMemoryMLPTest(MLPWriteTest, unittest.TestCase):
  def test_formal_budgets_and_kernel_settings(self):
    for exp,normal,memory in [(A,207360000,31251600),(B,168843600,69768000)]:
      c=self.config(exp)
      c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params)
      total=sum(int(np.prod(v.shape)) for v in flat.values())
      ordinary=sum(int(np.prod(v.shape)) for p,v in flat.items() if 'mlp' in p)
      mem=sum(int(np.prod(v.shape)) for p,v in flat.items() if 'memory_mlp' in p)
      self.assertEqual((total,ordinary,mem),(432113280,normal,memory))
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs)
      self.assertEqual((c.bam_pallas_body,c.bam_pallas_vmem_mib),('v8t',48))
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertTrue(c.sa_use_fused_bwd_kernel)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)

  def tiny(self, exp=A, layers=6):
    c=self.config(exp, bam_splash_attention=False, base_emb_dim=150,
        base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=layers,
        base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,mlp_dim_by_block=[32,32],
        bam_memory_mlp_dim=16,bam_memory_mlp_last_write_dim=12,
        bam_final_local_mlp_dim=32,bam_last_mlp_dim=40,
        emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,bam_layer_modes=c.bam_layer_modes[:layers])
    return c

  def test_branches_are_functionally_separate(self):
    c=self.tiny();mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    mod=fusion.SubDecoderLayer(c,mesh,sliding_window_size=4,layer_inx=0)
    x=jax.random.normal(jax.random.key(1),(1,4,150))
    m=jax.random.normal(jax.random.key(2),(1,4,75,32))
    tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tokens)
    args=(x,seg,pos,tokens,None,True,MODEL_MODE_TRAIN,None)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(mod.init(jax.random.key(3),*args,M_in=m,is_global=True,layer_index=0)['params'])
      def call(p):return mod.apply({'params':p},*args,M_in=m,is_global=True,layer_index=0)
      y,mm=call(p)
      for branch,index in [('memory_mlp',0),('mlp',1)]:
        pp=copy.deepcopy(p);pp[branch]['wo']['kernel']=pp[branch]['wo']['kernel']*1.2+0.01
        yp,mp=call(pp)
        np.testing.assert_array_equal((yp,mp)[index],(y,mm)[index])
        self.assertGreater(float(jnp.max(jnp.abs((yp,mp)[1-index]-(y,mm)[1-index]))),0)
      for which,zero,live in [(0,'memory_mlp','mlp'),(1,'mlp','memory_mlp')]:
        grad=jax.grad(lambda pp:jnp.sum(call(pp)[which]**2))(p)
        self.assertEqual(sum(float(jnp.sum(g*g)) for g in jax.tree.leaves(grad[zero])),0)
        self.assertGreater(sum(float(jnp.sum(g*g)) for g in jax.tree.leaves(grad[live])),0)

  def test_scan_causality_live_memory_gradients_and_terminal_omission(self):
    for exp in [A,B]:
      c=self.tiny(exp);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quantizations.configure_quantization(c))
      tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      def call(p,t):return model.apply({'params':p},t,pos,t,seg,seg,enable_dropout=False)[0]
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=nn.unbox(model.init(jax.random.key(5),tok,pos,tok,seg,seg,enable_dropout=False)['params'])
        y=call(p,tok);changed=call(p,tok.at[:,3].set(7))
        np.testing.assert_allclose(y[:,:3],changed[:,:3],rtol=1e-6,atol=1e-6)
        val,grad=jax.jit(jax.value_and_grad(lambda pp:jnp.mean(call(pp,tok)**2)))(p)
        self.assertTrue(np.isfinite(float(val)))
        flat=flatten_dict(grad)
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
        for path,v in flat.items():
          if 'memory_mlp' in path:
            self.assertGreater(float(jnp.sum(v*v)),0,str(path))
        paths=['/'.join(path) for path in flatten_dict(p)]
        self.assertTrue(any('final_local_layer_0/block/memory_mlp/' in path for path in paths))
        self.assertFalse(any('final_local_layer_1/block/memory_mlp/' in path for path in paths))
        self.assertFalse(any('final_local_layer_1/block/mlp_address_' in path for path in paths))

  def test_v8t_two_group_forward_and_backward(self):
    keys=iter(jax.random.split(jax.random.key(9),10))
    rnd=lambda shape:jax.random.normal(next(keys),shape)
    m=rnd((1,128,75,32));groups=[(rnd((1,128,2,75)),rnd((1,128,2)),rnd((1,128,2,32))) for _ in range(2)]
    flat=tuple(a for group in groups for a in group);eps=1e-6
    def reference(m,*f):
      for i in range(0,len(f),3):
        c,g,a=f[i:i+3]
        c=normalizations.rms_norm(c,dtype=jnp.float32,epsilon=eps,statistics_dtype=jnp.float32)*jax.nn.sigmoid(g)[...,None]
        a=normalizations.rms_norm(a,dtype=jnp.float32,epsilon=eps,statistics_dtype=jnp.float32)
        m=m+jnp.einsum('btnk,btnv->btkv',c,a)
      return m
    def kernel(m,*f):
      gs=[f[i:i+3] for i in range(0,len(f),3)]
      return jnp.swapaxes(bam_pallas.write(jnp.swapaxes(m,-1,-2),gs,epsilon=eps,body='v8t',interpret=True),-1,-2)
    np.testing.assert_allclose(kernel(m,*flat),reference(m,*flat),rtol=2e-5,atol=2e-5)
    ct=rnd(m.shape);argnums=tuple(range(7))
    a=jax.grad(lambda *v:jnp.sum(kernel(*v)*ct),argnums=argnums)(m,*flat)
    b=jax.grad(lambda *v:jnp.sum(reference(*v)*ct),argnums=argnums)(m,*flat)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=1e-4,atol=1e-4)

if __name__=='__main__':unittest.main()
