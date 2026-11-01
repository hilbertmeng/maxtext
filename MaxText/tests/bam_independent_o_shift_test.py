"""Independent O time mixing: kernel/reference, tied gates, whole-model gradients."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

import max_utils
import pyconfig
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import bam_pallas_v8 as v8
from layers.models import Transformer

H = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedTruePile'
RUN = H.replace('TruePile','IndependentOMShiftTruePile')


def inputs(wide=False, independent=True):
  b,t,n,v,k,rs = (1,128,2,32,75,18) if wide else (2,256,2,16,32,8)
  keys = iter(jax.random.split(jax.random.key(40),24))
  rand = lambda shape: jax.random.normal(next(keys),shape)
  m = rand((b,t,v,k))
  sw = rand((4*n,v))*.2
  xs = (rand((b,t,n,v)),rand((b,t,n)),rand((b,t,n,v)),rand((b,t,n)),
        rand((b,t,(2*n if independent else n),v)),rand((b,t,n)),rand((b,t,n)),
        rand((b,t,n,rs)),rand((b,t,n,rs)))
  sg = tuple(jax.nn.sigmoid(rand((b,t,n))) for _ in range(4))
  valid = jnp.ones((b,t),bool).at[:,0].set(False).at[:,64].set(False)
  gc = jax.nn.sigmoid(rand((b,t,v)))*valid[...,None]
  go = jax.nn.sigmoid(rand((b,t,v)))*valid[...,None]
  return m,sw,xs,sg,gc,go


def reference(m,sw,xs,sg,gc,go,**kwargs):
  rq,lq,rk,lk,rr,lo,lv,qs,ks = xs
  n = rq.shape[2]
  rv,ro = jnp.split(rr,2,axis=2) if rr.shape[2] == 2*n else (rr,rr)
  rms = lambda r:r*jax.lax.rsqrt(jnp.mean(r*r,axis=-1,keepdims=True)+1e-4)
  raw = (rq,rk,rv,ro)
  gates = (lq,lk,lv,lo)
  st = sw.reshape(4,n,sw.shape[-1])
  weights = [.1*jax.nn.sigmoid(logit)[...,None]*rms(r)+st[i]*sg[i][...,None]
             for i,(r,logit) in enumerate(zip(raw,gates))]
  prev = jnp.concatenate((m[:,:1],m[:,:-1]),axis=1)
  mc = m+gc[...,None]*(prev-m)
  mo = m+go[...,None]*(prev-m)
  out = [jnp.einsum('btvk,btnv->btnk',matrix,weight)
         for matrix,weight in zip((mc,mc,mc,mo),weights)]
  local = m.shape[-1]-qs.shape[-1] if m.shape[-1]==32 else m.shape[-1]
  return ((jnp.concatenate((out[0][...,:local],qs),-1)*.125,
           jnp.concatenate((out[1][...,:local],ks),-1),out[2],out[3],m))


def kernel(m,sw,xs,sg,gc,go,tile=64):
  cols = m.shape[-1]-xs[-1].shape[-1] if m.shape[-1]==32 else m.shape[-1]
  return v8.read_tm(m,sw,*xs,qk_cols=cols,read_epsilon=1e-4,key_scale=.1,
                    q_scale=.125,forward_tile=tile,reverse_tile=tile,
                    interpret=True,static_gates=sg,shift=gc,o_shift=go)


class IndependentOShiftTest(MLPWriteTest,unittest.TestCase):
  def assert_relative(self,a,b,tol,label):
    a,b=np.asarray(a),np.asarray(b)
    error=np.max(np.abs(a-b))/(np.max(np.abs(b))+1e-12)
    self.assertLess(error,tol,(label,float(error)))

  def test_kernel_forward_all_gradients_and_tile_boundaries(self):
    for independent in (False,True):
      data=inputs(independent=independent)
      cts=tuple(jax.random.normal(jax.random.key(100+i),y.shape) for i,y in enumerate(reference(*data)))
      def loss(fn,*args):return sum(jnp.sum(y*ct) for y,ct in zip(fn(*args),cts))
      for tile in (64,256):
        fn=lambda *a:kernel(*a,tile=tile)
        for i,(a,b) in enumerate(zip(fn(*data),reference(*data))):
          self.assert_relative(a,b,2e-5,('forward',independent,tile,i))
        ga=jax.grad(lambda *a:loss(fn,*a),argnums=(0,1,2,3,4,5))(*data)
        gb=jax.grad(lambda *a:loss(reference,*a),argnums=(0,1,2,3,4,5))(*data)
        for (path,a),b in zip(jax.tree_util.tree_flatten_with_path(ga)[0],jax.tree.leaves(gb)):
          self.assert_relative(a,b,3e-5,('gradient',independent,tile,str(path)))

  def test_tied_gates_equal_original_kernel_and_gradients(self):
    m,sw,xs,sg,gc,_=inputs()
    def tied(m,sw,xs,sg,g):return kernel(m,sw,xs,sg,g,g)
    def original(m,sw,xs,sg,g):
      return v8.read_tm(m,sw,*xs,qk_cols=24,read_epsilon=1e-4,key_scale=.1,
                        q_scale=.125,forward_tile=64,reverse_tile=64,
                        interpret=True,static_gates=sg,shift=g)
    for a,b in zip(tied(m,sw,xs,sg,gc),original(m,sw,xs,sg,gc)):
      self.assert_relative(a,b,2e-5,'tied output')
    cts=tuple(jax.random.normal(jax.random.key(130+i),y.shape) for i,y in enumerate(tied(m,sw,xs,sg,gc)))
    def loss(fn,*a):return sum(jnp.sum(o*c) for o,c in zip(fn(*a),cts))
    ga=jax.grad(lambda *a:loss(tied,*a),argnums=(0,1,2,3,4))(m,sw,xs,sg,gc)
    gb=jax.grad(lambda *a:loss(original,*a),argnums=(0,1,2,3,4))(m,sw,xs,sg,gc)
    for a,b in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):
      self.assert_relative(a,b,3e-5,'tied gradient sum')

  def test_qk93_and_gate_influence_isolation(self):
    data=inputs(wide=True)
    m,sw,xs,sg,gc,go=data
    a,b=kernel(*data),reference(*data)
    for u,v in zip(a,b):self.assert_relative(u,v,2e-5,'QK93')
    changed=kernel(m,sw,xs,sg,gc,go*.1)
    for i in (0,1,2,4):self.assert_relative(a[i],changed[i],1e-7,'QKV+carry independent of O shift')
    self.assertGreater(float(jnp.max(jnp.abs(a[3]-changed[3]))),.01)
    dc,do=jax.grad(lambda c,o:jnp.sum(kernel(m,sw,xs,sg,c,o)[3]**2),argnums=(0,1))(gc,go)
    np.testing.assert_array_equal(dc,jnp.zeros_like(dc))
    self.assertGreater(float(jnp.linalg.norm(do)),0)

  def test_formal_budget_and_shared_scan(self):
    c=self.config(RUN)
    c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,*_=train_compile.get_shaped_inputs(mesh,c)
    flat=flatten_dict(nn.unbox(args[0].params))
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432091186)
    self.assertEqual(c.mlp_dim,2499)
    self.assertEqual(c.bam_last_mlp_dim,3893)
    self.assertEqual(c.bam_memory_mlp_dim,1128)
    self.assertTrue(c.scan_layers and c.bam_final_local_layer_count==1)
    for part in ('layers','final_local_layer'):
      self.assertTrue(any(part in p and 'O_M_shift_gate' in p for p in flat))
    self.assertFalse(any('final_local_layer' in p and 'memory_mlp' in p for p in flat))

  def test_full_model_native_pallas_all_gradients(self):
    c=self.config(RUN,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,
        base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,bam_pallas_write=False,bam_splash_attention=False,
        mlp_dim=32,bam_memory_mlp_dim=16,bam_last_mlp_dim=32,
        bam_layer_modes=['local_qk+local_v+local_o']*3,bam_attention_write_layers=[0,1],
        bam_mlp_write_layers=None,emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,max_target_length=128,
        max_prefill_predict_length=128,query_chunk_size=128)
    pc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_pallas_core=True,
        bam_pallas_write=True,bam_splash_attention=True,bam_pallas_interpret=True,
        bam_pallas_fused_inputs=True)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    native,fused=Transformer(c,mesh,quant=None),Transformer(pc,mesh,quant=None)
    tok=jax.random.randint(jax.random.key(150),(1,128),0,32)
    pos=(jnp.arange(128)%64)[None]
    seg=jnp.where(jnp.arange(128)<64,1,2)[None]
    original=splash.make_splash_mha
    def interpret(*a,**kw):return original(*a,**dict(kw,interpret=True))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=interpret):
      p,pp=(nn.unbox(m.init(jax.random.key(151),tok,pos,seg,tok)) for m in (native,fused))
      for path,value in flatten_dict(p).items():np.testing.assert_array_equal(value,flatten_dict(pp)[path])
      def loss(m,p):
        y=m.apply(p,tok,pos,seg,tok)[0]
        return jnp.mean(y*y),y
      (_,a),ga=jax.value_and_grad(lambda p:loss(native,p),has_aux=True)(p)
      (_,b),gb=jax.value_and_grad(lambda p:loss(fused,p),has_aux=True)(p)
    np.testing.assert_allclose(a,b,rtol=4e-4,atol=4e-6)
    for path,value in flatten_dict(ga).items():
      np.testing.assert_allclose(value,flatten_dict(gb)[path],rtol=3e-3,atol=5e-6,err_msg=str(path))
    self.assertGreater(sum(float(jnp.sum(v*v)) for p,v in flatten_dict(ga).items() if 'O_M_shift_gate' in p),0)
    print('INDEPENDENT_O_SHIFT_MODEL_ALL_GRADIENTS_OK',flush=True)


if __name__ == '__main__':unittest.main()
