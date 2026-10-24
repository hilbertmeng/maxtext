"""Periodic standard-attention writers: budgets, no M reads, and v8t equivalence."""
import copy
import math
import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import pyconfig
import train_compile
from common_types import MODEL_MODE_TRAIN
from layers import fusion
from layers.models import Transformer
from bam_prop_test_utils import MLPWriteTest

PREFIX = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1MHAWriter'
ARMS = [PREFIX + f'Every{period}{kind}MemoryTruePile'
        for period in (4, 3) for kind in ('Keep', 'Drop')]

class MHAWriterTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, period=4, keep=True, length=4):
    c=self.config(PREFIX+f'Every{period}{"Keep" if keep else "Drop"}MemoryTruePile',
        bam_splash_attention=False,base_emb_dim=150,base_num_query_heads=2,
        base_num_kv_heads=2,base_num_decoder_layers=6,base_mlp_dim=32,vocab_size=32,
        dtype='float32',weight_dtype='float32')
    selected=[i for i in range(6) if (i%4==1 if period==4 else (i+1)%3==0)]
    writers=[i for i in range(4) if keep or i not in selected]
    c.get_keys().update(bam_pallas_core=False,max_target_length=length,
        max_prefill_predict_length=length,query_chunk_size=length,
        bam_mha_writer_layers=selected,
        bam_layer_modes=['write' if i in selected else 'local_qk+local_v+local_o' for i in range(6)],
        bam_attention_write_layers=list(range(4)),bam_mlp_write_layers=writers,
        bam_mlp_dim_by_layer=[32]*6,bam_memory_mlp_dim_by_layer=[16 if i in writers else 0 for i in range(6)],
        bam_memory_mlp_last_write_dim=0,bam_final_local_layer_count=2 if period==4 else 3,
        bam_final_local_mlp_dim=None,bam_last_mlp_dim=0,
        emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8)
    return c

  def test_formal_budgets_tree_and_local_ratios(self):
    for name in ARMS:
      c=self.config(name);c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      total=sum(math.prod(v.shape) for v in flat.values())
      self.assertLessEqual(abs(total-432113280),1800,(name,total))
      self.assertEqual(c.bam_attention_write_layers,list(range(16)))
      for i in (16,17):
        self.assertNotIn(i,c.bam_mlp_write_layers)
        self.assertEqual(c.bam_memory_mlp_dim_by_layer[i],0)
      for i in c.bam_mha_writer_layers:
        if i>=16:continue
        h=c.bam_memory_mlp_dim_by_layer[i]
        if 'Keep' in name:
          ratio=3600*c.bam_mlp_dim_by_layer[i]/(3600*h+458000)
          self.assertLess(abs(ratio-2),.005)
        else:self.assertEqual(h,0)
      for path in flat:
        text='/'.join(path)
        if '/final_local_layer_' in '/'+text:
          off=int(next(p for p in path if p.startswith('final_local_layer_')).split('_')[-1])
          i=18-c.bam_final_local_layer_count+off
          if i>=16:self.assertFalse(set(path)&{'memory_mlp','mlp_address_down','mlp_address_up','mlp_write_gate','P_loc_down','P_loc_up','W_gw'})
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)

  def test_standard_writer_does_not_read_m_and_branches_are_separate(self):
    for keep in (True,False):
      c=self.tiny(keep=keep);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      mod=fusion.SubDecoderLayer(c,mesh,quant=None,layer_inx=1)
      x=jax.random.normal(jax.random.key(1),(1,4,150));m=jax.random.normal(jax.random.key(2),(1,4,75,32))
      pos=jnp.arange(4)[None];seg=jnp.ones((1,4),jnp.int32);args=(x,seg,pos,seg,None,True,MODEL_MODE_TRAIN,None)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=nn.unbox(mod.init(jax.random.key(3),*args,M_in=m,is_global=True)['params'])
        call=lambda q,mm:mod.apply({'params':q},*args,M_in=mm,is_global=True)
        y,mm=call(p,m);y2,mm2=call(p,2*m)
        np.testing.assert_array_equal(y,y2)
        np.testing.assert_allclose(mm-m,mm2-2*m,rtol=2e-4,atol=1e-6)
        attn=p['self_attention'];self.assertEqual(attn['query']['kernel'].shape,(150,2,75))
        self.assertEqual(attn['key']['kernel'].shape,(150,2,75));self.assertEqual(attn['value']['kernel'].shape,(150,2,75))
        self.assertFalse(any('read' in q or 'static_' in q or 'M_shift' in q for q in attn))
        self.assertNotIn('W_R',attn)
        self.assertEqual('memory_mlp' in p,keep)
        if keep:
          changed=copy.deepcopy(p);changed['memory_mlp']['wo']['kernel']+=.01
          yp,mp=call(changed,m);np.testing.assert_array_equal(y,yp)
          self.assertGreater(float(jnp.max(jnp.abs(mp-mm))),0)
        grad=jax.grad(lambda mm:jnp.sum(call(p,mm)[0]))(m)
        np.testing.assert_array_equal(grad,jnp.zeros_like(m))

  def test_scan_causality_and_live_writers(self):
    for period in (4,3):
      for keep in (True,False):
        c=self.tiny(period,keep);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
        model=Transformer(c,mesh,quant=None);tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
          p=nn.unbox(model.init(jax.random.key(4),tok,pos,tok,seg,seg,enable_dropout=False)['params'])
          call=lambda q,t:model.apply({'params':q},t,pos,t,seg,seg,enable_dropout=False)[0]
          y=call(p,tok);z=call(p,tok.at[:,3].set(7));np.testing.assert_allclose(y[:,:3],z[:,:3],rtol=1e-6,atol=1e-6)
          grad=jax.jit(jax.grad(lambda q:jnp.mean(call(q,tok)**2)))(p)
        for path,v in flatten_dict(grad).items():
          self.assertTrue(np.isfinite(np.asarray(v)).all(),str(path))
          if 'memory_mlp' in path:self.assertGreater(float(jnp.sum(v*v)),0,str(path))
        for path in flatten_dict(p):
          text='/'.join(path)
          if 'final_local_layer_1/' in text and period==4 or 'final_local_layer_2/' in text and period==3:
            self.assertFalse(set(path)&{'full_m_read_down','M_shift_gate','P_loc_down','memory_mlp'})

  def test_pallas_keep_full_forward_and_all_gradients(self):self.check_pallas(True)
  def test_pallas_drop_full_forward_and_all_gradients(self):self.check_pallas(False)

  def check_pallas(self,keep):
    c=self.tiny(3,keep,length=128)
    pc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_pallas_core=True,
        bam_pallas_interpret=True,bam_pallas_body='v8t',bam_pallas_fused_inputs=True)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    xla=Transformer(c,mesh,quant=None);pallas=Transformer(pc,mesh,quant=None)
    tok=jax.random.randint(jax.random.key(20),(1,128),0,32);pos=jnp.arange(128)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(xla.init(jax.random.key(21),tok,pos,seg,tok))
      other=nn.unbox(pallas.init(jax.random.key(21),tok,pos,seg,tok))
      self.assertEqual(jax.tree.structure(p),jax.tree.structure(other))
      for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(other)):np.testing.assert_array_equal(a,b)
      leaves,tree=jax.tree.flatten(p);keys=jax.random.split(jax.random.key(22),len(leaves))
      p=jax.tree.unflatten(tree,[v+.01*jax.random.normal(k,v.shape,v.dtype) for v,k in zip(leaves,keys)])
      call=lambda model,q:model.apply(q,tok,pos,seg,tok,enable_dropout=False)[0].astype(jnp.float32)
      target=jax.random.normal(jax.random.key(23),call(xla,p).shape)
      def loss(model,q):
        y=call(model,q);return jnp.mean(y*target),y
      (lx,yx),gx=jax.value_and_grad(lambda q:loss(xla,q),has_aux=True)(p)
      (lp,yp),gp=jax.value_and_grad(lambda q:loss(pallas,q),has_aux=True)(p)
    np.testing.assert_allclose(yp,yx,rtol=1e-4,atol=1e-6)
    np.testing.assert_allclose(lp,lx,rtol=1e-4,atol=1e-7)
    worst=[]
    for (path,a),b in zip(jax.tree_util.tree_flatten_with_path(gx)[0],jax.tree.leaves(gp)):
      err=float(jnp.max(jnp.abs(a-b))/(jnp.max(jnp.abs(a))+1e-12));worst.append((err,jax.tree_util.keystr(path)))
    worst.sort(reverse=True);print('MHA_WRITER_PALLAS_WORST_GRAD',keep,worst[:5],flush=True)
    self.assertLess(worst[0][0],1e-3,worst[:5])

if __name__=='__main__':unittest.main()
