"""Independent reader, shared pre-attention write features, and exact refunds."""
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
from layers import fusion, quantizations
from layers.models import Transformer
from bam_prop_test_utils import MLPWriteTest

PREFIX = 'BamMediumPropMShiftReadG050DedicatedMemoryMLP'
ARMS = [(PREFIX + f'Ratio{ratio}to1SharedWritePreAttention{refund}RefundTruePile', ordinary, memory)
        for ratio, normal, mem in [(2, 2535, 1140), (3, 2852, 823)]
        for refund, ordinary, memory in [('Ordinary', normal + 85, mem), ('Memory', normal, mem + 85)]]

POST = PREFIX + 'Ratio2to1SharedWritePostAttentionOrdinaryRefundTruePile'
ARMS.append((POST,2620,1140))

class SharedWriteActivationTest(MLPWriteTest, unittest.TestCase):
  def tiny(self):
    c = self.config(ARMS[0][0], bam_splash_attention=False, base_emb_dim=150,
                    base_num_query_heads=2, base_num_kv_heads=2,
                    base_num_decoder_layers=6, base_mlp_dim=32, vocab_size=32,
                    dtype='float32', weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False, mlp_dim_by_block=[32,32],
        bam_memory_mlp_dim=16, bam_memory_mlp_last_write_dim=0,
        bam_final_local_mlp_dim=32, bam_last_mlp_dim=40,
        emb_bam_num_head=2, bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8, bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8, bam_layer_modes=c.bam_layer_modes[:6])
    return c

  def test_formal_four_budgets_and_independent_reader(self):
    for name, ordinary, memory in ARMS:
      c = self.config(name)
      c.get_keys().update(max_target_length=256, max_prefill_predict_length=256)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(nn.unbox(args[0].params))
      size = lambda v: math.prod(v.shape)
      self.assertEqual(sum(size(v) for v in flat.values()), 432114480)
      self.assertEqual(sum(size(v) for p,v in flat.items() if 'mlp' in p),
                       17 * 3600 * ordinary + 3600 * 3812)
      self.assertEqual(sum(size(v) for p,v in flat.items() if 'memory_mlp' in p),
                       17 * 3600 * memory)
      self.assertFalse(any('mlp_address_down' in p for p in flat))
      for branch in ['P_loc_down', 'full_m_read_down', 'mlp_address_up', 'mlp_write_gate']:
        self.assertTrue(any(branch in p for p in flat), branch)
      control_names = {'mlp_address_up','mlp_write_gate','mlp_write_gate_bias'}
      self.assertEqual(sum(size(v) for p,v in flat.items() if set(p) & control_names), 2563600)
      self.assertFalse(c.bam_share_read_attention_write_down)
      self.assertFalse(c.bam_share_mlp_write_down)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertEqual((c.bam_pallas_body,c.bam_pallas_vmem_mib,c.tensorboard_gcs_rotate_period),('v8t',48,500))
      c.get_keys()['bam_pallas_core'] = False
      xargs, *_ = train_compile.get_shaped_inputs(mesh, c)
      other = flatten_dict(nn.unbox(xargs[0].params))
      self.assertEqual({p:v.shape for p,v in flat.items()}, {p:v.shape for p,v in other.items()})

  def test_actual_activation_reuse_and_unchanged_gate_input(self):
    self.check_actual_activation(False)

  def test_post_attention_activation_and_identical_initial_parameters(self):
    self.check_actual_activation(True)

  def check_actual_activation(self,post):
    c=self.tiny()
    if post:c.get_keys()['bam_shared_write_activation_input']='post_attention'
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    block=fusion.SubDecoderLayer(c,mesh,quantizations.configure_quantization(c),layer_inx=0)
    x=jax.random.normal(jax.random.key(10),(1,4,150));M=jax.random.normal(jax.random.key(11),(1,4,75,32))
    pos=jnp.arange(4)[None];seg=jnp.ones((1,4),jnp.int32)
    args=(x,seg,pos,seg,None,True,MODEL_MODE_TRAIN,None)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=block.init(jax.random.key(12),*args,M_in=M)
      if post:
        pre=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_shared_write_activation_input='pre_attention')))
        other=fusion.SubDecoderLayer(pre,mesh,quantizations.configure_quantization(pre),layer_inx=0).init(jax.random.key(12),*args,M_in=M)
        left=flatten_dict(nn.unbox(p));right=flatten_dict(nn.unbox(other))
        self.assertEqual(set(left),set(right))
        for path in left:np.testing.assert_array_equal(left[path],right[path],err_msg=str(path))
      seen={}
      def capture(next_fun,args,kwargs,ctx):
        out=next_fun(*args,**kwargs)
        if ctx.method_name=='__call__' and ctx.module.name in (
            'P_loc_down','full_m_read_down','mlp_address_up','P_loc_up','mlp_write_gate','W_gw','memory_mlp'):
          seen.setdefault(ctx.module.name,[]).append((args[0],out))
        return out
      with nn.intercept_methods(capture):block.apply(p,*args,M_in=M)
    self.assertEqual(len(seen['P_loc_down']),1)
    shared=jax.nn.gelu(seen['P_loc_down'][0][1])
    np.testing.assert_array_equal(seen['P_loc_up'][0][0],shared)
    np.testing.assert_array_equal(seen['mlp_address_up'][0][0],shared)
    np.testing.assert_array_equal(seen['full_m_read_down'][0][0],seen['W_gw'][0][0])
    np.testing.assert_array_equal(seen['mlp_write_gate'][0][0],seen['memory_mlp'][0][0])
    expected=seen['mlp_write_gate'][0][0] if post else seen['W_gw'][0][0]
    np.testing.assert_array_equal(seen['P_loc_down'][0][0],expected)
    self.assertGreater(float(jnp.max(jnp.abs(seen['mlp_write_gate'][0][0]-seen['W_gw'][0][0]))),1e-5)

  def test_scanned_causality_and_live_both_write_gradients(self):
    self.check_scanned(False)

  def test_post_scanned_causality_and_live_both_write_gradients(self):
    self.check_scanned(True)

  def check_scanned(self,post):
    c=self.tiny()
    if post:c.get_keys()['bam_shared_write_activation_input']='post_attention'
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    model=Transformer(c,mesh,quantizations.configure_quantization(c))
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(model.init(jax.random.key(13),tok,pos,tok,seg,seg,enable_dropout=False)['params'])
      def call(q,t):return model.apply({'params':q},t,pos,t,seg,seg,enable_dropout=False)[0]
      y=call(p,tok);yp=call(p,tok.at[:,3].set(9))
      np.testing.assert_allclose(y[:,:3],yp[:,:3],rtol=1e-6,atol=1e-6)
      val,g=jax.jit(jax.value_and_grad(lambda q:jnp.mean(call(q,tok)**2)))(p)
    self.assertTrue(np.isfinite(float(val)))
    flat=flatten_dict(g)
    self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
    for branch in ['P_loc_down','P_loc_up','full_m_read_down','mlp_address_up','memory_mlp','mlp_write_gate']:
      leaves=[v for p,v in flat.items() if branch in p]
      self.assertTrue(leaves,branch)
      self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves),0.,branch)

  def test_pallas_fused_logits_all_gradients_and_projection_count(self):
    self.check_pallas(False)

  def test_post_pallas_fused_logits_all_gradients_and_projection_count(self):
    self.check_pallas(True)

  def check_pallas(self,post):
    c=self.config('BamPallasMShiftTinyXlaTest', vocab_size=32)
    c.get_keys().update(max_target_length=128,max_prefill_predict_length=128,query_chunk_size=128,
        bam_share_attention_mlp_write_activation=True,
        bam_memory_mlp_dim=16,bam_mlp_write_every=1,bam_mlp_write_offset=0,
        bam_skip_final_mlp_write=True,bam_mlp_write_address_rank=32,
        bam_shared_write_activation_input='post_attention' if post else 'pre_attention')
    pc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),
        bam_pallas_core=True,bam_pallas_interpret=True,bam_pallas_body='v8t',bam_pallas_fused_inputs=True)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    xla=Transformer(c,mesh,quant=None);pallas=Transformer(pc,mesh,quant=None)
    tok=jax.random.randint(jax.random.key(20),(1,128),0,32)
    pos=jnp.arange(128)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(xla.init(jax.random.key(21),tok,pos,seg,tok))
      leaves,tree=jax.tree.flatten(p)
      keys=jax.random.split(jax.random.key(22),len(leaves))
      p=jax.tree.unflatten(tree,[v+.01*jax.random.normal(k,v.shape,v.dtype) for v,k in zip(leaves,keys)])
      def call(model,q):
        return model.apply(q,tok,pos,seg,tok,enable_dropout=False)[0].astype(jnp.float32)
      target=jax.random.normal(jax.random.key(23),call(xla,p).shape)
      def loss(model,q):
        y=call(model,q)
        return jnp.mean(y*target),y
      (lx,yx),gx=jax.value_and_grad(lambda q:loss(xla,q),has_aux=True)(p)
      seen=[];direct=[]
      def capture(next_fun,args,kwargs,ctx):
        if ctx.method_name=='_fused_input_projections' and 'full_m_read_down' in args[1]:seen.append(args[1])
        if ctx.method_name=='__call__' and ctx.module.name=='P_loc_down':
          if not post:raise AssertionError('Pre-attention fused apply recomputed P_loc_down')
          direct.append(ctx.module.scope.path)
        return next_fun(*args,**kwargs)
      with nn.intercept_methods(capture):
        (lp,yp),gp=jax.value_and_grad(lambda q:loss(pallas,q),has_aux=True)(p)
    self.assertGreaterEqual(len(seen),4)  # remat replays forward projections in backward
    self.assertTrue(all(names.count('full_m_read_down')==1 for names in seen))
    if post:
      self.assertTrue(any(names.count('P_loc_down')==0 for names in seen))
      self.assertTrue(any(names.count('P_loc_down')==1 for names in seen))  # terminal writer remains pre-attention
      self.assertGreaterEqual(len(direct),3)
    else:self.assertTrue(all(names.count('P_loc_down')==1 for names in seen))
    np.testing.assert_allclose(yp,yx,rtol=1e-4,atol=1e-6)
    np.testing.assert_allclose(lp,lx,rtol=1e-4,atol=1e-7)
    worst=[]
    for (path,a),b in zip(jax.tree_util.tree_flatten_with_path(gx)[0],jax.tree.leaves(gp)):
      err=float(jnp.max(jnp.abs(a-b))/(jnp.max(jnp.abs(a))+1e-12))
      worst.append((err,jax.tree_util.keystr(path)))
    worst.sort(reverse=True)
    print('SHARED_WRITE_PALLAS_WORST_GRAD',worst[:5])
    self.assertLess(worst[0][0],1e-3,worst[:5])

if __name__=='__main__':unittest.main()
