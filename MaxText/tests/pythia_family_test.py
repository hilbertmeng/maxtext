"""Pythia MediumProp family: full-size accounting and circuit regression."""
import contextlib,io,json,math,unittest,copy
from types import SimpleNamespace
import pyconfig
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from pythia_test import PythiaTest
from layers.attentions import mix_bam_previous_token, bam_m_shift_health

ARMS=['PythiaMediumPropT2048','MuddPythiaMediumPropT2048','BamPythiaMediumPropT2048MShiftReadG050ReadAttentionSharedR256']

class PythiaFamilyTest(PythiaTest):
  def test_fusion_matches_pythia_parallel_circuit(self):
    c=self.config(scan_layers=False);m,mesh,p=self.init(c);p=nn.unbox(p)
    fc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),pythia_use_fusion=True)))
    fm,_=self.model(fc);fp=copy.deepcopy(p)
    for i in range(c.num_decoder_layers):
      layer=fp['decoder'][f'layers_{i}']
      layer['pre_self_attention_layer_norm']=layer.pop('input_layernorm')
      layer['post_self_attention_layer_norm']=layer.pop('post_attention_layernorm')
      fp['decoder'][f'layers_{i}']={'block':layer}
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      expected=m.apply({'params':p},*self.inputs(),enable_dropout=False)[0]
      actual=fm.apply({'params':fp},*self.inputs(),enable_dropout=False)[0]
      np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
      g=jax.grad(lambda p:jnp.mean(m.apply({'params':p},*self.inputs(),enable_dropout=False)[0]))(p)
      fg=jax.grad(lambda p:jnp.mean(fm.apply({'params':p},*self.inputs(),enable_dropout=False)[0]))(fp)
      for i in range(c.num_decoder_layers):
        a=g['decoder'][f'layers_{i}'];b=fg['decoder'][f'layers_{i}']['block']
        for name in ('self_attention','mlp'):
          for x,y in zip(jax.tree.leaves(a[name]),jax.tree.leaves(b[name])):
            np.testing.assert_allclose(x,y,atol=3e-6,rtol=3e-5)

  def test_mudd_bam_gradients(self):
    for arm in ARMS[1:]:
      opts=dict(exp_class=arm,base_num_decoder_layers=3,base_mlp_dim=256,
          pythia_rotary_dim=4,remat_policy='full')
      if arm.startswith('Bam'):
        opts.update(base_emb_dim=300,base_num_query_heads=4,base_num_kv_heads=4,
            head_dim=75,pythia_rotary_dim=18,base_num_decoder_layers=6)
      c=self.config(**opts)
      patch = {} if not c.bam_enabled else dict(emb_bam_num_head=4,mlp_dim_by_block=[256,224,256],bam_local_full_m_read_bottleneck_dim=64,bam_write_v_bottleneck_dim=64,bam_mlp_write_address_rank=64,bam_splash_attention=False,bam_layer_modes=c.bam_layer_modes[:6])
      c=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),**patch)))
      m,mesh,p=self.init(c);p=nn.unbox(p)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        loss,g=jax.jit(jax.value_and_grad(lambda p:jnp.mean(m.apply({'params':p},*self.inputs(),enable_dropout=False)[0])))(p)
        self.assertTrue(np.isfinite(loss))
        for a in jax.tree.leaves(g):self.assertTrue(np.isfinite(a).all())
        # Every layer must receive an MLP gradient, including through growing history.
        flat=flatten_dict(g)
        mlp_grads=[v for k,v in flat.items() if k[-3:]==('mlp','wi','kernel')]
        self.assertTrue(mlp_grads)
        for a in mlp_grads:self.assertGreater(float(jnp.linalg.norm(a)),0.)
        if c.bam_enabled:
          paths=['/'.join(k) for k in flatten_dict(p)]
          self.assertTrue(any('full_m_read_down' in k for k in paths))
          self.assertFalse(any('P_loc_down' in k for k in paths))
          self.assertTrue(any('mlp_address_down' in k for k in paths))
          shift=[v for k,v in flat.items() if 'm_shift' in '/'.join(k).lower()]
          self.assertTrue(shift);self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in shift),0.)

  def test_boundaries_column_broadcast_and_nonrecursive_source(self):
    m=jnp.arange(6*3*2,dtype=jnp.float32).reshape(1,6,3,2)
    g=jnp.broadcast_to(jnp.array([.25,.75]),(1,6,2))
    seg=jnp.array([[1,1,2,2,2,0]])
    pos=jnp.array([[0,1,0,1,0,0]])
    y=mix_bam_previous_token(m,g,seg,pos)
    expected=np.asarray(m).copy()
    for t in (1,3): expected[:,t]=np.array([.75,.25])*np.asarray(m[:,t])+np.array([.25,.75])*np.asarray(m[:,t-1])
    np.testing.assert_array_equal(y,expected)
    np.testing.assert_array_equal(mix_bam_previous_token(m,jnp.zeros_like(g)),m)
    all_previous=mix_bam_previous_token(m,jnp.ones_like(g))
    np.testing.assert_array_equal(all_previous[:,1:],m[:,:-1])
    grad=jax.grad(lambda a:jnp.sum(mix_bam_previous_token(a,g,seg,pos)[:,1]))(m)
    np.testing.assert_allclose(grad[:,0],jnp.broadcast_to(jnp.array([.25,.75]),(1,3,2)))
    np.testing.assert_allclose(grad[:,1],jnp.broadcast_to(jnp.array([.75,.25]),(1,3,2)))
    np.testing.assert_array_equal(grad[:,2:],jnp.zeros_like(grad[:,2:]))

  def test_health_masks_and_known_mix(self):
    m=jnp.array([0.,2.,4.,100.],jnp.float32).reshape(1,4,1,1)
    gate=jnp.array([.99,.25,.25,.99],jnp.float32).reshape(1,4,1)
    segments=jnp.array([[1,1,1,2]])
    mixed=mix_bam_previous_token(m,gate,segments)
    stats,columns,state=bam_m_shift_health(m,mixed,gate,segments)
    np.testing.assert_allclose(stats[0],.25)
    np.testing.assert_allclose(columns[0],.25)
    np.testing.assert_allclose(state[4],.5/np.sqrt(10),rtol=1e-6)
    self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in (stats,columns,state)))
    empty=bam_m_shift_health(m,m,gate,jnp.zeros_like(segments))
    self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in empty))

  def test_full_size_shapes(self):
    counts={}
    for arm in ARMS:
      c=self.config(exp_class=arm,base_emb_dim=1200,base_num_decoder_layers=18,
          base_num_query_heads=16,base_num_kv_heads=16,head_dim=75,vocab_size=50432)
      m,mesh=self.model(c)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),contextlib.redirect_stdout(io.StringIO()):
        p=jax.eval_shape(lambda k:m.init(k,*self.inputs(),enable_dropout=False)['params'],jax.random.key(0))
      p=nn.unbox(p);flat=flatten_dict(p);count=sum(math.prod(a.shape) for a in flat.values())
      mlp=sum(math.prod(a.shape) for k,a in flat.items() if 'mlp' in k and any(x in k for x in ('wi','wo')))
      counts[arm]={'params':count,'mlp':mlp,'shapes':{'/'.join(k):list(a.shape) for k,a in flat.items()}}
      self.assertFalse(c.qk_norm);self.assertEqual(c.pythia_rotary_dim,18)
      if c.dense_conn:
        for i in range(1,18):self.assertEqual(p['decoder'][f'layers_{i}']['compose']['mlp']['dynamic_dense_conn2']['kernel'].shape[-2:],(4,i+1))
      if c.bam_enabled:
        self.assertEqual(c.bam_m_shift_mode,'read');self.assertEqual(c.bam_m_shift_init,.5)
        self.assertTrue(c.bam_local_full_m_read_share_down)
        self.assertTrue(c.bam_share_read_attention_write_down)
        self.assertFalse(bool(c.bam_share_mlp_write_down))
      self.assertEqual(count,434356347 if c.dense_conn else (432356690 if c.bam_enabled else 432360000))
      print('PYTHIA_FAMILY_PARAMS',arm,count,'MLP',mlp,flush=True)
    import os
    if os.environ.get('PYTHIA_SHAPE_OUTPUT'):
      with open(os.environ['PYTHIA_SHAPE_OUTPUT'],'w') as f:json.dump(counts,f,indent=2)

if __name__=='__main__':unittest.main()
