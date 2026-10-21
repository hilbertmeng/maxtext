"""Full-head normalization, RoPE order, matched budgets, causal gradients."""
import unittest
from unittest import mock
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import attentions,dc,normalizations
from layers.models import Transformer
BAM='BamMediumPropMShiftReadG050FullQKNormTruePile'
MHA='BamMHAMediumPropC256QKNormTruePile'
PARENTS=['BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile','BamMHAMediumPropC256TruePile']
class FullQKNormTest(MLPWriteTest,unittest.TestCase):
 def mesh(self,c):return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
 def tiny(self,exp):
  c=self.config(exp,bam_splash_attention=False,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
  c.get_keys().update(emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,bam_layer_modes=c.bam_layer_modes[:3])
  if not c.bam_mha_control:c.get_keys().update(mlp_dim_by_block=[32,24,32])
  return c
 def test_formal_budget_and_scale_shapes(self):
  for exp,parent in zip([BAM,MHA],PARENTS):
   totals=[]
   for e in [parent,exp]:
    c=self.config(e);args,*_=train_compile.get_shaped_inputs(self.mesh(c),c);flat=flatten_dict(nn.unbox(args[0].params));totals.append(sum(int(np.prod(v.shape)) for v in flat.values()))
    if e==exp:
     norm=[v for p,v in flat.items() if 'qk_norm' in p];self.assertEqual(sum(int(np.prod(v.shape)) for v in norm),2700)
     self.assertTrue(all(75 in v.shape for v in norm),[v.shape for v in norm]);self.assertFalse(c.record_training_health_metrics);self.assertEqual(c.tensorboard_gcs_rotate_period,500)
   self.assertEqual(totals[1]-totals[0],2700);print('QKNORM_BUDGET_OK',exp,totals,flush=True)
 def test_complete_head_before_rope(self):
  for exp in [BAM,MHA]:
   c=self.tiny(exp);c.get_keys().update(scan_layers=False,remat_policy='none');mesh=self.mesh(c);m=Transformer(c,mesh,quant=None)
   tok=jnp.array([[1,2,3,4]]);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
   events=[];norms=[];original_norm=dc.QKNorm.__call__;original_rope=attentions.Attention.apply_rotary_embedding
   def norm(mod,q,k):
    events.append(('norm',q.shape[-1]));out=original_norm(mod,q,k)
    for x in out:jax.debug.callback(lambda v:norms.append(np.asarray(v)),x,ordered=True)
    return out
   def rope(mod,x,*a,**kw):events.append(('rope',x.shape[-1]));return original_rope(mod,x,*a,**kw)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(dc.QKNorm,'__call__',norm),mock.patch.object(attentions.Attention,'apply_rotary_embedding',rope):
    params=m.init(jax.random.key(22),tok,pos,tok,mask,mask,enable_dropout=False)['params']
   jax.effects_barrier();self.assertEqual(events,[('norm',75),('rope',18 if exp==BAM else 74),('rope',18 if exp==BAM else 74)]*3)
   self.assertEqual(len(norms),6)
   for a in norms:np.testing.assert_allclose(np.mean(a*a,axis=-1),1,rtol=.003,atol=.003)
   print('QKNORM_INIT_RMS',exp,[float(np.sqrt(np.mean(a*a))) for a in norms],flush=True)
 def test_scan_gradients_and_causality(self):
  for exp in [BAM,MHA]:
   c=self.tiny(exp);mesh=self.mesh(c);m=Transformer(c,mesh,quant=None);tok=jnp.array([[1,2,3,4]]);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    p=m.init(jax.random.key(24),tok,pos,tok,mask,mask,enable_dropout=False)['params']
    def f(p,t):return m.apply({'params':p},t,pos,t,mask,mask,enable_dropout=False)[0]
    loss,g=jax.jit(jax.value_and_grad(lambda p:jnp.mean(f(p,tok)**2)))(p)
    flat=flatten_dict(nn.unbox(g));self.assertTrue(np.isfinite(float(loss)));self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
    for name in ['q_norm','k_norm']:
     self.assertGreater(sum(float(jnp.sum(v*v)) for path,v in flat.items() if name in path),0)
    if exp==BAM:self.assertGreater(sum(float(jnp.sum(v*v)) for path,v in flat.items() if 'M_shift_gate' in path),0)
    np.testing.assert_allclose(f(p,tok)[:,:3],f(p,tok.at[:,3].set(5))[:,:3],atol=1e-6,rtol=1e-6)
 def test_learned_scales_precede_rope_and_zero_safe(self):
  c=self.tiny(BAM);q=jnp.arange(1,76,dtype=jnp.float32)[None,None,None,:];mod=dc.QKNorm(c)
  p=mod.init(jax.random.key(1),q,q);p=nn.unbox(p);p['params']['q_norm']['scale']=jnp.linspace(0,1,75)
  a,b=mod.apply(p,q,q);expected=normalizations.rms_norm(q,dtype=q.dtype,epsilon=1e-6)*(1+jnp.linspace(0,1,75))
  np.testing.assert_allclose(a,expected,rtol=1e-6)
  a,b=mod.apply(p,jnp.zeros_like(q),jnp.zeros_like(q));self.assertTrue(np.isfinite(a).all());np.testing.assert_array_equal(a,0)
 def test_default_and_standard_arm_compatibility(self):
  for full,width in [(True,75),(False,18)]:
   c=self.tiny(BAM);c.get_keys().update(bam_qk_norm_after_concat=full,scan_layers=False,remat_policy='none')
   mesh=self.mesh(c);m=Transformer(c,mesh,quant=None)
   tok=jnp.array([[1,2,3,4]]);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    params=m.init(jax.random.key(29),tok,pos,tok,mask,mask,enable_dropout=False)['params']
   norms=[v for p,v in flatten_dict(nn.unbox(params)).items() if 'qk_norm' in p]
   self.assertEqual(len(norms),6);self.assertTrue(all(v.shape==(width,) for v in norms))
  c=self.config('BamPythia2p8BSotaIndependentR768Pallas')
  self.assertTrue(c.qk_norm and c.bam_qk_norm_after_concat)
 def test_pythia_full_head_parameter_budgets(self):
  expected={'Pythia2p8B':2775208960,'BamPythia2p8BSota':2775043296,
            'BamPythia2p8BSotaIndependentR768':2774919648,
            'BamPythia2p8BSotaIndependentR768Pallas':2774919648}
  for exp,total in expected.items():
   c=self.config(exp,attention='dot_product',bam_splash_attention=False)
   c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
   args,*_=train_compile.get_shaped_inputs(self.mesh(c),c)
   flat=flatten_dict(nn.unbox(args[0].params));actual=sum(int(np.prod(v.shape)) for v in flat.values())
   self.assertEqual(actual,total,exp)
   norms=[v for p,v in flat.items() if 'qk_norm' in p]
   if exp!='Pythia2p8B':
    self.assertTrue(norms);self.assertTrue(all(80 in v.shape for v in norms),[v.shape for v in norms])
   print('PYTHIA_QKNORM_BUDGET_OK',exp,actual,flush=True)
if __name__=='__main__':unittest.main()
