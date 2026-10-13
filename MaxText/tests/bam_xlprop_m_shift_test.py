"""XL single M-shift: actual 28-layer budget, v40 gate, terminal scan and raw carry."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import train_compile
import train
from bam_prop_test_utils import MLPWriteTest
from layers.attentions import BamAttention,mix_bam_previous_token
from layers.models import Transformer
PARENT='BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile'
NEW='BamXLPropSharedR400GeneralReadMShiftReadG050TruePile'
class XLMShiftTest(MLPWriteTest,unittest.TestCase):
  def mesh(self,c):return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
  def tiny(self):
    c=self.config(NEW,bam_splash_attention=False,base_emb_dim=192,base_num_query_heads=2,
        base_num_kv_heads=2,base_num_decoder_layers=4,base_mlp_dim=32,vocab_size=32,
        dtype='float32',weight_dtype='float32')
    c.get_keys().update(emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,mlp_dim_by_block=[32,24,32],
        bam_final_local_mlp_dim=28,bam_layer_modes=c.bam_layer_modes[:4])
    return c
  def test_formal_parameter_budget_and_terminal_gate(self):
    for exp,total in ((PARENT,1432386360),(NEW,1432383640)):
      c=self.config(exp);args,_,_,_=train_compile.get_shaped_inputs(self.mesh(c),c)
      flat=flatten_dict(nn.unbox(args[0].params))
      actual=sum(int(np.prod(v.shape)) for v in flat.values())
      self.assertEqual(actual,total,(exp,actual,total))
      if exp==NEW:
        gates=[v for p,v in flat.items() if 'M_shift_gate' in p and p[-1]=='kernel']
        self.assertEqual(len(gates),4)
        self.assertEqual(sum(int(np.prod(v.shape)) for v in gates),28*1920*40)
        self.assertTrue(all(v.shape[0]==1920 and v.shape[-1]==40 for v in gates))
        self.assertEqual(c.bam_m_shift_init,.5)
        self.assertEqual(c.learning_rate_schedule_steps,50000)
        self.assertEqual(c.bam_final_local_mlp_dim,6126)
        self.assertEqual(c.tensorboard_gcs_rotate_period,500)
        self.assertEqual(c.wd_mults[-1],('.*M_shift_gate_b0$',0.0))
        self.assertNotIn(('.*_gate_b0$',0.0),c.wd_mults)
      print('XL_MSHIFT_BUDGET_OK',exp,actual,flush=True)
  def test_v40_read_only_carry_initialization_and_boundaries(self):
    c=self.tiny();a=BamAttention(config=c,mesh=self.mesh(c),num_query_heads=2,num_kv_heads=2,
        head_dim=96,bam_k=96,bam_v=40,max_target_length=4,max_prefill_predict_length=4,
        attention_kernel='dot_product_chunk',dtype=c.dtype,weight_dtype=c.weight_dtype,
        layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
    x=jax.random.normal(jax.random.key(1),(1,4,192));m=jax.random.normal(jax.random.key(2),(1,4,96,40))
    pos=jnp.array([[0,1,0,1]]);seg=jnp.array([[1,1,2,2]])
    v=a.init(jax.random.key(3),x,x,pos,seg,M_in=m,defer_write=True)
    p=nn.unbox(v['params']);np.testing.assert_array_equal(p['M_shift_gate']['kernel'],0)
    np.testing.assert_array_equal(p['M_shift_gate_b0'],0)
    _,carry,_=a.apply(v,x,x,pos,seg,M_in=m,defer_write=True)
    np.testing.assert_array_equal(carry,m)
    mix=mix_bam_previous_token(m,jnp.full((1,4,40),.5),seg,pos)
    np.testing.assert_array_equal(mix[:,[0,2]],m[:,[0,2]])
    np.testing.assert_allclose(mix[:,1],.5*(m[:,0]+m[:,1]),rtol=1e-6,atol=1e-6)
  def test_scan_tail_gradient_causality_and_compact_health(self):
    c=self.tiny();mesh=self.mesh(c);model=Transformer(c,mesh,quant=None)
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(model.init(jax.random.key(4),tok,pos,seg,tok)['params'])
      def f(params,tokens=tok):
        out=model.apply({'params':params},tokens,pos,seg,tokens,enable_dropout=False)
        return out[0] if isinstance(out,tuple) else out
      loss,grad=jax.jit(jax.value_and_grad(lambda params:jnp.mean(f(params)**2)))(p)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      gates=[v for path,v in flatten_dict(grad).items() if 'M_shift_gate' in path and path[-1]=='kernel']
      self.assertEqual(len(gates),4)
      self.assertTrue(all(float(jnp.linalg.norm(v))>0 for v in gates))
      np.testing.assert_allclose(f(p)[:,:3],f(p,tok.at[:,3].set(9))[:,:3],rtol=1e-6,atol=1e-6)
      _,collections=model.apply({'params':p},tok,pos,seg,tok,enable_dropout=False,mutable=['intermediates'])
      metrics={'scalar':{}};train.record_bam_concat_health_metrics(metrics,collections,c)
      health={k:v for k,v in metrics['scalar'].items() if k.startswith('bam/m_shift/')}
      self.assertEqual(len(health),4*5)
      self.assertFalse(any('/column_' in k for k in health))
      for layer in range(4):np.testing.assert_allclose(health[f'bam/m_shift/layer_{layer:03d}/m/gate_mean'],.5,rtol=1e-6)
if __name__=='__main__':unittest.main()
