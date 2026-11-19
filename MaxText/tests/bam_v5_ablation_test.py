"""Model-tree, ablation semantics and unchanged-kernel full-gradient controls."""
import contextlib,io,json,math,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig,max_utils
from layers.models import Transformer,EmbeddingBamWrite
from layers.bam_v5_ablation import Settings,NAMES
from bam_prop_test_utils import MLPWriteTest

PREFIX='BamMediumPropV5Ablation'
SUFFIX=('StaticMLPWrite','StaticAttentionWrite','StaticAllWrites','StaticReads','StaticReadsWrites',
        'NoDynamicGates','NoAddressNorm','NoDynamicGatesAddressNorm','NoMShift','NoMemoryMLP',
        'NoLocalO','NoAttentionWriteContentNorm','MReadQKPartialRoPE')

class AblationTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,name,kernel=False,dtype='float32'):
    c=self.config(name,dtype=dtype,weight_dtype='float32',attention='dot_product')
    small=c.bam_memory_mlp_dim!=0
    c.get_keys().update(emb_dim=32,num_query_heads=2,num_kv_heads=2,head_dim=16,bam_k=16,bam_v=8,
        num_decoder_layers=2,mlp_dim=24,bam_last_mlp_dim=24,vocab_size=32,emb_bam_num_head=2,
        bam_local_qk_col_output_dim=16,bam_standard_qk_dim=4,bam_partial_rope_nope_dim=16,
        bam_layer_modes=['local_qk+local_v+local_o']*2,bam_attention_write_layers=[0],
        bam_mlp_write_layers=[0] if small else [],bam_memory_mlp_dim=12 if small else 0,
        bam_local_full_m_read_bottleneck_dim=8,bam_write_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,emb_bam_v_bottleneck_dim=8,
        bam_pallas_core=kernel,bam_pallas_write=kernel,bam_pallas_interpret=kernel,
        bam_pallas_fused_inputs=False,bam_pallas_packed_read_keys=False,
        max_target_length=128,max_prefill_predict_length=128,query_chunk_size=64,
        bam_splash_attention=False)
    return c

  def evaluate(self,name,kernel=False,params=None,dtype='float32',gradient=True):
    c=self.tiny(name,kernel,dtype)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    model=Transformer(c,mesh,quant=None)
    tok=jax.random.randint(jax.random.key(7),(1,128),0,32)
    pos=(jnp.arange(128)%64)[None];seg=(jnp.arange(128)//64+1)[None];target=(tok+3)%32
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      if params is None:
        params=nn.unbox(model.init(jax.random.key(8),tok,pos,target,jnp.ones_like(tok),seg,enable_dropout=False))
        def perturb(path,v):
          if path[-1].key in ('common_vo_read_scale','common_qk_read_scale'):return v*-.7
          return v
        params=jax.tree.map_with_path(perturb,params)
      def loss(p):
        xent=model.apply(p,tok,pos,target,jnp.ones_like(tok),seg,enable_dropout=False)[0]
        return jnp.mean(xent.astype(jnp.float32))
      result=jax.jit(jax.value_and_grad(loss) if gradient else loss)(params)
    return params,result

  def test_formal_budgets_and_parameter_omission(self):
    # Full-size abstract trees are audited, not inferred from config widths.
    import mediumprop_total_budget_depth_test as budget
    expected=json.loads(Path('experiments/bam_llama2_medium/v5_ablation_budgets.json').read_text())
    h=budget.TotalBudgetDepthTest();h.setUp()
    try:
      for row in expected:
        with contextlib.redirect_stdout(io.StringIO()):c,flat=h.shaped(row['name'])
        self.assertEqual(sum(math.prod(v.shape)for v in flat.values()),row['total'])
        self.assertLessEqual(abs(row['residual']),1800)
        paths=['/'.join(p) for p in flat]
        ab=Settings(row['mode'])
        if ab.static_read:
          self.assertFalse(any('full_m_read_down' in p or '_c8_up' in p or 'W_R_up' in p or 'general_'in p for p in paths))
        if ab.no_gate:
          self.assertFalse(any(x in p for p in paths for x in ('W_lq_gate','W_lk_gate','W_lv_gate','W_R_gate','W_gw','gw_b0','mlp_write_gate','W_emb_g')))
        if ab.static_write('mlp'):self.assertFalse(any('mlp_address_'in p for p in paths))
        if ab.static_write('attention'):self.assertFalse(any('P_loc_'in p for p in paths))
        if ab.static_write('embedding'):self.assertFalse(any('W_emb_v_'in p for p in paths))
        if ab.name=='no_memory_mlp':self.assertFalse(any('memory_mlp/'in p or 'mlp_address_'in p for p in paths))
        self.assertEqual((c.steps,c.learning_rate_schedule_steps),(13500,13500))
        self.assertTrue(c.record_training_health_metrics)
        self.assertFalse(c.record_internal_nn_metrics)
    finally:h.tearDown()

  def test_control_matches_parent_all_gradients(self):
    params,want=self.evaluate('BamMediumPropV5')
    _,got=self.evaluate(PREFIX+'ControlTruePile',params=params)
    np.testing.assert_allclose(got[0],want[0],rtol=1e-6,atol=1e-6)
    self.compare_gradients(got[1],want[1],2e-3)

  def compare_gradients(self,actual,expected,tol=5e-4):
    aa,ee=flatten_dict(actual),flatten_dict(expected);self.assertEqual(aa.keys(),ee.keys())
    worst=0
    for p in aa:
      a,e=np.asarray(aa[p]),np.asarray(ee[p]);self.assertTrue(np.isfinite(a).all(),p)
      rel=np.max(np.abs(a-e))/(np.max(np.abs(e))+1e-10)
      worst=max(worst,float(rel));self.assertLess(rel,tol,str(p))
    return worst

  def test_native_pallas_full_model_gradients(self):
    for suffix in [x for x in SUFFIX if 'AddressNorm' not in x]:
      with self.subTest(suffix=suffix):
        name=PREFIX+suffix+'TruePile'
        params,ref=self.evaluate(name)
        _,got=self.evaluate(name,True,params)
        np.testing.assert_allclose(got[0],ref[0],rtol=3e-4,atol=2e-6)
        print('ABLATION_PALLAS',suffix,self.compare_gradients(got[1],ref[1],3e-3),flush=True)
        jax.clear_caches()

  def test_native_address_norm_ablations_finite(self):
    for suffix in ('NoAddressNorm','NoDynamicGatesAddressNorm'):
      for dtype in ('float32','bfloat16'):
        _,out=self.evaluate(PREFIX+suffix+'TruePile',dtype=dtype)
        self.assertTrue(np.isfinite(np.asarray(out[0])).all())
        for p,x in flatten_dict(out[1]).items():self.assertTrue(np.isfinite(np.asarray(x)).all(),p)
        jax.clear_caches()

if __name__=='__main__':unittest.main()
