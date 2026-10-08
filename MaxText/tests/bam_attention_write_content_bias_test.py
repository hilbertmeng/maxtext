"""Attention content bias preserves parent RNGs and both write schedules."""
import functools,unittest
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils,train,train_compile
from layers import attentions,quantizations
from layers.models import Transformer
from bam_mlp_write_test import MLPWriteTest
PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile'
EXP=PARENT.replace('TruePile','AttnWriteContentBiasTruePile')
class AttentionWriteContentBiasTest(unittest.TestCase):
  setUp=MLPWriteTest.setUp
  tearDown=MLPWriteTest.tearDown
  config=MLPWriteTest.config
  def test_full_budget_and_bias_health(self):
    c=self.config(EXP);self.assertTrue(c.bam_local_v_static_zero_init)
    self.assertEqual(c.mlp_dim_by_block,[3901,3774,3901])
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
    flat=flatten_dict(args[0].params)
    self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432117728)
    self.assertEqual(sum(int(np.prod(v.shape)) for p,v in flat.items() if 'attn_write_content_bias' in p),21600)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      scalar=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
    for i in range(18):
      for tag in ('bam_rms','standard_rms','bam_over_standard'):
        self.assertIn(f'bam/concat/attn_write_content_bias_amplitude/layer_{i:03d}/{tag}',scalar)
    print('ATTN_CONTENT_BIAS_FULL_BUDGET_AND_54_HEALTH_SCALARS_OK',flush=True)
  def test_parent_init_parity_and_both_write_paths(self):
    c=self.config(EXP,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_write_v_bottleneck_dim=16,bam_record_concat_health=False)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,
        head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,
        mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype,
        layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
    x=jax.random.normal(jax.random.key(1),(1,4,150));y=jax.random.normal(jax.random.key(2),(1,4,2,75))
    m=jax.random.normal(jax.random.key(3),(1,4,75,32))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(a.init(jax.random.key(4),y,x,m,method=a._write))['params']
      c.get_keys()['bam_attn_write_content_pre_rms_bias']=False
      original=nn.unbox(a.init(jax.random.key(4),y,x,m,method=a._write))['params']
      for path,v in flatten_dict(original).items():np.testing.assert_array_equal(v,flatten_dict(p)[path])
      old=a.apply({'params':original},y,x,m,method=a._write)
      c.get_keys()['bam_attn_write_content_pre_rms_bias']=True
      new=a.apply({'params':p},y,x,m,method=a._write)
      np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
      bias=jax.random.normal(jax.random.key(5),(2,75))*.1
      changed=dict(p,attn_write_content_bias={'bias':bias})
      actual,gate=a.apply({'params':changed},y,x,m,method=a._write)
      base_shift,_=a.apply({'params':original},y+bias,x,m,method=a._write)
      np.testing.assert_allclose(actual,base_shift,rtol=1e-6,atol=1e-6)
      content,address,pending_gate=a.apply({'params':changed},y,x,method=a._deferred_write_factors)
      ref=attentions._update_bam_matrix(m,jnp.einsum('btnk,btnv->btkv',content,address),c.bam_lambda_decay)
      np.testing.assert_allclose(actual,ref,rtol=1e-6,atol=1e-6)
      np.testing.assert_array_equal(gate,pending_gate)
      def loss(b):
        z=a.apply({'params':dict(p,attn_write_content_bias={'bias':b})},y,x,m,method=a._write)[0]
        return jnp.sum(z*z)
      g=jax.grad(loss)(bias)
    self.assertTrue(np.isfinite(np.asarray(g)).all());self.assertGreater(float(jnp.sum(g*g)),0.)
    print('ATTN_CONTENT_BIAS_PARENT_RNG_PARITY_REGULAR_DEFERRED_AND_LIVE_GRADIENT_OK',flush=True)
