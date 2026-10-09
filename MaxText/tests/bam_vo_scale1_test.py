"""Scale-only experiment: unchanged tree/init, live VO scaling and gradients."""
import unittest
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils, train_compile
import bam_mlp_write_test as fixture
from layers import attentions
PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
EXP=PARENT.replace('TruePile','VOReadScale1TruePile')
class VOScaleTest(unittest.TestCase):
  setUp=fixture.MLPWriteTest.setUp
  tearDown=fixture.MLPWriteTest.tearDown
  config=fixture.MLPWriteTest.config
  def test_budget_and_config(self):
    trees=[]
    for name in (PARENT,EXP):
      c=self.config(name);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params);trees.append({p:v.shape for p,v in flat.items()})
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432096128)
      self.assertEqual(c.mlp_dim_by_block,[3901,3774,3901]);self.assertEqual(c.bam_read_key_scale,.2)
      self.assertEqual(c.bam_read_gate_init,.05);self.assertFalse(c.bam_local_v_static_zero_init)
      self.assertFalse(c.bam_disable_local_o);self.assertTrue(c.bam_record_concat_health)
      self.assertEqual(c.DATASET_VARIANT,'truepile4096')
    self.assertEqual(trees[0],trees[1]);print('TREE_BUDGET_GATES_QK_HEALTH_UNCHANGED',flush=True)
  def test_zero_init_live_reads_and_gradients(self):
    x=jax.random.normal(jax.random.key(10),(1,4,150));m=jax.random.normal(jax.random.key(11),(1,4,75,32));records=[]
    for name in (PARENT,EXP):
      c=self.config(name,dtype='float32',weight_dtype='float32');c.get_keys().update(bam_write_v_bottleneck_dim=16,bam_record_concat_health=False)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,
        max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',
        dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=nn.unbox(a.init(jax.random.key(12),m,x,method=a._independent_local_vo))
        out=a.apply(p,m,x,method=a._independent_local_vo)
        for y in out:np.testing.assert_array_equal(y,0)
        grad=jax.grad(lambda q:sum(jnp.sum(v) for v in a.apply(q,m,x,method=a._independent_local_vo)))(p)
      records.append((c,mesh,a,p,grad))
    for p,v in flatten_dict(records[0][3]).items():np.testing.assert_array_equal(v,flatten_dict(records[1][3])[p])
    g0,g1=[flatten_dict(z[4]) for z in records]
    for p in g0:
      self.assertTrue(np.isfinite(np.asarray(g1[p])).all())
      if 'W_R' in p:
        expected=np.asarray(5*g0[p]);actual=np.asarray(g1[p])
        relative_error=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
        self.assertLess(relative_error,2e-6)
    self.assertGreater(sum(float(jnp.sum(g*g)) for p,g in g1.items() if 'W_R' in p),0)
    outs=[]
    for c,mesh,a,p,g in records:
      flat=flatten_dict(p)
      for path,v in list(flat.items()):
        if 'W_R' in path and path[-1]=='kernel':flat[path]=jax.random.normal(jax.random.key(19),v.shape)*.02
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):outs.append(a.apply(unflatten_dict(flat),m,x,method=a._independent_local_vo))
    for arm in (0,1):np.testing.assert_allclose(outs[1][arm],5*outs[0][arm],rtol=2e-5,atol=1e-6)
    print('ZERO_INITIAL_FORWARD_IDENTICAL_VO_GRAD_AND_LIVE_READ_X5',flush=True)
if __name__=='__main__':unittest.main()
