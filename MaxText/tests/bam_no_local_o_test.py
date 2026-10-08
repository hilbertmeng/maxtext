"""LocalO ablation: exact budget, retained initialization, forward/backward equivalence."""
import functools, unittest
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils, train_compile, train
import bam_mlp_write_test as fixture
from layers.models import Transformer
from layers import quantizations
PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
EXP=PARENT.replace('TruePile','NoLocalOTruePile')
def is_o(path):return any(x in path for x in ('W_R_gate','W_R_gate_b0','static_o_key'))
class NoLocalOTest(unittest.TestCase):
  setUp=fixture.MLPWriteTest.setUp
  tearDown=fixture.MLPWriteTest.tearDown
  config=fixture.MLPWriteTest.config
  def test_full_budget_and_health(self):
    shapes=[]
    for exp,expected in [(PARENT,432096128),(EXP,432129824)]:
      c=self.config(exp);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params);total=sum(int(np.prod(v.shape)) for v in flat.values())
      self.assertEqual(total,expected);shapes.append(flat)
      self.assertEqual(c.DATASET_VARIANT,'truepile4096');self.assertTrue(c.bam_record_concat_health)
      self.assertEqual(c.bam_read_key_scale,.2);self.assertEqual(c.bam_read_gate_init,.05)
      if exp==EXP:
        self.assertFalse(any(is_o(p) for p in flat))
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
          metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
        self.assertIn('bam/concat/local_v_gate/layer_017/mean',metrics)
        self.assertFalse(any('local_o_gate' in k or 'static_o_amplitude' in k for k in metrics))
        self.assertIn('bam/concat/mlp_write_gate/layer_016/mean',metrics)
      print('PARAMS',exp,total,flush=True)
    removed=set(shapes[0])-set(shapes[1]);self.assertTrue(removed)
    self.assertTrue(all(is_o(p) for p in removed),removed)
    self.assertEqual(sum(int(np.prod(shapes[0][p].shape)) for p in removed),355104)
  def test_retained_initialization_and_no_o_reference(self):
    tokens=jnp.array([[1,2,3,4]],jnp.int32);call=(tokens,jnp.arange(4)[None],tokens,jnp.ones_like(tokens),jnp.ones_like(tokens))
    records=[]
    for exp in (PARENT,EXP):
      c=self.config(exp,dtype='float32',weight_dtype='float32')
      c.get_keys().update(base_emb_dim=150,emb_dim=150,num_query_heads=2,num_kv_heads=2,base_num_query_heads=2,base_num_kv_heads=2,
          base_num_decoder_layers=3,num_decoder_layers=3,base_mlp_dim=64,mlp_dim=64,mlp_dim_by_block=[64]*3,vocab_size=32,
          bam_layer_modes=['local_qk+local_v+local_o']*3,bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,
          emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c))
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
      records.append((c,mesh,model,flatten_dict(nn.unbox(p))))
    parent=records[0][3];child=records[1][3]
    self.assertTrue(all(is_o(p) for p in set(parent)-set(child)))
    for p,v in child.items():np.testing.assert_array_equal(v,parent[p],err_msg=str(p))
    # Exercise nonzero dynamic V; suppress O in the unchanged parent as a reference.
    for p,v in list(parent.items()):
      if 'W_R' in p and p[-1]=='kernel':parent[p]=jax.random.normal(jax.random.key(91),v.shape)*.02;child[p]=parent[p]
      elif 'W_R_gate_b0' in p:parent[p]=jnp.full_like(v,-jnp.inf)
      elif is_o(p):parent[p]=jnp.zeros_like(v)
    results=[]
    for c,mesh,model,p in records:
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        def loss(q):return jnp.mean(model.apply({'params':q},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]**2)
        value,grad=jax.jit(jax.value_and_grad(loss))(unflatten_dict(p))
      self.assertTrue(np.isfinite(float(value)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      results.append((value,flatten_dict(grad)))
    np.testing.assert_allclose(results[0][0],results[1][0],rtol=1e-6,atol=1e-6)
    for p,g in results[1][1].items():np.testing.assert_allclose(g,results[0][1][p],rtol=2e-4,atol=2e-5,err_msg=str(p))
    for part in ('W_R','mlp_write_gate'):
      self.assertGreater(sum(float(jnp.sum(g*g)) for p,g in results[1][1].items() if part in p),0)
    print('NO_O_REFERENCE_RETAINED_INIT_AND_SCANNED_GRAD_OK',flush=True)
if __name__=='__main__':unittest.main()
