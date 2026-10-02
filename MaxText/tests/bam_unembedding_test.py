"""Focused Direct32 boundary and scanned model/health checks."""
import functools, tempfile, unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig, max_utils, train, train_compile
from layers import quantizations
from layers.models import Transformer
from layers.bam_unembedding import BamDynamicUnembedding, HEALTH_NAMES

EXP='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteStaticEveryThirdUnembedDirect32TruePile'
EXPS=[EXP.replace('StaticEveryThird','EveryThird'),EXP]

class UnembeddingTest(unittest.TestCase):
  def setUp(self):
    self.tmp=tempfile.TemporaryDirectory();Path(self.tmp.name,'audit').mkdir()
  def tearDown(self):self.tmp.cleanup()
  def config(self, exp=EXP, **kw):
    return pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=exp,
        run_name='audit',enable_checkpointing=False,base_output_directory=self.tmp.name+'/',
        jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,
        max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,**kw)
  def test_parameters_health(self):
    for exp,expected in zip(EXPS,[432087840,432090912]):
      c=self.config(exp);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params)
      total=sum(int(np.prod(v.shape)) for v in flat.values())
      boundary=sum(int(np.prod(v.shape)) for p,v in flat.items() if 'dynamic_unembedding_read' in p)
      self.assertEqual(boundary,637216);self.assertEqual(total,expected)
      self.assertEqual(c.DATASET_VARIANT,'truepile4096')
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
      for name in HEALTH_NAMES:self.assertIn('bam/unembedding/'+name,metrics['scalar'])
      print('PARAM_HEALTH_OK',exp,total,boundary,flush=True)
  def test_boundary_and_scanned_gradients(self):
    c=self.config(dtype='float32',weight_dtype='float32')
    c.get_keys().update(base_emb_dim=150,emb_dim=150,num_query_heads=2,num_kv_heads=2,
        base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,
        num_decoder_layers=3,base_mlp_dim=64,mlp_dim=64,mlp_dim_by_block=[64]*3,
        vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    x=jax.random.normal(jax.random.key(1),(1,4,150))
    m=jax.random.normal(jax.random.key(2),(1,4,75,32))
    b=BamDynamicUnembedding(c)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      v=b.init(jax.random.key(3),x,m)
      np.testing.assert_array_equal(b.apply(v,x,m),x)
      p=nn.unbox(v['params']);p['read_key']['kernel']=jnp.ones_like(p['read_key']['kernel'])*.01
      y=b.apply({'params':p},x,m);y2=b.apply({'params':p},x,m*7)
      np.testing.assert_allclose(y,y2,rtol=1e-4,atol=1e-4)
      # Gate is the only amplitude coefficient: compare explicitly with the outer read.
      norm=lambda z:z*jax.lax.rsqrt(jnp.mean(z*z,axis=-1,keepdims=True)+c.normalization_layer_epsilon)
      raw_key=(norm(x)@p['read_key']['kernel'].reshape(150,-1)).reshape(1,4,2,32)
      key_eps=c.normalization_layer_epsilon if c.bam_read_key_epsilon is None else c.bam_read_key_epsilon
      key=raw_key*jax.lax.rsqrt(jnp.mean(raw_key*raw_key,axis=-1,keepdims=True)+key_eps)
      mn=m*jax.lax.rsqrt(jnp.mean(m*m,axis=(-2,-1),keepdims=True)+c.normalization_layer_epsilon)
      expected=x+.05*jnp.einsum('btkv,btnv->btnk',mn,key).reshape(x.shape)
      np.testing.assert_allclose(y,expected,rtol=1e-4,atol=1e-4)
      model=Transformer(c,mesh,quantizations.configure_quantization(c))
      tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens)
      call=(tokens,pos,tokens,mask,mask)
      params=model.init({'params':jax.random.key(4),'dropout':jax.random.key(5),'aqt':jax.random.key(6)},*call,enable_dropout=False)['params']
      def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(6)})[0])
      val,grad=jax.jit(jax.value_and_grad(loss))(params)
      self.assertTrue(np.isfinite(float(val)))
      self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad)))
      flat=flatten_dict(nn.unbox(grad));kg=[g for p,g in flat.items() if 'dynamic_unembedding_read' in p and 'read_key' in p]
      self.assertGreater(sum(float(jnp.sum(g*g)) for g in kg),0)
    print('BOUNDARY_SCAN_GRAD_OK',float(val),flush=True)

if __name__=='__main__':unittest.main()
