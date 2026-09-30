"""XL shared-normalized embedding/layer writes: budget, equation and gradients."""
import contextlib, io, math, unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers import rmt
from tests.rmt_xlprop_test import XLPropTest

EXP='RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNormLearnedScaleSharedWriteEmbedNorm'

class SharedWriteEmbedTest(unittest.TestCase):
  config=XLPropTest.config
  model_args=XLPropTest.model_args

  def test_target_budget_and_initialization(self):
    cfg=self.config(EXP)
    assert cfg.mlp_dim==6643 and cfg.rmt_embedding_shared_write_norm
    assert cfg.rmt_embedding_shared_content and cfg.rmt_static_write_content_norm
    assert cfg.rmt_dynamic_write_bottleneck_dim==384 and cfg.rmt_matrix_read_norm=='all'
    assert cfg.scan_layers and not cfg.rmt_block_scan
    assert cfg.DATASET_VARIANT=='truepile4096'
    for k in ('rmt_pallas_write','rmt_fused_attention_read','rmt_fused_write_mlp_read'):
      assert not cfg.get_keys().get(k,False)
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      tree=nn.unbox(jax.eval_shape(lambda k:model.init(k,**args)['params'],jax.random.key(1)))
    assert sum(math.prod(x.shape) for x in jax.tree.leaves(tree))==1432453720
    dec=tree['decoder'];assert 'embedding_write_content' not in dec
    shape=[60,96];shape.insert(cfg.param_scan_axis,28)
    assert dec['layers']['attn_norm']['scale'].shape==tuple(shape)
    assert dec['layers']['mlp_norm']['scale'].shape==tuple(shape)
    assert dec['dynamic_embedding_write']['address_down'].shape==(1920,384)

  def test_shared_normalized_embedding_equation(self):
    cfg=self.config(EXP);cfg.get_keys()['dtype']=jnp.float32
    x=jax.random.normal(jax.random.key(2),(1,2,cfg.emb_dim))*.2
    y=x.reshape(1,2,cfg.num_query_heads,cfg.head_dim)
    norm=lambda a:rmt.normalizations.rms_norm(a,dtype=a.dtype,
        epsilon=cfg.normalization_layer_epsilon,statistics_dtype=jnp.float32)
    module=rmt.RMTDynamicWrite(cfg,60,name='dynamic_embedding_write')
    p=module.init(jax.random.key(3),x,norm(y),content_is_normalized=True)
    a,g=module.apply(p,x,norm(y),address_only=True,content_is_normalized=True)
    dy,_=module.apply(p,x,norm(y),content_is_normalized=True)
    key=jax.random.normal(jax.random.key(4),(20,60))*.1
    got=jnp.einsum('btnv,nk->btkv',norm(y),key)+dy
    expected=jnp.einsum('btnk,btnv->btkv',key+g[...,None]*norm(a),norm(y))
    np.testing.assert_allclose(got,expected,rtol=2e-5,atol=3e-6)
    assert np.isfinite(jax.grad(lambda x:jnp.sum(module.apply(p,x,norm(x.reshape(y.shape)),content_is_normalized=True)[0]))(x)).all()

  def test_scanned_gradient_and_health(self):
    cfg=self.config(EXP,base_num_decoder_layers=2,base_emb_dim=640,
        head_dim=32,base_mlp_dim=128,vocab_size=128)
    cfg.get_keys()['dtype']=jnp.float32
    model,args=self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      p=nn.unbox(model.init(jax.random.key(5),**args)['params'])
      def loss(p):
        out,aux=model.apply({'params':p},**args,mutable=['intermediates'])
        return jnp.sum(out[0]),aux
      (v,aux),g=jax.jit(jax.value_and_grad(loss,has_aux=True))(p)
    assert np.isfinite(v) and all(np.isfinite(x).all() for x in jax.tree.leaves((g,aux)))
    assert 'embedding_write_content' not in g['decoder']
    assert float(jnp.linalg.norm(g['decoder']['dynamic_embedding_write']['address_up']))>0
    for arm in ('attn','mlp'):
      assert float(jnp.linalg.norm(g['decoder']['layers'][arm+'_norm']['scale']))>0
      assert float(jnp.linalg.norm(g['decoder']['layers'][arm+'_write_key']))>0

if __name__=='__main__': unittest.main()
