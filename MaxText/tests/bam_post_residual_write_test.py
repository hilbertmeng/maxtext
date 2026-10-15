"""Post-residual write controls: exact budget, semantic inputs and causal gradients."""
import functools
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import fusion
from layers.models import Transformer

PARENT = 'BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile'
EXP = 'BamMediumPropMShiftReadG050PostResidualWriteControlsTruePile'

class PostResidualWriteTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, enabled=True):
    c=self.config(EXP if enabled else PARENT,bam_splash_attention=False,
        base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,
        base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,
        dtype='float32',weight_dtype='float32')
    c.get_keys().update(emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,mlp_dim_by_block=[32,24,32],
        bam_layer_modes=c.bam_layer_modes[:3],record_internal_nn_metrics=False,
        bam_record_concat_health=False,bam_record_m_shift_health=False)
    return c

  def test_full_size_budget_and_health_off(self):
    trees=[]
    for name in (PARENT,EXP):
      c=self.config(name)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
      flat=flatten_dict(args[0].params)
      trees.append({p:v.shape for p,v in flat.items()})
      self.assertEqual(sum(int(np.prod(v.shape)) for v in flat.values()),432111680)
      self.assertEqual(c.mlp_dim_by_block,[3802,3676,3802])
      if name==EXP:
        self.assertFalse(c.record_training_health_metrics)
        self.assertFalse(c.record_internal_nn_metrics)
        self.assertTrue(all(not v for k,v in c.get_keys().items() if k.startswith('bam_record_')))
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
          metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
        self.assertFalse(any('grad_norm' in k or k.startswith('bam/') or k.startswith('raw_grads/') for k in metrics['scalar']))
    self.assertEqual(trees[0],trees[1])
    print('POST_RESIDUAL_FULL_BUDGET_HEALTH_OK',432111680,flush=True)

  def test_controls_use_updated_residual_and_preserve_contents(self):
    c=self.tiny();mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
    x=jax.random.normal(jax.random.key(10),(1,4,150))
    m=jax.random.normal(jax.random.key(11),(1,4,75,32))
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    call=(x,seg,pos,tok,None,True,'train',None)
    for layer in (0,1):
      new=fusion.SubDecoderLayer(c,mesh,layer_inx=layer,sliding_window_size=4)
      old=fusion.SubDecoderLayer(self.tiny(False),mesh,layer_inx=layer,sliding_window_size=4)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        variables=new.init(jax.random.key(20),*call,M_in=m,is_global=True)
        original=old.init(jax.random.key(20),*call,M_in=m,is_global=True)
        a=flatten_dict(nn.unbox(variables['params']));b=flatten_dict(nn.unbox(original['params']))
        self.assertEqual(a.keys(),b.keys())
        for path in a:np.testing.assert_array_equal(a[path],b[path],err_msg=str(path))
        tape={}
        def intercept(next_fun,args,kwargs,ctx):
          y=next_fun(*args,**kwargs)
          name=ctx.module.name
          if ctx.method_name=='__call__' and name in ('pre_self_attention_layer_norm','post_self_attention_layer_norm','W_gw','P_loc_down','mlp_write_gate','mlp_address_down','self_attention','mlp'):
            tape.setdefault(name,[]).append((args[0] if args else kwargs.get('inputs_q'),y))
          if ctx.method_name in ('_write','_deferred_write_factors'):
            tape.setdefault('content',[]).append(args[0])
          return y
        with nn.intercept_methods(intercept):out,mout=new.apply(variables,*call,M_in=m,is_global=True)
        oldout,oldm=old.apply(original,*call,M_in=m,is_global=True)
        np.testing.assert_allclose(out,oldout,rtol=2e-6,atol=2e-6)
        an=tape['pre_self_attention_layer_norm'];fn=tape['post_self_attention_layer_norm']
        np.testing.assert_allclose(an[1][0],x+tape['self_attention'][0][1][0],rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(tape['W_gw'][0][0],an[1][1],rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(tape['P_loc_down'][0][0],an[1][1],rtol=2e-6,atol=2e-6)
        if layer==1:
          np.testing.assert_allclose(fn[1][0],fn[0][0]+tape['mlp'][0][1],rtol=2e-6,atol=2e-6)
          np.testing.assert_allclose(tape['mlp_write_gate'][0][0],fn[1][1],rtol=2e-6,atol=2e-6)
          np.testing.assert_allclose(tape['mlp_address_down'][0][0],fn[1][1],rtol=2e-6,atol=2e-6)
        self.assertGreater(float(jnp.linalg.norm(mout-oldm)),1e-5)
    print('POST_RESIDUAL_CONTROLS_INIT_CONTENT_OK',flush=True)

  def test_scan_gradients_and_causality(self):
    c=self.tiny();mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
    model=Transformer(c,mesh,quant=None)
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    def call(p,t):return model.apply({'params':p},t,pos,t,seg,seg,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},tok,pos,tok,seg,seg,enable_dropout=False)['params']
      val,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(call(p,tok)**2)))(params)
      self.assertTrue(np.isfinite(float(val)))
      self.assertTrue(all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad)))
      flat=flatten_dict(nn.unbox(grad))
      for name in ('W_gw','P_loc_down','mlp_write_gate','mlp_address_down'):
        self.assertGreater(sum(float(jnp.sum(v*v)) for p,v in flat.items() if name in p),0,name)
      y=call(params,tok);changed=call(params,tok.at[:,3].set(8))
      np.testing.assert_allclose(y[:,:3],changed[:,:3],rtol=3e-5,atol=3e-5)
    print('POST_RESIDUAL_SCAN_GRAD_CAUSAL_OK',float(val),flush=True)

if __name__=='__main__':unittest.main()
