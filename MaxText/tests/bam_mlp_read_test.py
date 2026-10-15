"""MLP BAM reads: per-layer budgets, write ordering, gradients and causality."""
import json
import math
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import bam_mlp, fusion
from layers.models import Transformer

PARENT='BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile'
ARMS=['BamMediumPropMShiftReadG050MLPReadWriteLayersTruePile',
      'BamMediumPropMShiftReadG050MLPReadNonWriteLayersTruePile',
      'BamMediumPropMShiftReadG050MLPReadAllLayersTruePile']

class MlpBamReadTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,arm):
    c=self.config(arm,bam_splash_attention=False,base_emb_dim=150,
        base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,
        base_mlp_dim=128,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,
        emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8,bam_mlp_read_rank=8,
        mlp_dim_by_block=[128,120,128],bam_layer_modes=c.bam_layer_modes[:3])
    return c

  def test_full_shapes_and_per_layer_budget(self):
    result={}
    parent_groups=None
    for i,arm in enumerate([PARENT]+ARMS):
      c=self.config(arm);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,_,_,_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      count=sum(math.prod(v.shape) for v in flat.values())
      expected=432111680+(0 if i==0 else [384,768,1152][i-1])
      self.assertEqual(count,expected)
      readers=[p for p in flat if 'mlp_bam_read' in p]
      self.assertEqual(sum(math.prod(flat[p].shape) for p in readers),
                       392464*[0,6,12,18][i])
      prefixes={p[:p.index('mlp')] for p in flat if 'mlp' in p}
      groups={prefix:sum(math.prod(a.shape) for p,a in flat.items()
                         if p[:len(prefix)]==prefix) for prefix in prefixes}
      if parent_groups is None:
        parent_groups=groups
      else:
        selected_prefixes={p[:p.index('mlp_bam_read')] for p in readers}
        self.assertEqual(set(groups),set(parent_groups))
        for prefix,total in groups.items():
          # Each scanned group contains six copies of one slot: repayment is local.
          self.assertEqual(total-parent_groups[prefix],384 if prefix in selected_prefixes else 0)
      selected=[j for j in range(18) if bam_mlp.read_enabled(c,j)]
      self.assertEqual(len(selected),[0,6,12,18][i])
      self.assertFalse(c.record_training_health_metrics)
      self.assertTrue(all(not v for k,v in c.get_keys().items() if k.startswith('bam_record_')))
      # Scan has three independent parameter groups, each repeated six times.
      for p,a in flat.items():
        if 'mlp' in p and p[-2]=='wi_1':
          gate=flat[p[:-2]+('wi_0','kernel')];output=flat[p[:-2]+('wo','kernel')]
          # Width itself identifies selected slots without relying on scan names.
          width=gate.shape[-1];ordinary=a.shape[-1]
          self.assertIn(width-ordinary,(0,600))
          self.assertEqual(output.shape[0],width)
      result[arm]={'params':count,'selected_layers':selected,'mlp_widths':c.mlp_dim_by_block,
          'dense_matrix_parameters':sum(math.prod(v.shape) for p,v in flat.items()
              if p[-1] in ('kernel','logits_dense'))}
      print('MLP_READ_BUDGET',arm,count,'layers',selected,flush=True)
    Path('/data0/xd/bam_diagnostics/mediumprop-mlp-bam-read').mkdir(parents=True,exist_ok=True)
    Path('/data0/xd/bam_diagnostics/mediumprop-mlp-bam-read/parameter-audit.json').write_text(json.dumps(result,indent=2)+'\n')

  def test_post_attention_read_and_single_decay(self):
    for arm in ARMS:
      c=self.tiny(arm);c.get_keys()['bam_lambda_decay']=.73
      mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
      for index in range(3):
        layer=fusion.SubDecoderLayer(c,mesh,layer_inx=index,sliding_window_size=4)
        x=jax.random.normal(jax.random.key(1),(1,4,150));matrix=jax.random.normal(jax.random.key(2),(1,4,75,32))
        tok=jnp.array([[1,2,3,4]]);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
        args=(x,seg,pos,tok,None,True,'train',None)
        tape={}
        def intercept(next_fun,a,k,ctx):
          y=next_fun(*a,**k)
          if isinstance(ctx.module,bam_mlp.BamMlpRead) and ctx.method_name=='__call__':tape['read']=(a[0],a[1])
          if ctx.method_name=='materialize_attention_write':tape['materialized']=(a,y)
          if ctx.method_name=='merge_mlp_write':tape['merged']=(a,k)
          if ctx.module.name=='post_self_attention_layer_norm' and ctx.method_name=='__call__':tape['norm']=y
          if ctx.module.name=='self_attention' and ctx.method_name=='__call__':tape['attention']=y
          return y
        with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
          v=layer.init(jax.random.key(3),*args,M_in=matrix,is_global=True)
          with nn.intercept_methods(intercept):_,final=layer.apply(v,*args,M_in=matrix,is_global=True)
        selected=bam_mlp.read_enabled(c,index)
        self.assertEqual('read' in tape,selected)
        if selected:
          np.testing.assert_array_equal(tape['read'][0],tape['norm'])
          if index==1:
            factors=tape['materialized'][0][0];attn_delta=jnp.sum(factors[0][...,None]*factors[1][...,None,:],axis=-3)
            np.testing.assert_allclose(tape['read'][1],.73*matrix+attn_delta,rtol=2e-6,atol=2e-6)
            a,k=tape['merged'];np.testing.assert_array_equal(k['attention_write_delta'],attn_delta)
            norm=lambda z:z*jax.lax.rsqrt(jnp.mean(z*z,axis=-1,keepdims=True)+c.normalization_layer_epsilon)
            scale=1/jnp.sqrt(c.num_query_heads) if c.bam_sqrt_n_scale else 1.
            mlp_delta=jnp.sum((scale*a[1][...,None]*norm(a[0]))[...,None]*norm(k['independent_address'])[...,None,:],axis=-3)
            np.testing.assert_allclose(final,.73*matrix+(attn_delta+mlp_delta),rtol=2e-6,atol=2e-6)
          else:
            self.assertNotIn('materialized',tape)
            np.testing.assert_array_equal(tape['read'][1],tape['attention'][1])

  def test_single_contraction_matches_two_read_formula(self):
    for dtype in (jnp.float32,jnp.bfloat16):
      c=self.tiny(ARMS[0]);c.get_keys()['dtype']=dtype
      reader=bam_mlp.BamMlpRead(c)
      x=jax.random.normal(jax.random.key(40),(1,8,150)).astype(dtype)
      matrix=jax.random.normal(jax.random.key(41),(1,8,75,32)).astype(dtype)
      with nn.partitioning.axis_rules([]):
        variables=reader.init(jax.random.key(42),x,matrix)
        actual=reader.apply(variables,x,matrix)
      p=nn.unbox(variables['params'])
      dense=lambda name:y_cast(jnp.einsum('btd,d...->bt...',x,p[name]['kernel'].astype(dtype)))
      y_cast=lambda v:v.astype(dtype)
      hidden=nn.gelu(dense('read_down'))
      key=jnp.einsum('btr,rnv->btnv',hidden,p['read_up']['kernel'].astype(dtype))+p['read_pre_bias'].astype(dtype)
      direction=key*jax.lax.rsqrt(jnp.mean(key.astype(jnp.float32)**2,-1,keepdims=True)+c.bam_read_key_epsilon).astype(dtype)
      key=direction*(c.bam_read_key_scale*jax.nn.sigmoid(dense('read_gate')+p['read_gate_b0'].astype(dtype)))
      dynamic=jnp.einsum('btkv,btnv->btnk',matrix,key)
      static=jnp.einsum('btkv,vn->btnk',matrix,p['static_key'].astype(dtype))
      g=1.1*jax.nn.sigmoid(dense('static_gate')+p['static_gate_b0'].astype(dtype))
      expected=(dynamic+static*(g/c.bam_static_read_gate_init)[...,None]).reshape(actual.shape)
      error=float(jnp.linalg.norm(actual.astype(jnp.float32)-expected.astype(jnp.float32))/jnp.linalg.norm(expected.astype(jnp.float32)))
      self.assertLess(error,1e-6 if dtype==jnp.float32 else .015)
      print('ONE_READ_RELATIVE_ERROR',str(dtype),error,flush=True)

  def test_scan_gradients_and_causality(self):
    for arm in ARMS:
      c=self.tiny(arm);mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
      model=Transformer(c,mesh,quant=None);tok=jnp.array([[1,2,3,4]]);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},tok,pos,tok,seg,seg,enable_dropout=False)['params']
        f=jax.jit(lambda p,t:model.apply({'params':p},t,pos,t,seg,seg,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])
        val,g=jax.jit(jax.value_and_grad(lambda p:jnp.mean(f(p,tok)**2)))(p)
        self.assertTrue(np.isfinite(val));flat=flatten_dict(nn.unbox(g))
        self.assertTrue(all(np.isfinite(v).all() for v in flat.values()))
        for name in ['read_down','read_up','static_key','wi_0','wo']:
          values=[v for p,v in flat.items() if name in p]
          self.assertTrue(values,name);self.assertGreater(sum(float(jnp.linalg.norm(v)) for v in values),0,name)
        np.testing.assert_allclose(f(p,tok)[:,:3],f(p,tok.at[:,3].set(9))[:,:3],rtol=3e-5,atol=3e-5)

if __name__=='__main__':unittest.main()
