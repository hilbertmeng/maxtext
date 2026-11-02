"""Whole-M read pre-norm: formal budget, shift/carry boundaries and all gradients."""
import contextlib
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils
import pyconfig
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import attentions
from layers.models import Transformer

PARENT = ('BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeys'
          'LogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile')
KEEP = PARENT.removesuffix('TruePile') + 'MPreNormTruePile'
DROP = PARENT.removesuffix('TruePile') + 'MPreNormNoReadScaleTruePile'


def oracle_view(module, matrix):
  if matrix is None or module._m_read_norm == 'none':
    return matrix
  # Flatten the matrix axes; normalize as one vector with a single RMS per token.
  x = matrix.astype(jnp.float32)
  flat = x.reshape(x.shape[:-2] + (-1,))
  denominator = jnp.sqrt(jnp.sum(flat*flat, axis=-1, keepdims=True) / flat.shape[-1]
                         + module._rms_epsilon)
  return (flat / denominator).reshape(x.shape).astype(matrix.dtype)


def drop_scale_params(params):
  return unflatten_dict({k: v for k, v in flatten_dict(params).items()
                         if k[-1] != 'common_vo_read_scale'})


class MatrixPreNormTest(MLPWriteTest, unittest.TestCase):
  def tiny(self, run, dtype='float32'):
    c = self.config(run, dtype=dtype, weight_dtype='float32')
    c.get_keys().update(emb_dim=150, num_query_heads=2, num_kv_heads=2, head_dim=75,
        num_decoder_layers=3, mlp_dim=24, vocab_size=32, query_chunk_size=2,
        bam_k=75, bam_v=32, bam_abs_v_compression_dim=2,
        bam_local_qk_direct_c8=True, bam_local_qk_c8_rank=2,
        bam_partial_rope_nope_dim=75,
        bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0,1], bam_mlp_write_layers=[0,1],
        bam_memory_mlp_dim=12, bam_last_mlp_dim=24, emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=4, emb_bam_v_bottleneck_dim=4,
        bam_mlp_write_address_rank=4, bam_local_full_m_read_bottleneck_dim=4,
        bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False)
    return c

  @contextlib.contextmanager
  def model(self, cfg):
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      yield Transformer(cfg, mesh, quant=None)

  def inputs(self):
    return (jnp.array([[1,2,3,4]],jnp.int32), jnp.array([[0,1,0,1]],jnp.int32),
            jnp.array([[1,1,2,2]],jnp.int32), jnp.array([[2,3,4,5]],jnp.int32))

  def test_formal_configs_budgets_and_native_reference(self):
    configs = [self.config(n) for n in [PARENT, KEEP, DROP]]
    totals = []
    for c in configs:
      c.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      shaped, *_ = train_compile.get_shaped_inputs(mesh,c)
      flat = flatten_dict(nn.unbox(shaped[0].params))
      totals.append(sum(np.prod(v.shape,dtype=np.int64) for v in flat.values()))
      if c is not configs[0]:
        self.assertEqual(c.bam_m_read_norm,'rms')
        self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.bam_pallas_packed_read_keys)
        self.assertEqual((c.remat_policy,c.bam_splash_residual_name),('full',''))
        self.assertTrue(c.bam_splash_attention and c.record_training_health_metrics)
        self.assertFalse(c.record_internal_nn_metrics or c.bam_record_m_shift_health)
        self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
        self.assertEqual(c.compare_runs,[PARENT])
        self.assertEqual(c.bam_attention_write_layers,list(range(17)))
        self.assertTrue(c.bam_skip_final_mlp_write and c.scan_layers)
        self.assertFalse(c.bam_pair_scan)
    self.assertEqual(totals,[432112210,432112210,432112192])
    print('FORMAL_BUDGETS',totals,flush=True)

  def test_keep_and_drop_have_identical_initial_function_and_common_gradients(self):
    cfg=self.tiny(KEEP);dcfg=self.tiny(DROP);tokens,pos,seg,target=self.inputs()
    with self.model(cfg) as model, self.model(dcfg) as other:
      params=nn.unbox(model.init(jax.random.key(9),tokens,pos,seg,target))
      scales=[v for p,v in flatten_dict(params).items() if p[-1]=='common_vo_read_scale']
      self.assertEqual(sum(v.size for v in scales),3)
      for v in scales:np.testing.assert_array_equal(v,jnp.ones_like(v))
      no_scale=drop_scale_params(params)
      def loss(m,p):return jnp.mean(m.apply(p,tokens,pos,seg,target)[0].astype(jnp.float32))
      l,g=jax.value_and_grad(lambda p:loss(model,p))(params)
      r,h=jax.value_and_grad(lambda p:loss(other,p))(no_scale)
      np.testing.assert_allclose(l,r,rtol=1e-6,atol=1e-7)
      fg,fh=flatten_dict(drop_scale_params(g)),flatten_dict(h)
      self.assertEqual(fg.keys(),fh.keys())
      for p in fg:np.testing.assert_allclose(fg[p],fh[p],rtol=1e-5,atol=1e-6,err_msg=str(p))
      self.assertGreater(sum(float(jnp.linalg.norm(v)) for p,v in flatten_dict(g).items()
                             if p[-1]=='common_vo_read_scale'),0.)

  def test_flattened_vector_oracle_forward_and_every_gradient(self):
    for run in [KEEP,DROP]:
      cfg=self.tiny(run);tokens,pos,seg,target=self.inputs()
      with self.model(cfg) as model:
        params=nn.unbox(model.init(jax.random.key(3),tokens,pos,seg,target))
        loss=lambda p:jnp.mean(model.apply(p,tokens,pos,seg,target)[0].astype(jnp.float32))
        actual,grad=jax.value_and_grad(loss)(params)
        with mock.patch.object(attentions.BamAttention,'_matrix_for_read',oracle_view):
          expected,oracle_grad=jax.value_and_grad(loss)(params)
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
        for p,v in flatten_dict(grad).items():
          self.assertTrue(np.isfinite(v).all(),p)
          np.testing.assert_allclose(v,flatten_dict(oracle_grad)[p],rtol=1e-4,atol=2e-6,err_msg=str(p))

  def test_shift_then_norm_and_raw_carry(self):
    cfg=self.tiny(KEEP);cfg.get_keys().update(bam_attention_write_layers=[],bam_mlp_write_layers=[])
    tokens,pos,seg,target=self.inputs();original=attentions.BamAttention._matrix_for_read
    def intercept(next_fun,args,kwargs,ctx):
      module=ctx.module
      if ctx.method_name!='__call__' or not isinstance(module,attentions.BamAttention):
        return next_fun(*args,**kwargs)
      raw=kwargs['M_in']
      def measured(self,matrix):
        value=original(self,matrix)
        self.sow('intermediates','read_norm_boundary',jnp.stack((raw,matrix,value)))
        return value
      with mock.patch.object(attentions.BamAttention,'_matrix_for_read',measured):
        result=next_fun(*args,**kwargs)
      # Read pre-norm and read-only shift cannot replace the carried matrix.
      if not module.is_initializing():
        module.sow('intermediates','raw_carry_boundary',jnp.stack((raw,result[1])))
      return result
    with self.model(cfg) as model:
      params=nn.unbox(model.init(jax.random.key(5),tokens,pos,seg,target))
      with nn.intercept_methods(intercept):
        _,collections=model.apply(params,tokens,pos,seg,target,mutable=['intermediates'])
      found=[]
      for path,value in flatten_dict(collections['intermediates']).items():
        if path[-1] != 'raw_carry_boundary':continue
        while isinstance(value,tuple):value=value[0]
        if 'layers' in path:
          np.testing.assert_array_equal(value[:,0],value[:,1])
        else:np.testing.assert_array_equal(value[0],value[1])
      for p,v in flatten_dict(collections['intermediates']).items():
        if p[-1]!='read_norm_boundary':continue
        while isinstance(v,tuple):self.assertEqual(len(v),1);v=v[0]
        found.extend(list(v) if 'layers' in p else [v])
      self.assertEqual(len(found),3)
      for v in found:
        raw,mixed,normalized=v
        previous=jnp.concatenate((jnp.zeros_like(raw[:,:1]),raw[:,:-1]),axis=1)
        valid=jnp.array([[False,True,False,True]])[:,:,None,None]
        expected=jnp.where(valid,(raw+previous)*.5,raw)
        np.testing.assert_allclose(mixed,expected,rtol=1e-6,atol=1e-7)
        ref=oracle_view(SimpleNamespace(_m_read_norm='rms',_rms_epsilon=cfg.normalization_layer_epsilon),expected)
        np.testing.assert_allclose(normalized,ref,rtol=1e-6,atol=1e-7)

  def test_bfloat16_statistics_and_zero_matrix_are_finite(self):
    cfg=self.tiny(KEEP,dtype='bfloat16');tokens,pos,seg,target=self.inputs()
    with self.model(cfg) as model:
      params=nn.unbox(model.init(jax.random.key(4),tokens,pos,seg,target))
      loss,grad=jax.value_and_grad(lambda p:jnp.mean(model.apply(p,tokens,pos,seg,target)[0].astype(jnp.float32)))(params)
      self.assertTrue(np.isfinite(loss))
      self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(grad)))
    # Exercise the production read-view method directly with zero/small matrices.
    class MatrixView(nn.Module):
      _m_read_norm = 'rms'
      _rms_epsilon = 1e-6
      @nn.compact
      def __call__(self,x):
        return attentions.BamAttention._matrix_for_read(self,x)
    view=MatrixView()
    zero=jnp.zeros((1,2,3,4),jnp.float32)
    result=view.apply({},zero)
    np.testing.assert_array_equal(result,zero)
    gradient=jax.grad(lambda x:jnp.sum(view.apply({},x)))(zero)
    np.testing.assert_allclose(gradient,1000.,rtol=1e-6)
    matrix=jax.random.normal(jax.random.key(8),(1,2,3,4)).astype(jnp.bfloat16)*.003
    expected=oracle_view(view,matrix)
    np.testing.assert_array_equal(view.apply({},matrix),expected)

if __name__=='__main__':unittest.main()
