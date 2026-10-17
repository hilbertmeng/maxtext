"""Pythia integration, independent NeoX reference, initialization and compatibility."""
import copy
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import orbax.checkpoint as ocp
import common_types
import optax
import pyconfig
from layers.models import Transformer
from layers import initializers, normalizations
from pythia_config import rotary_width


class PythiaTest(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    Path(self.tmp.name, 'test').mkdir()

  def tearDown(self):
    self.tmp.cleanup()

  def config(self, **kwargs):
    args = dict(exp_class='PythiaArchitecture', run_name='test',
        base_output_directory=self.tmp.name+'/', enable_checkpointing=False,
        jax_cache_dir='', log_config=False, dataset_type='synthetic',
        base_emb_dim=32, base_num_decoder_layers=2, base_num_query_heads=2,
        base_num_kv_heads=2, head_dim=16, vocab_size=32,
        max_target_length=8, max_prefill_predict_length=4, per_device_batch_size=1.,
        attention='dot_product', dtype='float32', weight_dtype='float32',
        loss_chunk_size=4, remat_policy='none')
    args.update(kwargs)
    return pyconfig.initialize([None, 'MaxText/configs/base.yml'], **args)

  def model(self, c):
    mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
    return Transformer(c,mesh,quant=None),mesh

  def inputs(self):
    t=jnp.array([[1,2,3,4,5,6,7,8]],jnp.int32)
    return t,jnp.arange(8)[None],(t+1)%32,jnp.ones_like(t),jnp.ones_like(t)

  def init(self, c):
    m,mesh=self.model(c)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      p=m.init(jax.random.key(19),*self.inputs(),enable_dropout=False)['params']
    return m,mesh,p

  def test_config_and_init_roles(self):
    c=self.config(scan_layers=False)
    self.assertEqual(c.mlp_dim,128)
    self.assertEqual(self.config(base_mlp_dim=96).mlp_dim,96)
    self.assertEqual(rotary_width(96),24)
    self.assertEqual(rotary_width(75,explicit=18),18)
    for hd,fraction,width in [(75,.25,-1),(12,.25,-1),(16,.25,17),(16,1.5,-1)]:
      with self.assertRaises(ValueError):rotary_width(hd,fraction,width)
    for opts in [dict(qkv_bias=False),dict(bam_enabled=True),dict(qk_norm=False,rope_half=True),
                 dict(num_experts=2),dict(dense_conn=True),dict(mlp_activations=['silu','linear'])]:
      with self.assertRaises((ValueError,AssertionError)):self.config(**opts)
    m,mesh,p=self.init(c);flat=flatten_dict(nn.unbox(p))
    for path,a in flat.items():
      if path[-1]=='bias':np.testing.assert_array_equal(a,0)
      elif path[-1]=='scale':np.testing.assert_array_equal(a,1)
    for output in (False,True):
      a=initializers.get_model_init(c,residual_output=output)(jax.random.key(3),(1024,1024),jnp.float32)
      target=2/c.num_decoder_layers/np.sqrt(c.emb_dim) if output else np.sqrt(2/(5*c.emb_dim))
      self.assertAlmostEqual(float(a.std())/target,1,delta=.005)
    self.assertTrue(any(path[-2:]==('out','bias') for path in flat))
    self.assertTrue(any(path[-2:]==('wo','bias') for path in flat))
    # With a fixed RNG each role must actually be wired, not just implemented.
    for path,a in flat.items():
      if path[-1]=='kernel' or path[-1] in ('embedding','logits_dense'):
        target=2/c.num_decoder_layers/np.sqrt(c.emb_dim) if path[-2] in ('out','wo') else np.sqrt(2/(5*c.emb_dim))
        self.assertAlmostEqual(float(a.std())/target,1,delta=.09,msg=str(path))

  def test_scan_gradients_causality_and_checkpoint(self):
    c=self.config();m,mesh,p=self.init(c);args=self.inputs()
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      f=jax.jit(lambda p,t:m.apply({'params':p},t,*args[1:],enable_dropout=False)[0])
      loss,g=jax.value_and_grad(lambda p:jnp.mean(f(p,args[0])))(p)
      self.assertTrue(np.isfinite(loss))
      for a in jax.tree.leaves(nn.unbox(g)):self.assertTrue(np.isfinite(a).all())
      np.testing.assert_allclose(f(p,args[0])[:,:6],f(p,args[0].at[:,6:].set(17))[:,:6],atol=2e-6)
      plain=nn.unbox(p)
      ckpt=ocp.PyTreeCheckpointer();dest=Path(self.tmp.name)/'ckpt'
      ckpt.save(str(dest),plain);restored=ckpt.restore(str(dest),item=plain);ckpt.close()
      np.testing.assert_array_equal(f(plain,args[0]),f(restored,args[0]))
      # Map scan's parameter axis into explicit layers, then compare same weights.
      unscanned=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),scan_layers=False)))
      um,_=self.model(unscanned);up=copy.deepcopy(plain)
      layers=up['decoder'].pop('layers')
      for i in range(c.num_decoder_layers):
        up['decoder'][f'layers_{i}']=jax.tree.map(lambda a:jnp.take(a,i,axis=c.param_scan_axis),layers)
      np.testing.assert_allclose(f(plain,args[0]),um.apply({'params':up},*args,enable_dropout=False)[0],atol=2e-6)

  def test_bfloat16_remat_training(self):
    c=self.config(dtype='bfloat16',remat_policy='full')
    m,mesh,p=self.init(c);args=self.inputs()
    optimizer=optax.adam(1e-3);state=optimizer.init(p)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      @jax.jit
      def step(p,state):
        loss,g=jax.value_and_grad(lambda p:jnp.mean(m.apply({'params':p},*args,enable_dropout=False)[0]))(p)
        updates,state=optimizer.update(g,state,p)
        return optax.apply_updates(p,updates),state,loss,g
      losses=[]
      for _ in range(3):
        p,state,loss,g=step(p,state);losses.append(float(loss))
        for a in jax.tree.leaves(nn.unbox(g)):self.assertTrue(np.isfinite(a).all())
      self.assertLess(losses[-1],losses[0])

  def test_fused_projection_parity_and_cached_decode(self):
    c=self.config(scan_layers=False);m,mesh,p=self.init(c);args=self.inputs()
    plain=nn.unbox(p)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      expected=m.apply({'params':plain},*args,enable_dropout=False)[0]
      fused=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),fused_qkv=True,fused_mlp=True)))
      fm,_=self.model(fused);fp=copy.deepcopy(plain)
      for i in range(c.num_decoder_layers):
        layer=fp['decoder'][f'layers_{i}'];attn=layer['self_attention']
        qkv=[attn.pop(name) for name in ('query','key','value')]
        attn['qkv_proj']={k:jnp.stack([v[k] for v in qkv],axis=1 if k=='kernel' else 0) for k in ('kernel','bias')}
        layer['mlp']['wi']={k:jnp.expand_dims(v,axis=1 if k=='kernel' else 0) for k,v in layer['mlp']['wi'].items()}
      np.testing.assert_allclose(expected,fm.apply({'params':fp},*args,enable_dropout=False)[0],atol=2e-6)
      # Exercise the partial-RoPE cache through the complete decoder.
      short=tuple(a[:,:4] for a in args)
      outputs,cache=m.apply({'params':plain},*short,enable_dropout=False,
          model_mode=common_types.MODEL_MODE_PREFILL,mutable=['cache'])
      np.testing.assert_allclose(expected[:,:4],outputs[0],atol=2e-6)
      for i in range(4,8):
        token_args=tuple(a[:,i:i+1] for a in args[:4])
        outputs,cache=m.apply({'params':plain,**cache},*token_args,enable_dropout=False,
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,mutable=['cache'])
        np.testing.assert_allclose(expected[:,i:i+1],outputs[0],atol=2e-6)

  def test_torch_neox_reference(self):
    # Optional external dependency lives in its own interpreter, not the pinned JAX env.
    python=os.environ.get('PYTHIA_REFERENCE_PYTHON','/home/xd/miniconda3/envs/tune/bin/python')
    if not Path(python).exists():self.skipTest('set PYTHIA_REFERENCE_PYTHON to a torch+transformers interpreter')
    c=self.config(scan_layers=False);m,mesh,p=self.init(c);args=self.inputs()
    # Nonzero biases/scales exercise paths that zero initialization can conceal.
    flat=flatten_dict(nn.unbox(p))
    for k,a in list(flat.items()):
      if k[-1]=='bias':flat[k]=jnp.linspace(-.07,.09,a.size).reshape(a.shape)
      if k[-1]=='scale':flat[k]=jnp.linspace(.8,1.2,a.size).reshape(a.shape)
    p=unflatten_dict(flat)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      def call(p):return m.apply({'params':p},*args,enable_dropout=False,
          capture_intermediates=lambda mod,method:method=='__call__' and mod.__class__.__name__=='PythiaDecoderLayer',mutable=['intermediates'])
      (xent,_,pred),inter=call(p)
      hidden=inter['intermediates']['decoder']['layers_1']['__call__'][0][0]
      logits=m.apply({'params':p},hidden,method=m.logits_from_hidden_states)
      g=jax.grad(lambda p:jnp.mean(m.apply({'params':p},*args,enable_dropout=False)[0]))(p)
    source=Path(self.tmp.name)/'jax.npz';target=Path(self.tmp.name)/'torch.npz'
    np.savez(source,**{'/'.join(k):np.array(v) for k,v in flatten_dict(p).items()},tokens=np.array(args[0]),targets=np.array(args[2]),parallel_residual=np.array(c.pythia_parallel_residual))
    subprocess.run([python,str(Path(__file__).with_name('pythia_torch_reference.py')),str(source),str(target)],check=True,
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=''),timeout=120)
    ref=np.load(target)
    np.testing.assert_allclose(logits,ref['logits'],atol=3e-6,rtol=2e-5)
    np.testing.assert_allclose(xent,ref['xent'],atol=3e-6,rtol=2e-5)
    for k,a in flatten_dict(g).items():np.testing.assert_allclose(a,ref['grad/'+'/'.join(k)],atol=3e-6,rtol=5e-4,err_msg=str(k))
    print('PYTHIA_NEOX_PARITY_OK max_logit_error',float(np.max(np.abs(logits-ref['logits']))),flush=True)


if __name__=='__main__':unittest.main()
