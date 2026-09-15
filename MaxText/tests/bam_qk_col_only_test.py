"""Column-only QK keeps the parent's column operator and removes actual row parameters."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import core
import max_utils
import bam_local_fetch_test
from layers.attentions import BamAttention, _packed_local_layout, _transform_bam_read_key


class QKColOnlyTest(unittest.TestCase):
  config = bam_local_fetch_test.LocalFetchTest.config

  def test_mapped_column_forward_and_gradient(self):
    base = 'BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis'
    cfg = self.config(base)
    cfg.get_keys().update(dtype=jnp.float32,head_dim=128,emb_dim=256)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    kwargs=dict(config=cfg,num_query_heads=2,num_kv_heads=2,head_dim=128,
        bam_k=64,bam_v=32,max_target_length=8,max_prefill_predict_length=8,
        mesh=mesh,attention_kernel='dot_product_chunk',dtype=jnp.float32,
        layer_mode='local_qk+local_o',attention_type=cfg.attention_type)
    full=BamAttention(**kwargs)
    x=jax.random.normal(jax.random.key(1),(1,8,256))
    m=jax.random.normal(jax.random.key(2),(1,8,64,32))
    args=(x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
    call=dict(M_in=m,deterministic=True,layer_index=2)
    p=core.unfreeze(full.init({'params':jax.random.key(3)},*args,**call)['params'])
    arms=full.apply({'params':p},method=lambda mod:list(mod._local_arms.values()))
    old,_=_packed_local_layout(arms,True)
    pname=cfg.bam_local_packed_parameter_name or 'W_local_packed'
    leaf=p[pname]['kernel']
    p[pname]['kernel']=leaf.replace(value=.02*jax.random.normal(jax.random.key(4),leaf.value.shape))
    bias=p['W_lq_bias']
    p['W_lq_bias']=bias.replace(value=.01*jax.random.normal(jax.random.key(5),bias.value.shape))

    def compact(params):
      out=core.unfreeze(params)
      value=out[pname]['kernel'].value
      pieces=[]
      for arm,(bs,gs,hs) in zip(arms,old):
        lead=value.shape[:-1]
        if arm.name in ('q','k'):
          if arm.name=='q':
            pieces.append(value[:,bs].reshape(lead+arm.key_shape)[...,64:].reshape(lead+(-1,)))
          pieces.extend((value[:,gs].reshape(lead+arm.gate_shape)[...,1],
                         value[:,hs].reshape(lead+arm.mix_shape)[...,1,:].reshape(lead+(-1,))))
        else:
          pieces.extend(value[:,s] for s in (bs,gs,hs))
      out[pname]['kernel']=out[pname]['kernel'].replace(value=jnp.concatenate(pieces,-1))
      out['W_lq_bias']=out['W_lq_bias'].replace(value=out['W_lq_bias'].value[...,64:])
      for name in ('q','k'):
        leaf=out[f'W_l{name}_gate_b0']
        out[f'W_l{name}_gate_b0']=leaf.replace(value=leaf.value[...,1])
      return out

    # Separate config avoids mutating the reference module's definition.
    cfg2=self.config('BamXLSharedBasisQKColOnlyMLP')
    cfg2.get_keys().update(dtype=jnp.float32,head_dim=128,emb_dim=256)
    small=BamAttention(**dict(kwargs,config=cfg2))
    psmall=compact(p)
    init=small.init({'params':jax.random.key(3)},*args,**call)['params']
    self.assertEqual([z.shape for z in jax.tree.leaves(init)],
                     [z.shape for z in jax.tree.leaves(psmall)])
    def reads(mod,inputs,state):
      projected=mod._local_inputs(inputs)
      cache=mod._shared_qk_basis(state,projected)
      return tuple(mod._read_local(n,state,inputs,projected,cache)[...,:64] for n in ('q','k'))
    def result(params,state,pruned):
      return (small.apply({'params':compact(params)},x,state,method=reads) if pruned
              else full.apply({'params':params},x,state,method=reads))
    for a,b in zip(result(p,m,False),result(p,m,True)):
      np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-5)
    def loss(params,state,pruned):
      return sum(jnp.mean(y*y) for y in result(params,state,pruned))
    for a,b in zip(jax.tree.leaves(jax.grad(loss,(0,1))(p,m,False)),
                   jax.tree.leaves(jax.grad(loss,(0,1))(p,m,True))):
      np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-5)

  def test_direct_packed_shapes_and_active_gradient(self):
    cfg=self.config('BamXLSharedBasisQKDirectC8MLP')
    cfg.get_keys().update(dtype=jnp.float32,head_dim=128,emb_dim=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    mod=BamAttention(config=cfg,num_query_heads=2,num_kv_heads=2,head_dim=128,bam_k=64,bam_v=32,
        max_target_length=8,max_prefill_predict_length=8,mesh=mesh,
        attention_kernel='dot_product_chunk',dtype=jnp.float32,
        layer_mode='local_qk+full',attention_type=cfg.attention_type)
    x=jax.random.normal(jax.random.key(6),(1,8,256))
    m=jax.random.normal(jax.random.key(7),(1,8,64,32))
    args=(x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
    call=dict(M_in=m,deterministic=True,layer_index=2)
    p=core.unfreeze(mod.init({'params':jax.random.key(8)},*args,**call)['params'])
    self.assertEqual(p['W_lq_bias'].value.shape,(2,8))
    self.assertEqual(p['W_lk_bias'].value.shape,(2,8))
    leaf=p['W_lq_bias']
    p['W_lq_bias']=leaf.replace(value=.03*jax.random.normal(jax.random.key(9),leaf.value.shape))
    def read(mod):
      inputs=mod._local_inputs(x)
      mc=mod._compress_full_fetch_state(m)
      return mod._read_local('q',mc,x,inputs)
    def reference(mod):
      keys,gates,_=mod._local_inputs(x)['q']
      keys=keys+mod.W_lq_bias
      gates=gates+mod.W_lq_gate_b0
      keys=_transform_bam_read_key(keys,'rms_gate',mod._local_key_scales['q'],
          rms_epsilon=mod._read_key_epsilon,rms_statistics_dtype=mod._read_rms_statistics_dtype,
          gate_logits=gates[...,None])
      return jnp.einsum('btkc,btnc->btnk',mod._compress_full_fetch_state(m),keys)
    actual=mod.apply({'params':p},method=read)
    expected=mod.apply({'params':p},method=reference)
    np.testing.assert_allclose(actual[...,:64],expected,rtol=2e-5,atol=2e-5)
    np.testing.assert_array_equal(actual[...,64:],0.)
    grad=jax.grad(lambda params:jnp.sum(mod.apply({'params':params},method=read)))(p)
    name=cfg.bam_local_packed_parameter_name or 'W_local_packed'
    self.assertGreater(float(jnp.linalg.norm(grad[name]['kernel'].value)),0.)
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(z))) for z in jax.tree.leaves(grad)))


if __name__=='__main__': unittest.main()
