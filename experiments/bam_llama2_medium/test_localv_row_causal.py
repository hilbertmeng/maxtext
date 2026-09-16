"""CPU checks for diagnostic-only intervention semantics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'MaxText/tests'))
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import core
from flax.traverse_util import flatten_dict, unflatten_dict
import bam_local_fetch_test
import localv_row_causal as probe
from layers.attentions import BamAttention
import max_utils


class CausalTest(unittest.TestCase):
  def test_transport(self):
    keys = jax.random.split(jax.random.key(22), 3)
    a = jax.nn.softmax(jax.random.normal(keys[0], (1,2,3,6)), -1)
    v = jax.random.normal(keys[1], (1,6,2,4))
    v0 = jax.random.normal(keys[2], v.shape)
    y = jnp.einsum('bnqs,bsnd->bqnd', a, v)
    y0 = jnp.einsum('bnqs,bsnd->bqnd', a, v0)
    diagonal = (jnp.arange(6)[None,:] == jnp.arange(3,6)[:,None])[None,None]
    self_part = jnp.einsum('bnqs,bsnd->bqnd', a*diagonal, v-v0)
    cross_part = jnp.einsum('bnqs,bsnd->bqnd', a*(~diagonal), v-v0)
    np.testing.assert_array_equal(probe.transport_edit(y,y0,a,v,v0,3,0,jnp.ones(2)),y)
    np.testing.assert_array_equal(probe.transport_edit(y,y0,a,v,v0,3,0,jnp.zeros(2)),y0)
    np.testing.assert_allclose(probe.transport_edit(y,y0,a,v,v0,3,0,jnp.array([0.,1.])),y-self_part,atol=1e-6)
    np.testing.assert_allclose(probe.transport_edit(y,y0,a,v,v0,3,0,jnp.array([1.,0.])),y-cross_part,atol=1e-6)

  def test_actual_module(self):
    c = bam_local_fetch_test.LocalFetchTest.config(self, probe.BASE)
    c.get_keys().update(dtype=jnp.float32, query_chunk_size=4)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    model = BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=128,
        bam_k=64,bam_v=32,max_target_length=8,max_prefill_predict_length=8,mesh=mesh,
        attention_kernel='dot_product_chunk',dtype=jnp.float32,
        layer_mode='local_qk+local_v+local_o',attention_type=c.attention_type)
    x = jax.random.normal(jax.random.key(2),(1,8,256))
    m = jax.random.normal(jax.random.key(3),(1,8,64,32))
    args = (x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32))
    kwargs = dict(M_in=m,deterministic=True,layer_index=1)
    variables = model.init({'params':jax.random.key(4),'aqt':jax.random.key(5)},*args,**kwargs)
    # Activate the zero-initialized read projection; retain all tree/partition metadata.
    flat = flatten_dict(core.unfreeze(variables))
    for path,value in list(flat.items()):
      if 'W_R' in path and path[-1]=='kernel':
        raw = value.value if hasattr(value,'value') else value
        raw = .03*jax.random.normal(jax.random.key(6),raw.shape)
        flat[path] = value.replace(value=raw) if hasattr(value,'value') else raw
    variables = core.freeze(unflatten_dict(flat))
    native = model.apply(variables,*args,**kwargs)
    def run(scales,route):
      with probe.interventions(jnp.asarray(scales),route):
        return model.apply(variables,*args,**kwargs)
    for route in (False,True):
      for a,b in zip(jax.tree.leaves(native),jax.tree.leaves(run(np.ones((24,5)),route))):
        np.testing.assert_array_equal(a,b)
    d = np.ones((24,5)); d[1,0]=0
    r = np.ones((24,5)); r[1,1:3]=0
    direct,transport = run(d,False),run(r,True)
    for a,b in zip(jax.tree.leaves(direct),jax.tree.leaves(transport)):
      np.testing.assert_allclose(a,b,atol=2e-6,rtol=1e-6)
    self.assertGreater(float(jnp.max(jnp.abs(direct[0]-native[0]))),1e-7)


if __name__ == '__main__':
  unittest.main()
