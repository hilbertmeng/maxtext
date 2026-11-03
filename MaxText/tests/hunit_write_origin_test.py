"""Native interpreted tiny model: hooks preserve outputs and scanned accounting closes."""
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from unittest import mock
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from layers.models import Transformer
import max_utils
from bam_fast_kernel_test import FastKernelTest,FAST
from hunit_write_origin import WriteObserver,LastReadAblation,aggregate
from head_spectral_diagnostic import precise_cross_entropy

def main():
 t=FastKernelTest();t.setUp()
 try:
  c=t.tiny(FAST,dtype='bfloat16');c.get_keys().update(bam_pallas_core=True,bam_pallas_write=True,bam_splash_attention=True,bam_pallas_fused_inputs=True,bam_pallas_interpret=True,bam_pallas_forward_tile=16,bam_pallas_read_tile=16,bam_pallas_write_tile=16,remat_policy='none')
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quant=None);x=(jnp.arange(128)%32)[None];y=(x+1)%32;pos=jnp.arange(128)[None];seg=jnp.ones_like(x);original=splash.make_splash_mha
  def interpret(*args,**kw):return original(*args,**dict(kw,interpret=True))
  max_utils.cross_entropy_with_logits=precise_cross_entropy
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=interpret):
   v=nn.unbox(model.init(jax.random.key(9),x,pos,y,seg,enable_dropout=False));gamma=v['params']['decoder']['lm_head']['decoder_norm']['scale'];bias=v['params']['decoder']['lm_head']['decoder_norm']['bias'];u=jnp.asarray(np.linalg.qr(np.random.default_rng(7).normal(size=(160,8)))[0],jnp.float32)
   native=jax.jit(lambda p:model.apply(p,x,pos,y,seg,enable_dropout=False)[0])(v)
   @jax.jit
   def observed(p):
    observer=WriteObserver(u,gamma)
    with observer.observe():out,collection=model.apply(p,x,pos,y,seg,enable_dropout=False,mutable=['intermediates'])
    return out[0],aggregate(observer,collection,gamma,bias)
   loss,s=observed(v);np.testing.assert_array_equal(loss,native);assert s['head_allocation'].shape==(3,7)
   print('LEDGER_CHECK', {k:np.asarray(s[k]).tolist() for k in ['head_allocation_closure','raw_allocation_closure','projection_ledger_relative_error','layer_continuity_max','head_total_power','raw_total_power']},flush=True)
   assert abs(float(s['head_allocation_closure']))<2e-5 and abs(float(s['raw_allocation_closure']))<2e-5
   assert float(s['projection_ledger_relative_error'])<1e-5 and float(s['layer_continuity_max'])==0
   @jax.jit
   def final_only(p):
    observer=WriteObserver(u,gamma,last_only=True)
    with observer.observe():out,collection=model.apply(p,x,pos,y,seg,enable_dropout=False,mutable=['intermediates'])
    return out[0],aggregate(observer,collection,gamma,bias)
   fl,fs=final_only(v);np.testing.assert_array_equal(fl,native);assert fs['head_allocation'].shape==(1,7) and abs(float(fs['head_allocation_closure']))<2e-5
   @jax.jit
   def controlled(p,mode):
    observer=LastReadAblation(mode)
    with observer.observe():out=model.apply(p,x,pos,y,seg,enable_dropout=False)
    return out[0],observer.h
   keep,kh=controlled(v,jnp.asarray(0));np.testing.assert_array_equal(keep,native)
   for mode in [1,2]:
    loss,hh=controlled(v,jnp.asarray(mode));assert np.isfinite(loss).all() and np.isfinite(hh).all()
   print('NATIVE_INTERPRETED_OUTPUT_EQUAL_SCAN_LEDGER_AND_LN_ALLOCATION_PASS',flush=True)
 finally:t.tearDown()
if __name__=='__main__':main()
