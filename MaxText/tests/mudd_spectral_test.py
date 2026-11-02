"""Independent native-feature, MUDD-zero-init and vocabulary-gradient contracts."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
from absl import flags
import mudd_spectral_capture as cap
import mudd_spectral_praxis as old
if not flags.FLAGS.is_parsed():flags.FLAGS(['mudd_spectral_tests'])

def test_old():
 p=old.configuration(d=80,nlayers=3,nheads=2,head_dim=40,vocab=128,small=True)
 full=old.base_layer.instantiate(p);p.skip_compute_loss=True
 m=old.base_layer.instantiate(p);x=jnp.asarray(np.random.default_rng(2).integers(0,128,(1,8)),jnp.int32)
 kw=dict(segment_ids=jnp.ones_like(x),segment_pos=jnp.broadcast_to(jnp.arange(8),x.shape))
 basis=jnp.asarray(np.linalg.qr(np.random.default_rng(8).normal(size=(80,4)))[0],jnp.float32)
 def forward(v,observe):
  c=cap.Capture('old',basis,8,3)
  with old.base_layer.JaxContext.new_context(hparams=old.base_layer.JaxContext.HParams(do_eval=True)):
   if observe:
    with nn.intercept_methods(c.interceptor):y=m.apply(v,x,jnp.zeros(x.shape,jnp.bfloat16),**kw)
    return y,c.result()
   return m.apply(v,x,jnp.zeros(x.shape,jnp.bfloat16),**kw)
 with old.base_layer.JaxContext.new_context(hparams=old.base_layer.JaxContext.HParams(do_eval=True)):
  v=full.init(jax.random.PRNGKey(7),x,jnp.zeros(x.shape,jnp.bfloat16),**kw)
 for mutate in [False,True]:
  if mutate:
   f=flatten_dict(v['params'])
   for k,a in f.items():
    if 'dynamic_dense_conn2' in k[-1]:f[k]=jnp.asarray(np.random.default_rng(len(k[-1])).normal(0,.12,a.shape),a.dtype)
   v=dict(v,params=unflatten_dict(f))
  native=forward(v,False);h,r=forward(v,True)
  np.testing.assert_array_equal(h,native);np.testing.assert_array_equal(h,r['h'])
  energy=np.asarray(r['composition'])
  if not mutate:np.testing.assert_array_equal(energy[...,1],0)
  else:assert energy[...,1].max()>0
  np.testing.assert_allclose(energy[...,2],energy[...,0]+energy[...,1]+2*energy[...,3],atol=1e-5,rtol=1e-5)
  norm=cap.final_norm(r['raw'],v['params']['final_ln']['scale'],v['params']['final_ln']['bias'],'old')
  np.testing.assert_array_equal(norm,h)
  w=v['params']['softmax']['logits_ffn']['linear']['w'];targets=(x+1)%128
  with old.base_layer.JaxContext.new_context(hparams=old.base_layer.JaxContext.HParams(do_eval=True)):
   logits=m.apply(v,h,method=lambda module,h:module.softmax.get_logits(h))
  np.testing.assert_array_equal(cap.native_head_ce(h,w,targets),cap.cross_entropy(logits,targets))
 print('OLD_NATIVE_CAPTURE_ZERO_INIT_LEARNED_EXTRA_NORM_HEAD_PASS',flush=True)

def test_gradient():
 rng=np.random.default_rng(17);h=jnp.asarray(rng.normal(size=(1,8,20)),jnp.float32);w=jnp.asarray(rng.normal(size=(20,40)),jnp.float32);y=jnp.arange(8)[None]%40
 basis=jnp.asarray(np.linalg.qr(rng.normal(size=(20,20)))[0],jnp.float32)
 grad,ce,s=cap.readout_gradient(h,w,y,basis[:,:1]);wf=w.astype(jnp.bfloat16).astype(jnp.float32)
 oracle=jax.grad(lambda q:jnp.mean(cap.cross_entropy(jnp.einsum('btd,dv->btv',h,q,precision=jax.lax.Precision.HIGHEST),y)))(wf)
 np.testing.assert_allclose(grad,oracle,atol=1e-6,rtol=2e-5)
 st=cap.stage_stats(h,h*.8,h,basis,basis)
 np.testing.assert_allclose(st['spectral_energy'].sum(-1),st['stage_total2'],rtol=2e-6,atol=1e-6)
 assert float(s[2])<=float(s[1]);print('HEAD_GRADIENT_AUTODIFF_AND_SPECTRAL_PARSEVAL_PASS',flush=True)

def test_serial():
 import pythia_test as reference
 import max_utils
 t=reference.PythiaTest();t.setUp()
 try:
  c=t.config(exp_class='MuddPythiaMediumPropT2048SerialMLP',base_num_decoder_layers=3,base_mlp_dim=256,
   pythia_rotary_dim=4,mudd_router_hidden_expand=[1,1,4],scan_layers=False,dtype='bfloat16',remat_policy='none')
  m,mesh,p=t.init(c)
  p=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,p,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
  f=flatten_dict(p)
  for key,a in f.items():
   if key[-2:]==('dynamic_dense_conn2','kernel'):f[key]=jnp.asarray(np.random.default_rng(a.size).normal(0,.12,a.shape),a.dtype)
  p=unflatten_dict(f);args=t.inputs();basis=jnp.asarray(np.linalg.qr(np.random.default_rng(1).normal(size=(32,4)))[0],jnp.float32)
  max_utils.cross_entropy_with_logits=cap.precise_cross_entropy
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   native=m.apply({'params':p},*args,enable_dropout=False)[0]
   observer=cap.Capture('serial',basis,8,3)
   with nn.intercept_methods(observer.interceptor):observed_native,col=m.apply({'params':p},*args,enable_dropout=False,mutable=['intermediates'])
   r=observer.result(col['intermediates'])
  ln=p['decoder']['lm_head']['decoder_norm'];np.testing.assert_array_equal(r['h'],cap.final_norm(r['raw'],ln['scale'],ln['bias'],'serial'))
  w=p['decoder']['lm_head']['logits_dense'];replay=cap.native_head_ce(r['h'],w,args[2]);np.testing.assert_allclose(replay,native,atol=1e-6,rtol=1e-6)
  np.testing.assert_array_equal(observed_native[0],native)
  e=np.asarray(r['composition']);np.testing.assert_allclose(e[...,2],e[...,0]+e[...,1]+2*e[...,3],rtol=1e-5,atol=1e-6)
  assert e[...,1].max()>0
  print('SERIAL_NATIVE_HEAD_CAPTURE_AND_LEARNED_EXTRA_PASS',flush=True)
 finally:t.tearDown()

def test_common_shift_rounding():
 # Softmax translation invariance holds before rounding; large common offsets
 # can destroy logit differences if BF16 rounding happens first.
 z=jnp.array([[[1024.,1024.25,1024.5]]],jnp.float32);y=jnp.array([[2]])
 c=z-z.mean(-1,keepdims=True)
 np.testing.assert_allclose(cap.cross_entropy(z,y),cap.cross_entropy(c,y),atol=1e-4)
 raw=cap.cross_entropy(z.astype(jnp.bfloat16),y);cent=cap.cross_entropy(c.astype(jnp.bfloat16),y)
 assert float(raw[0,0])>float(cent[0,0])+.1
 a=jnp.asarray(np.random.default_rng(37).normal(size=1024)*100,jnp.float32)
 np.testing.assert_array_equal(cap.explicit_bf16_round(a),a.astype(jnp.bfloat16).astype(jnp.float32))
 print('SOFTMAX_COMMON_SHIFT_AND_BF16_ROUNDING_PASS',flush=True)

if __name__=='__main__':test_old();test_gradient();test_serial();test_common_shift_rounding()
