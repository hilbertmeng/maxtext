"""Compare with real Praxis, removing exactly its logits-to-FP32 statement."""
import inspect,textwrap
import numpy as np
import jax,jax.numpy as jnp
from absl import flags
import mudd_spectral_praxis as compat
from praxis.layers import embedding_softmax
import max_utils
from mudd_softmax_precision_probe import project,pax_fp32,pax_bf16,maxtext_train_bf16,bf16_log_probs

if not flags.FLAGS.is_parsed():flags.FLAGS(['softmax_precision_test'])
source=textwrap.dedent(inspect.getsource(embedding_softmax.FullSoftmax._compute_logits_and_loss));needle='logits = logits.astype(jnp.float32)';assert source.count(needle)==1
env=dict(vars(embedding_softmax));exec(source.replace(needle,'',1),env)
Changed=type('BF16FullSoftmax',(embedding_softmax.FullSoftmax,),{'_compute_logits_and_loss':env['_compute_logits_and_loss']})
h=jnp.asarray(np.random.default_rng(12).normal(size=(1,8,4)),jnp.bfloat16);w=jnp.asarray(np.random.default_rng(13).normal(size=(4,32)),jnp.bfloat16);z=project(h,w);z.block_until_ready();y=jnp.arange(8)[None];weights=jnp.ones((1,8,1),jnp.float32)
assert z.dtype==jnp.bfloat16 and bf16_log_probs(z).dtype==jnp.bfloat16
with compat.base_layer.JaxContext.new_context(hparams=compat.base_layer.JaxContext.HParams(do_eval=True)):
 for cls,fn in [(embedding_softmax.FullSoftmax,pax_fp32),(Changed,pax_bf16)]:
  p=compat.pax_fiddle.Config(cls,name='softmax',input_dims=32,num_classes=32,feed_forward_tpl=None,fprop_dtype=jnp.bfloat16,dtype=jnp.float32)
  m=compat.base_layer.instantiate(p);v=m.init(jax.random.PRNGKey(1),z,weights,class_ids=y[...,None]);actual=jax.jit(lambda zz,yy:m.apply(v,zz,weights,class_ids=yy[...,None]).per_example_xent)(z,y);expected=fn(z,y)
  assert actual.dtype==expected.dtype==jnp.float32;np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
target=jax.nn.one_hot(y,32,dtype=jnp.float32)
value,_=jax.jit(jax.value_and_grad(lambda q:max_utils.cross_entropy_with_logits(q,target,0.)[0].mean()))(z)
np.testing.assert_allclose(value,maxtext_train_bf16(z,y).mean(),rtol=1e-6,atol=1e-6)
print('REAL_PRAXIS_AND_SINGLE_CAST_REMOVAL_AND_MAXTEXT_TRAIN_FORWARD_PASS',flush=True)
