import numpy as np
import jax.numpy as jnp
from head_attention_capture import statistics,METRICS,positions
q=jnp.zeros((2,128,3,4));k=q;cats=jnp.zeros((2,128),jnp.int32).at[:,10].set(1).at[:,20].set(3)
a,m=statistics(q,k,cats)
a=np.array(a)
ix={name:i for i,name in enumerate(METRICS)}
pos=positions(128)
np.testing.assert_allclose(a[...,ix['first']],np.broadcast_to(1/(pos+1),(2,3,len(pos))),rtol=1e-5)
np.testing.assert_allclose(a[...,ix['entropy_normalized']],1,atol=1e-6)
np.testing.assert_allclose(a[...,ix['content_entropy_normalized']],1,atol=1e-6)
np.testing.assert_allclose(a[...,ix['content']]+a[...,ix['sink_union']],1,atol=1e-6)
np.testing.assert_allclose(a[...,ix['content']],a[...,ix['content_uniform_share']],atol=1e-6)
# A synthetic head permanently attracted to key0 must not count as content-active.
k=k.at[:,0,:,0].set(100);q=q.at[:,:,:,0].set(1)
a,_=statistics(q,k,cats);a=np.array(a)
assert np.min(a[...,ix['first']])>.999
assert np.max(a[...,ix['content']])<1e-30
assert np.isfinite(a).all()
# Local causal mask removes key0 and restricts valid source support.
a,m=statistics(q,k,cats,window=16);m=np.array(m,dtype=np.float32)
assert np.max(m[...,0])==0
assert np.isfinite(np.array(a)).all()
print('UNIFORM_SINK_LOCAL_MASK_ENTROPY_CHECKS_PASSED')
