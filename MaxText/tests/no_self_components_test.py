"""Independent native Pallas branch-off checks for dynamic/static O decomposition."""
import os
import jax,jax.numpy as jnp
import numpy as np
from layers import bam_pallas
from no_self_o_components import split,stats

def main():
 dt=jnp.bfloat16 if os.environ.get('TEST_DTYPE','bfloat16')=='bfloat16' else jnp.float32;rng=np.random.default_rng(81027);B,T,H,C,K=1,128,8,8,16
 rand=lambda shape,scale=1:jnp.asarray(rng.normal(size=shape)*scale,dt)
 m=rand((B,T,C,K),.5);st=rand((4*H,C),.2);q=rand((B,T,H,C));k=rand(q.shape);vo=rand((B,T,2*H,C));ol=rand((B,T,H));vl=rand(ol.shape);qstd=rand((B,T,H,4));kstd=rand(qstd.shape);g=tuple(jax.nn.sigmoid(rand(ol.shape)).astype(dt) for _ in range(4));shift=jax.nn.sigmoid(rand((B,T,C))).astype(dt).at[:,0].set(0)
 args=(m,st,q,ol,k,vl,vo,ol,vl,qstd,kstd);kw=dict(qk_cols=12,read_epsilon=1e-6,key_scale=.7,static_gates=g,shift=shift,body='v8t',interpret=True,forward_tile=128,reverse_tile=128,vmem_mib=60)
 def run(*a,**k):return bam_pallas.read(*a,**k)[3]
 native=jax.jit(lambda:run(*args,**kw))();d,s,recon=jax.jit(lambda:split(args,kw))();ids=np.unique(np.linspace(64,127,64,dtype=np.int32));target=np.asarray(native[:,ids],float)
 rel=lambda x,y:float(np.sqrt(np.sum((np.asarray(x,float)-np.asarray(y,float))**2)/np.sum(np.asarray(y,float)**2)))
 assert rel(recon,target)<(.008 if dt==jnp.bfloat16 else 2e-5);assert rel(d+s,target)<(.012 if dt==jnp.bfloat16 else 2e-5)
 dynamic_only=tuple(jnp.zeros_like(g[3]) if i==3 else v for i,v in enumerate(g));dn=jax.jit(lambda:run(*args,**dict(kw,static_gates=dynamic_only)))()[:,ids]
 static_args=list(args);static_args[7]=jnp.full_like(ol,-1e6);sn=jax.jit(lambda:run(*static_args,**kw))()[:,ids]
 assert rel(d,dn)<(.012 if dt==jnp.bfloat16 else 2e-5);assert rel(s,sn)<(.012 if dt==jnp.bfloat16 else 2e-5)
 # Verify moment contraction against an explicit head/output projection.
 x=jnp.stack([d,s,d+s,target],3);w=rand((H,K,13)).astype(jnp.float32);pre,heads,total=stats(x,w);projected=jnp.einsum('bqhsk,hkd->bqhsd',x,w);expected=jnp.einsum('bqhsd,bqhtd->bhst',projected,projected);np.testing.assert_allclose(heads,expected,rtol=2e-5,atol=.001)
 print('NOSELF_O_COMPONENTS_NATIVE_BRANCH_TOGGLES_PASS',dt,'sumclosure',rel(d+s,target),'native_reconstruction',rel(recon,target),flush=True)
if __name__=='__main__':main()
