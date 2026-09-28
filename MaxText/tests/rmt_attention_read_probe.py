"""Independent equations and all gradients for the complete NoO attention read."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from layers.rmt_pallas_attention_read import attention_read
from tests.rmt_fused_write_read_probe import norm


def reference(m,s,c,scale,w,bb,gb,pos):
  h=16;rd=18
  x=norm(m[...,:h,:].reshape(m.shape[:2]+(1200,)))*scale
  p=jnp.dot(x,w)
  basis=p[...,:128].reshape(m.shape[:2]+(4,32))+bb
  mix=p[...,128:256].reshape(m.shape[:2]+(32,4))
  qg=jax.nn.sigmoid(p[...,256:288]+gb[:32])
  vk=p[...,288:416].reshape(m.shape[:2]+(16,8))
  vg=jax.nn.sigmoid(p[...,416:432]+gb[32:])
  rp=p[...,432:].reshape(m.shape[:2]+(32,18))
  static=jnp.einsum('btkv,hk->bthv',m,s)
  compressed=jnp.einsum('btkv,kr->btrv',m[...,16:,:],c)
  bread=jnp.einsum('btrc,btcv->btrv',basis,m[...,16:,:])
  key=jnp.einsum('bthr,btrc->bthc',mix.astype(jnp.float32),basis.astype(jnp.float32))
  inv=jax.lax.rsqrt(jnp.mean(key*key,axis=-1)+1e-6).astype(m.dtype)
  value=jnp.einsum('bthr,btrv->bthv',mix,bread)
  qk=static[...,:32,:]+value*(inv*(.2*qg))[...,None]
  timescale=10000.**(jnp.arange(9,dtype=jnp.float32)/9)
  phase=pos[:,:,None,None]/timescale
  cs=jnp.cos(phase).astype(m.dtype);sn=jnp.sin(phase).astype(m.dtype)
  a,b=rp[...,:9],rp[...,9:]
  rotated=jnp.concatenate((a*cs-b*sn,b*cs+a*sn),axis=-1)
  qk=jnp.concatenate((qk[...,:57],rotated),axis=-1)
  v=static[...,32:,:]+jnp.einsum('bthr,btrv->bthv',norm(vk)*(.2*vg)[...,None],compressed)
  return jnp.concatenate((qk[...,:16,:]/jnp.sqrt(75),qk[...,16:,:],v),axis=-2),x


def main():
  p=argparse.ArgumentParser();p.add_argument('--interpret',action='store_true')
  p.add_argument('--dtype',default='float32');p.add_argument('--tokens',type=int,default=128)
  p.add_argument('--reverse-tile',type=int,default=128)
  p.add_argument('--save-small',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();dt=getattr(jnp,a.dtype);t=a.tokens
  shapes=[(1,t,48,75),(48,48),(32,8),(1200,),(1200,1008),(4,32),(48,)]
  x=[jax.random.normal(jax.random.key(2000+i),sh,dtype=dt) for i,sh in enumerate(shapes)]
  for i in (1,2,4):x[i]*=.03
  x[3]=1+x[3]*.02;x[5]*=.1;x[6]=x[6]*.1-3
  pos=(jnp.arange(t,dtype=jnp.int32)+100)[None,:]
  rf=lambda *z:reference(*z,pos)
  expected=jax.jit(rf)(*x)
  dy=tuple(jax.random.normal(jax.random.key(2100+i),v.shape,dtype=dt) for i,v in enumerate(expected))
  bg=jax.jit(lambda *z:jax.vjp(rf,*z)[1](dy))(*x)
  fn=lambda *z:attention_read(*z,pos,interpret=a.interpret,reverse_tile=a.reverse_tile,save_small=a.save_small)
  actual=jax.jit(fn)(*x);ag=jax.jit(lambda *z:jax.vjp(fn,*z)[1](dy))(*x)
  errors=[float(jnp.linalg.norm(u.astype(jnp.float32)-v.astype(jnp.float32))/jnp.linalg.norm(v.astype(jnp.float32)))
          for u,v in zip((*actual,*ag),(*expected,*bg))]
  result={'relative_l2':errors};print(json.dumps(result),flush=True)
  Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
  assert np.isfinite(errors).all() and max(errors)<(5e-5 if dt==jnp.float32 else .04),errors


if __name__=='__main__':main()
