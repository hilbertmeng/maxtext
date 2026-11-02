"""Read-only vocabulary-head spectra and native BF16 checkpoint probes."""
import ast,concurrent.futures,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import tensorstore as ts
import jax,jax.numpy as jnp
_IO_CONTEXT=ts.Context({'cache_pool#ocdbt':{'total_bytes_limit':268435456}})

def restore(root,head_only=False,cache_dir=None):
 meta=json.loads(subprocess.check_output(['gsutil','cat',root+'/_METADATA'],text=True));wanted={}
 for k,v in meta['tree_metadata'].items():
  path=ast.literal_eval(k)
  if path[:2]!=('params','params') or v['value_metadata'].get('skip_deserialize',False):continue
  key=path[2:]
  if not head_only or key[:2]==('decoder','lm_head'):wanted[key]=path
 cache=Path(cache_dir) if cache_dir else None
 if cache:
  cache.mkdir(parents=True,exist_ok=True)
  identity=dict(checkpoint=root,metadata_sha256=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest())
  marker=cache/'identity.json'
  if marker.exists():assert json.loads(marker.read_text())==identity
  else:marker.write_text(json.dumps(identity,indent=2)+'\n')
 def read(p):
  path=cache/(hashlib.sha256(repr(p).encode()).hexdigest()+'.npy') if cache else None
  if path is not None and path.exists():return np.load(path,mmap_mode='r',allow_pickle=False)
  a=ts.open({'driver':'zarr3' if meta.get('use_zarr3') else 'zarr','kvstore':{'driver':'ocdbt','base':root+'/','cache_pool':'cache_pool#ocdbt'},'path':'.'.join(p)},open=True,read=True,context=_IO_CONTEXT).result()
  value=np.asarray(a.read().result())
  if path is not None:
   temp=path.with_suffix('.tmp')
   with temp.open('wb') as f:np.save(f,value,allow_pickle=False)
   temp.replace(path)
  return value
 step=int(read(('step',)))
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:flat=dict(zip(wanted,pool.map(read,wanted.values())))
 if head_only:assert len(flat)==3
 return flat,dict(checkpoint=root,state_step=step,parameter_count=sum(v.size for v in flat.values()),leaves=len(flat))

def abstract_params(root):
 """Read global array metadata only, for real-shape backend compile preflight."""
 meta=json.loads(subprocess.check_output(['gsutil','cat',root+'/_METADATA'],text=True));wanted=[]
 for key,value in meta['tree_metadata'].items():
  p=ast.literal_eval(key)
  if p[:2]==('params','params') and not value['value_metadata'].get('skip_deserialize',False):wanted.append(p)
 def shape(p):
  a=ts.open({'driver':'zarr3' if meta.get('use_zarr3') else 'zarr','kvstore':{'driver':'ocdbt','base':root+'/','cache_pool':'cache_pool#ocdbt'},'path':'.'.join(p)},open=True,read=True,context=_IO_CONTEXT).result()
  return p[2:],jax.ShapeDtypeStruct(a.shape,a.dtype.numpy_dtype)
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:return dict(pool.map(shape,wanted))

def precise_cross_entropy(logits,targets,z_loss=0.):
 z=logits.astype(jnp.float32);ce=-jnp.sum(targets.astype(jnp.float32)*jax.nn.log_softmax(z),-1)
 aux=jnp.asarray(z_loss,jnp.float32)*jax.scipy.special.logsumexp(z,-1)**2
 return ce+aux,aux

def native_ce(h,w,y):
 z=jnp.tensordot(h,w.astype(jnp.bfloat16),axes=((-1,),(0,)))
 return precise_cross_entropy(z,jax.nn.one_hot(y,w.shape[-1]))[0]

def FP32_readout(h,w,y):
 hf=h[0].astype(jnp.float32);wf=w.astype(jnp.bfloat16).astype(jnp.float32)
 z=jnp.einsum('td,dv->tv',hf,wf,precision=jax.lax.Precision.HIGHEST);e=jax.nn.softmax(z).at[jnp.arange(y.shape[1]),y[0]].add(-1)
 g=jnp.einsum('td,tv->dv',hf,e,precision=jax.lax.Precision.HIGHEST)/y.shape[1]
 ce=-jnp.take_along_axis(jax.nn.log_softmax(z),y[0,:,None],-1)[:,0]
 return g,ce,z

def BF16_round(z):
 b=jax.lax.bitcast_convert_type(z.astype(jnp.float32),jnp.uint32)
 b=(b+jnp.uint32(0x7fff)+((b>>jnp.uint32(16))&1))&jnp.uint32(0xffff0000)
 return jax.lax.bitcast_convert_type(b,jnp.float32)

class HeadCapture:
 def __init__(self):self.h=self.raw=None
 def interceptor(self,next_fun,args,kwargs,ctx):
  m=ctx.module
  if ctx.method_name=='__call__' and m.__class__.__name__=='LayerNorm' and m.name=='decoder_norm' and 'lm_head' in m.scope.path:
   assert self.h is None,'Unexpected repeated final LN';self.raw=args[0];out=next_fun(*args,**kwargs);self.h=out;return out
  return next_fun(*args,**kwargs)
