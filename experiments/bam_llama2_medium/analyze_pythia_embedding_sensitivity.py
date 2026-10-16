"""Actual-width first-layer synthetic probe; not a loss/eval or training-result predictor."""
import contextlib,io,json,math
from pathlib import Path
from types import SimpleNamespace
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
import pyconfig
from layers import fusion,pythia,models,initializers
from pythia_test import PythiaTest

STD=math.sqrt(2/(5*1200));ALPHA=.006/STD
ARMS=[('mha','PythiaMediumPropT2048SerialMLP'),('mudd','MuddPythiaMediumPropT2048SerialMLP'),('bam','BamPythiaMediumPropT2048SerialMLPBamInitFix')]
def rms(x):return jnp.sqrt(jnp.mean(jnp.asarray(x,jnp.float32)**2))
class First(nn.Module):
 cfg:object
 mesh:object
 @nn.compact
 def __call__(self,x):
  c=self.cfg;t=jnp.arange(x.shape[1],dtype=jnp.int32)[None];seg=jnp.ones_like(t)
  if c.bam_enabled:
   m=models.EmbeddingBamWrite(c,16,jnp.float32,jnp.float32,None,initializers.get_bam_init(c),name='seed')(x)
   y,_=fusion.SubDecoderLayer(c,self.mesh,layer_inx=0,sliding_window_size=64,name='layer')(x,seg,t,t,None,True,'train',None,M_in=m,is_global=True)
  elif c.dense_conn:
   y,_=fusion.SubDecoderLayer(c,self.mesh,layer_inx=0,sliding_window_size=64,name='layer')(x,seg,t,t,None,True,'train',None,is_global=True)
  else:
   y,_=pythia.PythiaDecoderLayer(c,self.mesh,sliding_window_size=64,name='layer')(x,seg,t,t,None,True,'train')
  return y

helper=PythiaTest();helper.setUp();rows=[]
try:
 x0=jax.random.normal(jax.random.key(9876),(1,64,1200));head=jax.random.normal(jax.random.key(600),(1200,256))*STD;targets=(jnp.arange(64)[None]+1)%256
 for label,arm in ARMS:
  with contextlib.redirect_stdout(io.StringIO()):
   c=helper.config(exp_class=arm,base_emb_dim=1200,base_num_decoder_layers=18,base_num_query_heads=16,base_num_kv_heads=16,head_dim=75,pythia_rotary_dim=18,vocab_size=256,max_target_length=64,max_prefill_predict_length=64,query_chunk_size=16,loss_chunk_size=64,scan_layers=False,bam_splash_attention=False)
  mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(c.mesh_axes)),c.mesh_axes)
  base=First(c,mesh)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),contextlib.redirect_stdout(io.StringIO()):p=base.init(jax.random.key(19),x0*.006)['params']
  for variant,std,scale in [('small',.006,1.),('large_compensated' if label=='bam' else 'large',STD,ALPHA if label=='bam' else 1.)]:
   cfg=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_embedding_write_input_scale=scale)));model=First(cfg,mesh);x=x0*std;captured={}
   def capture(next_fun,a,k,ctx):
    y=next_fun(*a,**k)
    if ctx.method_name=='__call__' and ctx.module.name in ('input_layernorm','pre_self_attention_layer_norm','post_attention_layernorm','post_self_attention_layer_norm','self_attention','mlp','seed'):
     v=y[0] if isinstance(y,tuple) else y
     captured[ctx.module.name]=float(rms(v))
    return y
   def loss(params,z,w):
    h=model.apply({'params':params},z);centered=h-jnp.mean(h,axis=-1,keepdims=True);n=centered*jax.lax.rsqrt(jnp.mean(centered**2,axis=-1,keepdims=True)+1e-5);logits=n@w
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits),targets[...,None],axis=-1))
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),contextlib.redirect_stdout(io.StringIO()):
    with nn.intercept_methods(capture):y=model.apply({'params':p},x)
    value,g=jax.jit(jax.value_and_grad(loss,argnums=1))(p,x,head)
   # adam_pax count0, unit LR, WD omitted: mu=g,nu=g². Not the actual zero-LR training step0.
   unit=g/(jnp.sqrt(g*g)+1e-8)
   row={'model':label,'configuration':arm,'variant':variant,'embedding_std':std,'input_rms':float(rms(x)),'output_rms':float(rms(y)),'captured_rms':captured,'input_gradient_rms':float(rms(g)),'adam_unit_update_rms':float(rms(unit)),'relative_adam_update_per_unit_lr':float(rms(unit)/rms(x))}
   rows.append(row);print(json.dumps(row),flush=True)
 out={'scope':'One seed, actualD1200/L18 initialization and first-layer widths,64 Gaussian unique token vectors,synthetic256-way head; no training/validation loss claims. Input gradient only, explicit parameter JIT args. Adam count0 unit-LR proxy, excludes WD.','configuration_names':dict(ARMS),'parameter_seed':19,'input_seed':9876,'head_seed':600,'dtype':'float32','rows':rows}
 Path('/data0/xd/bam_diagnostics/pythia-init/embedding-sensitivity-first-layer.json').write_text(json.dumps(out,indent=2)+'\n')
finally:helper.tearDown()
