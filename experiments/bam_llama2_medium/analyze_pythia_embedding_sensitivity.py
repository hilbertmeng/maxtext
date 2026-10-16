"""Actual-width first-layer synthetic probe; not a loss/eval or training-result predictor."""
import argparse,contextlib,hashlib,io,json,math
from pathlib import Path
from types import SimpleNamespace
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
import pyconfig
from layers import fusion,pythia,models,initializers
from pythia_test import PythiaTest

parser=argparse.ArgumentParser();parser.add_argument('--length',type=int,default=64);parser.add_argument('--skip-grad',action='store_true');parser.add_argument('--token-npz');parser.add_argument('--sequences',type=int,default=1);args=parser.parse_args();T=args.length
STD=math.sqrt(2/(5*1200));ALPHA=.006/STD
ARMS=[('mha','PythiaMediumPropT2048SerialMLP'),('mudd','MuddPythiaMediumPropT2048SerialMLP'),('bam','BamPythiaMediumPropT2048SerialMLPBamInitFix')]
STAT_VALID=None
def rms(x):
 x=jnp.asarray(x,jnp.float32)
 if STAT_VALID is not None and x.ndim>=3 and x.shape[:2]==STAT_VALID.shape:
  w=STAT_VALID.reshape(*STAT_VALID.shape,*([1]*(x.ndim-2)))
  return jnp.sqrt(jnp.sum(x*x*w)/(jnp.sum(STAT_VALID)*math.prod(x.shape[2:])))
 return jnp.sqrt(jnp.mean(x*x))
class First(nn.Module):
 cfg:object
 mesh:object
 @nn.compact
 def __call__(self,x):
  c=self.cfg;t=jnp.arange(x.shape[1],dtype=jnp.int32)[None];seg=jnp.ones_like(t) if TOKEN_SEG is None else TOKEN_SEG;pos=t if TOKEN_POS is None else TOKEN_POS;ids=t if TOKEN_IDS is None else TOKEN_IDS
  if c.bam_enabled:
   m=models.EmbeddingBamWrite(c,16,jnp.float32,jnp.float32,None,initializers.get_bam_init(c),name='seed')(x)
   y,_=fusion.SubDecoderLayer(c,self.mesh,layer_inx=0,sliding_window_size=c.max_target_length,name='layer')(x,seg,pos,ids,None,True,'train',None,M_in=m,is_global=True)
  elif c.dense_conn:
   y,_=fusion.SubDecoderLayer(c,self.mesh,layer_inx=0,sliding_window_size=c.max_target_length,name='layer')(x,seg,pos,ids,None,True,'train',None,is_global=True)
  else:
   y,_=pythia.PythiaDecoderLayer(c,self.mesh,sliding_window_size=c.max_target_length,name='layer')(x,seg,pos,ids,None,True,'train')
  return y

TOKEN_IDS=TOKEN_POS=TOKEN_SEG=None
helper=PythiaTest();helper.setUp();rows=[]
try:
 x0=jax.random.normal(jax.random.key(9876),(1,T,1200))
 if args.token_npz:
  with np.load(args.token_npz,allow_pickle=False) as batch:
   ik='inputs' if 'inputs' in batch.files else 'batch__inputs';pk='inputs_position' if 'inputs_position' in batch.files else 'batch__positions';sk='inputs_segmentation' if 'inputs_segmentation' in batch.files else 'batch__segments'
   ids=np.asarray(batch[ik])[:args.sequences,:T].astype(np.int32);assert ids.shape==(args.sequences,T)
   TOKEN_IDS=jnp.asarray(ids);TOKEN_POS=jnp.asarray(batch[pk][:args.sequences,:T]);TOKEN_SEG=jnp.asarray(batch[sk][:args.sequences,:T]);STAT_VALID=(TOKEN_SEG>0).astype(jnp.float32)
  unique,inverse=np.unique(ids,return_inverse=True);x0=jax.random.normal(jax.random.key(9876),(len(unique),1200))[inverse].reshape(args.sequences,T,1200)
 head=jax.random.normal(jax.random.key(600),(1200,256))*STD;targets=(jnp.arange(T)[None]+1)%256
 for label,arm in ARMS:
  with contextlib.redirect_stdout(io.StringIO()):
   c=helper.config(exp_class=arm,base_emb_dim=1200,base_num_decoder_layers=18,base_num_query_heads=16,base_num_kv_heads=16,head_dim=75,pythia_rotary_dim=18,vocab_size=256,max_target_length=T,max_prefill_predict_length=T,query_chunk_size=16,loss_chunk_size=64,scan_layers=False,bam_splash_attention=False)
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
    g=None if args.skip_grad else jax.jit(jax.value_and_grad(loss,argnums=1))(p,x,head)[1]
   # adam_pax count0, unit LR, WD omitted: mu=g,nu=g². Not the actual zero-LR training step0.
   unit=None if g is None else g/(jnp.sqrt(g*g)+1e-8)
   row={'model':label,'configuration':arm,'variant':variant,'embedding_std':std,'input_rms':float(rms(x)),'output_rms':float(rms(y)),'captured_rms':captured,'input_gradient_rms':None if g is None else float(rms(g)),'adam_unit_update_rms':None if unit is None else float(rms(unit)),'relative_adam_update_per_unit_lr':None if unit is None else float(rms(unit)/rms(x))}
   rows.append(row);print(json.dumps(row),flush=True)
 out={'scope':'One seed, actualD1200/L18 initialization and first-layer widths; Gaussian embeddings, optional saved token identities/segments/positions, synthetic256-way head; no training/validation loss claims. Input gradient only, explicit parameter JIT args. Adam count0 unit-LR proxy, excludes WD.','sequence_length':T,'sequences':args.sequences,'valid_tokens':None if STAT_VALID is None else int(jnp.sum(STAT_VALID)),'token_npz':args.token_npz,'token_ids_sha256':None if TOKEN_IDS is None else hashlib.sha256(np.asarray(TOKEN_IDS).tobytes()).hexdigest(),'unique_tokens':T if TOKEN_IDS is None else len(np.unique(np.asarray(TOKEN_IDS))),'gradient_probe':not args.skip_grad,'configuration_names':dict(ARMS),'parameter_seed':19,'input_seed':9876,'head_seed':600,'dtype':'float32','rows':rows}
 Path('/data0/xd/bam_diagnostics/pythia-init/embedding-sensitivity-first-layer'+(f'-PileT{T}' if args.token_npz else ('' if T==64 else f'-T{T}'))+'.json').write_text(json.dumps(out,indent=2)+'\n')
finally:helper.tearDown()
