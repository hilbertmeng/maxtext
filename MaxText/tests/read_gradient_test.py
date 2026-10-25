"""CPU: native gradient equals sum of zero-delta reader path gradients, including scan."""
import os,tempfile,re
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import exp,pyconfig,max_utils,precise_ce
from layers.models import Transformer
import read_gradient_capture as capture

def main():
 kind=os.environ.get('TEST_KIND','xl');pallas=kind=='pallas';is_py=kind in ['py','pallas'];n=5 if pallas else (8 if is_py else 7);B=1 if pallas else 2;T=128 if pallas else 16
 parent=exp.BamPallasPythiaR768TinyPallasV8TTest if pallas else (exp.BamPallasPythiaR768TinyXlaTest if is_py else exp.BamPallasFullMTinyXlaTest)
 attrs=dict(base_num_decoder_layers=n,bam_layer_modes=['local_qk+local_v+local_o']*n,vocab_size=128,dtype='float32',weight_dtype='float32',bam_pallas_core=pallas,bam_pallas_interpret=pallas,bam_pallas_body='v8t',bam_pallas_fused_inputs=pallas,bam_m_shift_mode='read',bam_m_shift_init=.5,max_target_length=T,max_prefill_predict_length=T,target_train_batch_size=B,per_device_batch_size=float(B),query_chunk_size=T)
 exp.ReadGradTiny=type('ReadGradTiny',(parent,),attrs);precise_ce.install()
 with tempfile.TemporaryDirectory() as out:
  os.mkdir(out+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='ReadGradTiny',run_name='test',base_output_directory=out+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);tok=jax.random.randint(jax.random.key(1),(B,T),0,128);pos=jnp.broadcast_to(jnp.arange(T),tok.shape);seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)};kw=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=model.init(rng,tok,pos,**kw)
  params=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,params,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
  def loss(weights):return jnp.mean(model.apply(weights,tok,pos,rngs=rng,**kw)[0])
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):native,native_grad=jax.jit(jax.value_and_grad(loss))(params)
  native_grad=jax.device_get(native_grad);capture.install();R=int(cfg.bam_local_full_m_read_bottleneck_dim);D=cfg.emb_dim;H=cfg.num_query_heads;V=cfg.bam_v
  zero=dict(up=jnp.zeros((n,2,R,H,1,V),jnp.float32),down=jnp.zeros((n,4,D,R),jnp.float32))
  def probe(delta):
   capture.DELTA=delta
   try:return loss(params)
   finally:capture.DELTA=None
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):value,grads=jax.jit(jax.value_and_grad(probe))(zero)
  np.testing.assert_allclose(value,native,atol=1e-7,rtol=1e-7)
  grads=jax.device_get(grads);checked=0;errors=[];max_native=0.
  for path,g in flatten_dict(native_grad).items():
   text='/'.join(path)
   which='up' if text.endswith('/self_attention/W_R_up/kernel') else 'down' if text.endswith('/self_attention/full_m_read_down/kernel') else None
   if which is None:continue
   layer=re.search(r'/layers_(\d+)/',text);tail=re.search(r'/final_local_layer(?:_(\d+))?/',text)
   if layer:pairs=[(int(layer[1]),g)]
   elif tail:
    tails=cfg.bam_final_local_layer_count
    if tails is None:tails=int(bool(cfg.bam_extra_final_local_layer))
    pairs=[(n-tails+int(tail[1] or 0),g)]
   else:
    role=re.search(r'/(local|fetch)_(\d+)/',text);assert role is not None
    pairs=[(i*cfg.bam_local_fetch_block_size+int(role[2]),np.take(g,i,axis=1)) for i in range(g.shape[1])]
   for l,reference in pairs:
    total=grads[which][l].sum(0);err=float(np.max(np.abs(total-reference)));scale=float(np.max(np.abs(reference)));max_native=max(max_native,scale);errors.append(err/(scale+1e-12));np.testing.assert_allclose(total,reference,atol=2e-7,rtol=2e-4,err_msg=f'{kind} {which} layer{l}');checked+=1
  assert checked==2*n,(checked,n)
  print('NATIVE_ZERO_DELTA_IDENTITY_AND_EXACT_PATH_GRADIENT_SUM_VERIFIED',kind,'layers',n,'matrices',checked,'max_relerr',max(errors),'max_native_gradient',max_native,flush=True)
if __name__=='__main__':main()
