import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import unfreeze
import write_geometry as w
from layers.attentions import BamAttention
with tempfile.TemporaryDirectory() as out:
 Path(out,'capture').mkdir()
 cfg=w.pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=w.BASE,run_name='capture',enable_checkpointing=False,base_output_directory=out+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',base_emb_dim=128,base_num_query_heads=2,base_num_kv_heads=2,head_dim=64,max_target_length=8,max_prefill_predict_length=8,query_chunk_size=4,per_device_batch_size=1.)
 cfg.get_keys()['bam_write_v_bottleneck_dim']=32
 for k in list(cfg.get_keys()):
  if k.startswith('bam_record_'):cfg.get_keys()[k]=False
 mesh=jax.sharding.Mesh(w.max_utils.create_device_mesh(cfg),cfg.mesh_axes)
 m=BamAttention(config=cfg,num_query_heads=2,num_kv_heads=2,head_dim=64,bam_k=48,max_target_length=8,max_prefill_predict_length=8,mesh=mesh,attention_kernel='dot_product_chunk',dtype=cfg.dtype,layer_mode='local_qk+local_o',read_side='col',attention_type=cfg.attention_type)
 x=jax.random.normal(jax.random.key(171),(1,8,128),dtype=cfg.dtype);M=jax.random.normal(jax.random.key(172),(1,8,48,32),dtype=cfg.dtype)
 args=(x,x,jnp.arange(8)[None],jnp.ones((1,8),jnp.int32));kw=dict(M_in=M,deterministic=True,layer_index=1)
 p=unfreeze(m.init({'params':jax.random.key(173)},*args,**kw)['params']);leaf=p['W_R']['kernel'];p['W_R']['kernel']=leaf.replace(value=.1*jax.random.normal(jax.random.key(174),leaf.value.shape,leaf.value.dtype))
 baseline=m.apply({'params':p},*args,**kw)
 with w.capture():result,inter=m.apply({'params':p},*args,**kw,mutable=['intermediates'])
 for a,b in zip(baseline,result):np.testing.assert_array_equal(a,b)
 f=np.asarray(inter['intermediates']['geometry'][0]);assert f.shape==(1,8,2,len(w.FIELDS)),f.shape
 assert np.nanmax(f[...,w.FIELDS.index('reconstruction_relative')])<.02
 assert np.nanmax(f[...,w.FIELDS.index('key_content_relative_error')])<.02
 print('FULL_MODULE_CAPTURE_PASS',f.shape,'reconstruction max',np.nanmax(f[...,w.FIELDS.index('reconstruction_relative')]))
