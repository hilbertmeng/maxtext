"""Parameter and differentiability gates for the uniform AllLocal TruePile run."""
import tempfile
from pathlib import Path
import functools
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig, max_utils, train, train_compile
from layers.models import Transformer
from layers import quantizations
EXP='BamXLPropK96EmbedVOnlyQK72AllLocalTruePile'
with tempfile.TemporaryDirectory() as out:
 Path(out,'audit').mkdir()
 c=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=EXP,run_name='audit',enable_checkpointing=False,base_output_directory=out+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.)
 assert c.scan_layers and not c.bam_pair_scan
 assert c.mlp_dim==6294 and c.mlp_dim_by_block is None
 mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
 args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
 leaves=flatten_dict(args[0].params)
 count=sum(int(np.prod(x.shape)) for x in leaves.values())
 assert count==1432432740,count
 assert not any('/value/' in '/'.join(k) or 'fetch_head_mix' in '/'.join(k) for k in leaves)
 with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
  metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
 assert 'learning/raw_grad_norm' in metrics['scalar']
 print('FULL_PARAM_AND_TRAIN_TRACE_OK',count,flush=True)
 # Preserve K96/V40/C10/head96/QK72+24; shrink only breadth/depth for execution.
 c.get_keys().update(base_emb_dim=192,emb_dim=192,num_query_heads=2,num_kv_heads=2,
     base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=2,num_decoder_layers=2,
     base_mlp_dim=128,mlp_dim=128,vocab_size=128,bam_layer_modes=['local_qk+local_v+local_o']*2,
     bam_write_v_bottleneck_dim=32,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=32)
 model=Transformer(c,mesh,quantizations.configure_quantization(c))
 tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens)
 call=(tokens,pos,tokens,mask,mask)
 with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
  params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
  def loss(p):
   result=model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})
   return jnp.mean(result[0])
  value,grad=jax.jit(jax.value_and_grad(loss))(params)
 assert np.isfinite(float(value))
 assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad))
 assert sum(float(jnp.sum(g.astype(jnp.float32)**2)) for g in jax.tree.leaves(grad))>0
 print('SCANNED_FORWARD_GRAD_OK',float(value),flush=True)
