"""Audit constant token budgets, exact parent parameter trees and affected histories."""
import contextlib, io, json, math, runpy, tempfile
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
import pyconfig, max_utils, train_compile
from layers.models import Transformer

defs=runpy.run_path('MaxText/exp.py')
rows=[('BamMHAMediumPropC256T2048Matched','BamMHAMediumPropC256TruePile'),
      ('MuddLlama2MediumPropC256T2048','MuddLlama2MediumPropTruePile'),
      ('BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadT2048',
       'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile')]
allowed={'model_name','DATASET_VARIANT','dataset_path','max_target_length',
         'per_device_batch_size','jax_cache_dir','compare_runs','bam_splash_attention'}
audit=[]
for name,parent in rows:
 a,b=defs[name],defs[parent]
 changed={k for k in dir(a) if not k.startswith('__') and getattr(a,k)!=getattr(b,k,None)}
 assert changed<=allowed,(name,changed-allowed)
 if 'FullMRead' not in name:assert a.bam_splash_attention is False
 assert a.max_target_length*a.per_device_batch_size == b.max_target_length*b.per_device_batch_size
 assert a.DATASET_VARIANT=='legacy2048' and a.steps==b.steps==13500
 counts=[]
 for klass in (name,parent):
  with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
   Path(tmp,'audit').mkdir()
   cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=klass,
      run_name='audit',enable_checkpointing=False,base_output_directory=tmp+'/',
      dataset_type='synthetic',jax_cache_dir='',log_config=False,bam_splash_attention=False)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
   args,_,_,_=train_compile.get_shaped_inputs(mesh,cfg)
   counts.append(sum(math.prod(v.shape) for v in jax.tree.leaves(nn.unbox(args[0].params))))
   assert cfg.global_batch_size_to_train_on*cfg.max_target_length==524288
 assert counts[0]==counts[1],(name,counts)
 audit.append(dict(run=name,parent=parent,params=counts[0],global_batch=256,targets_per_step=524288))
 print('CONTEXT_CONFIG_PARAM_OK',name,counts[0],flush=True)

# Short CPU sequences explicitly disable Splash. Cover the newly imported Mudd
# histories and the shared GELU/generalized-read path with finite gradients.
for name,_ in rows[1:]:
 with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
  Path(tmp,'tiny').mkdir()
  changes=dict(base_num_decoder_layers=3,base_emb_dim=150,base_num_query_heads=2,
      base_num_kv_heads=2,base_mlp_dim=256,vocab_size=32)
  cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=name,
      run_name='tiny',enable_checkpointing=False,base_output_directory=tmp+'/',
      dataset_type='synthetic',jax_cache_dir='',log_config=False,max_target_length=4,
      max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,
      bam_splash_attention=False,**changes)
  if 'FullMRead' in name:
   cfg.get_keys().update(bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,
       bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,
       emb_bam_num_head=2,bam_mlp_write_num_heads=0,mlp_dim_by_block=[16,12,16],
       bam_layer_modes=cfg.bam_layer_modes[:3])
  mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)),cfg.mesh_axes)
  model=Transformer(cfg,mesh,quant=None)
  tok=jnp.array([[1,2,3,4]],jnp.int32)
  args=dict(decoder_input_tokens=tok,decoder_positions=jnp.arange(4)[None],
      decoder_target_tokens=tok,decoder_target_mask=jnp.ones_like(tok,dtype=jnp.float32),
      decoder_segment_ids=jnp.ones_like(tok),enable_dropout=False)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
   params=nn.unbox(model.init(jax.random.key(9876),**args)['params'])
   def loss(p):
    out=model.apply({'params':p},**args)
    return jnp.mean((out[0] if isinstance(out,tuple) else out).astype(jnp.float32)**2)
   value,grad=jax.jit(jax.value_and_grad(loss))(params)
   assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad))
   if cfg.dense_conn:
    from tests.mudd_xlprop_test import MuddXLPropTest
    MuddXLPropTest().check_history(params,3)
   else:
    assert cfg.bam_general_column_read and not cfg.bam_static_read_gate_sigmoid
    assert cfg.bam_static_read_gate_init==.99
 print('CONTEXT_FORWARD_GRAD_OK',name,flush=True)
Path('/data0/xd/bam_diagnostics/mediumprop-context-t2048/config-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
