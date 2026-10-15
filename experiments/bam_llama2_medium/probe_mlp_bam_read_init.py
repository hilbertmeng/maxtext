"""CPU initialization-only probe; full model widths, all eighteen actual BAM layers."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from bam_prop_test_utils import MLPWriteTest
from layers import models, fusion, linears, bam_mlp, initializers

def rms(x):
 return float(jnp.sqrt(jnp.mean(x.astype(jnp.float32)**2)))
h=MLPWriteTest();h.setUp();rows=[]
try:
 for seed in (19,29,39):
  cfg=h.config('BamMediumPropMShiftReadG050MLPReadAllLayersTruePile',bam_splash_attention=False)
  cfg.get_keys().update(max_target_length=16,max_prefill_predict_length=16)
  mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,)*len(cfg.mesh_axes)),cfg.mesh_axes)
  x=(.006*jax.random.normal(jax.random.key(seed),(2,16,1200))).astype(cfg.dtype)
  embed=models.EmbeddingBamWrite(cfg,16,cfg.dtype,cfg.weight_dtype,None,initializers.get_model_init(cfg))
  pos=jnp.broadcast_to(jnp.arange(16),(2,16));tok=jnp.ones((2,16),jnp.int32)
  with mesh,nn.partitioning.axis_rules([]):
   v=embed.init(jax.random.key(seed+100),x);matrix=embed.apply(v,x)
   for index in range(18):
    layer=fusion.SubDecoderLayer(cfg,mesh,layer_inx=index,sliding_window_size=16)
    args=(x,tok,pos,tok,None,True,'train',None)
    params=layer.init(jax.random.key(seed+200+index),*args,M_in=matrix,is_global=True)
    (x,matrix),aux=layer.apply(params,*args,M_in=matrix,is_global=True,
      capture_intermediates=lambda m,n: n=='__call__' and isinstance(m,(bam_mlp.BamMlpRead,linears.DenseGeneral)),mutable=['intermediates'])
    cap=aux['intermediates'];read=cap['mlp_bam_read']['__call__'][0]
    ordinary=cap['mlp']['wi_1']['__call__'][0];g=cap['mlp']['wi_0']['__call__'][0]
    stat=dict(seed=seed,layer=index,read_rms=rms(read),ordinary_W1_rms=rms(ordinary),
      read_over_ordinary=rms(read)/rms(ordinary),
      gated_read_rms=rms(read*jax.nn.silu(g[...,-600:])),
      gated_ordinary_rms=rms(ordinary*jax.nn.silu(g[...,:-600])),
      output_rms=rms(x),carried_M_rms=rms(matrix))
    assert all(np.isfinite(a) for a in stat.values()),stat
    rows.append(stat);print(json.dumps(stat),flush=True)
 out=Path('/data0/xd/bam_diagnostics/mediumprop-mlp-bam-read/init-amplitudes.json')
 out.write_text(json.dumps(dict(scope='All18 actual full-width layers; three seeds; bf16; 32 synthetic Gaussian token embedding rows per seed; initialization only, no trained-loss evidence',rows=rows),indent=2)+'\n')
finally:h.tearDown()
