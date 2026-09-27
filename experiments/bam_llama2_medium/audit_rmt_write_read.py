"""Exact full-size parameter/scan audit for write/read scheduling changes."""
import contextlib,io,math,tempfile
from pathlib import Path
import jax,jax.numpy as jnp
import max_utils,pyconfig
from layers import models
for name in ('RMTWriteReadControlProfile','RMTWriteReadMergedProfile','RMTWriteReadSplitC64Profile','RMTWriteReadMergedC2048Profile'):
 with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
  Path(directory,'audit').mkdir()
  cfg=pyconfig.initialize([None,str(Path(__file__).resolve().parents[2]/'MaxText/configs/base.yml')],
    exp_class=name,run_name='audit',enable_checkpointing=False,base_output_directory=directory+'/',
    jax_cache_dir='',log_config=False,dataset_type='synthetic',per_device_batch_size=1.)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
  model=models.Transformer(config=cfg,mesh=mesh,quant=None)
  args=dict(decoder_input_tokens=jnp.ones((1,4096),jnp.int32),decoder_positions=jnp.arange(4096)[None],
    decoder_target_tokens=jnp.ones((1,4096),jnp.int32),decoder_target_mask=jnp.ones((1,4096),jnp.float32),
    decoder_segment_ids=jnp.ones((1,4096),jnp.int32),enable_dropout=False)
  shape=jax.eval_shape(lambda seed:model.init(seed,**args)['params'],jax.random.key(0))
  count=sum(math.prod(x.shape) for x in jax.tree.leaves(shape))
 assert count==432119360,(name,count)
 print(name,count,flush=True)
