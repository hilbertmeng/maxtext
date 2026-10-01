import os,tempfile,unittest,contextlib,io
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
import pyconfig
from layers import linears,models
class Probe(nn.Module):
 config:object
 @nn.compact
 def __call__(self,x,target,mask):
  cfg=self.config
  y=linears.MlpBlock(config=cfg,intermediate_dim=64,activations=cfg.mlp_activations,
      intermediate_dropout_rate=0,dtype=jnp.float32,weight_dtype=jnp.float32,name='mlp')(
      x,deterministic=True,probe_layer_index=jnp.asarray(27) if cfg.get_keys().get('rmt_crossscale_numeric_probe',False) else None)
  return models.OutputHead(config=cfg,shared_embedding=None,mesh=None,name='head')(y,target,mask,8,True)[0].mean()
class BamCheckpointHealthTest(unittest.TestCase):
 def test_forward_gradient_identity(self):
  cfg=pyconfig.initialize(['probe','MaxText/configs/base.yml','exp_class=BamXLPropK96EmbedVOnlyQK72AllLocalTruePile',
       'run_name=bam-health-cpu','base_output_directory=/tmp/bam-health-cpu/','enable_checkpointing=False',
       'vocab_size=128','per_device_batch_size=1','max_target_length=8','max_prefill_predict_length=8','log_config=False'])
  cfg.get_keys()['dtype']=jnp.float32
  with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
   os.environ['RMT_HEALTH_FILE']=d+'/health.jsonl';os.environ['RMT_NORM_TAP_FILE']=d+'/taps.jsonl'
   model=Probe(cfg);x=jax.random.normal(jax.random.key(1),(1,8,1920));target=jnp.arange(8)[None];mask=jnp.ones((1,8));params=nn.unbox(model.init(jax.random.key(2),x,target,mask))
   objective=lambda p:model.apply(p,x,target,mask)
   a,g=jax.jit(jax.value_and_grad(objective))(params)
   cfg.get_keys().update(rmt_crossscale_numeric_probe=True,rmt_crossscale_health_probe=True)
   b,dg=jax.jit(jax.value_and_grad(objective))(params);jax.effects_barrier()
   self.assertTrue(Path(os.environ['RMT_HEALTH_FILE']).stat().st_size>0)
  np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-6)
  for ga,gb in zip(jax.tree.leaves(g),jax.tree.leaves(dg)):np.testing.assert_allclose(ga,gb,atol=2e-6,rtol=2e-5)
if __name__=='__main__':unittest.main()
