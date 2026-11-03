"""Exercise real Transformer train_step with independent logit health and GA."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np
import optax

import max_utils
import pyconfig
from layers.models import Transformer
from layers.logits import HEALTH_METRIC_NAMES
from train import train_step


class LogitsTrainingTest(unittest.TestCase):
  def run_step(self, *, accumulation=1, health=True, precise=True):
    output=tempfile.TemporaryDirectory()
    self.addCleanup(output.cleanup)
    (Path(output.name)/'logits-test').mkdir()
    cfg = pyconfig.initialize(['test','MaxText/configs/base.yml'],
        run_name='logits-test', base_output_directory=output.name+'/',
        enable_checkpointing=False, log_config=False,
        decoder_block='pythia', init_method='pythia',
        qkv_bias=True, mlp_activations=['gelu_exact'],
        base_emb_dim=16, base_mlp_dim=64, base_num_decoder_layers=2,
        base_num_query_heads=2, base_num_kv_heads=2, head_dim=8, vocab_size=32,
        scan_layers=False, attention='dot_product', per_device_batch_size=2.,
        max_target_length=8, max_prefill_predict_length=8,
        gradient_accumulation_steps=accumulation,
        record_training_health_metrics=False, record_internal_nn_metrics=False,
        record_logits_health_metrics=health, logits_health_sample_size=16,
        logits_fp32_output_bf16_backward=precise)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    model=Transformer(cfg,mesh,quant=None)
    b=cfg.global_batch_size_to_train_on
    tokens=jnp.arange(b*8,dtype=jnp.int32).reshape(b,8)%32
    positions=jnp.broadcast_to(jnp.arange(8,dtype=jnp.int32),(b,8))
    mask=jnp.ones_like(tokens)
    data=dict(inputs=tokens,inputs_position=positions,inputs_segmentation=mask,
              targets=(tokens+1)%32,targets_segmentation=mask)
    with mesh,nn_rules(cfg):
      params=model.init({'params':jax.random.PRNGKey(1)},tokens,positions,
          decoder_target_tokens=data['targets'],decoder_target_mask=mask,
          decoder_segment_ids=mask,enable_dropout=False)
      state=TrainState.create(apply_fn=model.apply,params=params,tx=optax.sgd(.001))
      shardings=SimpleNamespace(params=jax.tree.map(lambda _:NamedSharding(mesh,PartitionSpec()),params))
      new_state,metrics=jax.jit(lambda state,data:train_step(
          model,cfg,shardings,state,data,jax.random.PRNGKey(2)))(state,data)
      jax.block_until_ready((new_state,metrics))
    self.assertEqual(int(new_state.step),1)
    self.assertTrue(np.isfinite(metrics['scalar']['learning/loss']))
    return metrics,new_state

  def test_real_train_step_exports_only_two_independent_health_metrics(self):
    on,state_on=self.run_step(health=True)
    off,state_off=self.run_step(health=False)
    for name in HEALTH_METRIC_NAMES:
      self.assertIn(name,on['scalar']);self.assertNotIn(name,off['scalar'])
      self.assertTrue(np.isfinite(on['scalar'][name]))
    self.assertNotIn('learning/raw_grad_norm',on['scalar'])
    np.testing.assert_array_equal(on['scalar']['learning/loss'],off['scalar']['learning/loss'])
    for a,b in zip(jax.tree.leaves(state_on.params),jax.tree.leaves(state_off.params)):
      np.testing.assert_array_equal(a,b)

  def test_real_accumulated_train_step_exports_pooled_health(self):
    metrics,_=self.run_step(accumulation=2)
    for name in HEALTH_METRIC_NAMES:
      self.assertIn(name,metrics['scalar']);self.assertTrue(np.isfinite(metrics['scalar'][name]))


def nn_rules(cfg):
  from flax import linen as nn
  return nn.partitioning.axis_rules(cfg.logical_axis_rules)


if __name__=='__main__':
  unittest.main()
