"""Same-batch deterministic CE only at native comparison samples."""
from types import SimpleNamespace
from unittest import mock
import unittest

from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import train


class TrainingLossProbeTest(unittest.TestCase):
  def test_exact_registry_sampling_steps(self):
    sampled=[s for s in range(13500) if bool(train.no_dropout_loss_sample_step(s,200,25,10))]
    expected=[c+d for c in range(200,13500,200) for d in (-20,-10,0,10,20)]
    self.assertEqual(sampled,expected)
    self.assertEqual(len(sampled),335)
    self.assertFalse(bool(train.no_dropout_loss_sample_step(180,0,25,10)))

  def test_probe_keeps_update_uses_same_preupdate_batch_and_executes_conditionally(self):
    calls=[]
    class ToyModel(nn.Module):
      @nn.compact
      def __call__(self, inputs, positions, decoder_segment_ids, decoder_target_mask,
                   decoder_target_tokens, enable_dropout):
        weight=self.param('weight',nn.initializers.ones,())
        pred=nn.Dropout(rate=.5)(inputs*weight,deterministic=not enable_dropout)
        if not enable_dropout:
          jax.debug.callback(lambda _:calls.append('forward'),jnp.asarray(0))
        xent=(pred-decoder_target_tokens)**2
        return xent,jnp.sum(pred==decoder_target_tokens),pred
    model=ToyModel()
    cfg=SimpleNamespace(use_dpo=False,gradient_accumulation_steps=1,optimizer_memory_host_offload=False,
        enable_dropout=True,micro_batch_size_to_train_on=2,micro_batch_size_to_eval_on=1,
        mtp_num_layers=0,num_experts=1,debug=False,gradient_clipping_threshold=0.,
        record_training_health_metrics=False,record_internal_nn_metrics=False,
        train_no_dropout_loss_interval=200,train_no_dropout_loss_window=25,
        train_no_dropout_loss_sample_period=10)
    data=dict(inputs=jnp.arange(1,13,dtype=jnp.float32).reshape(4,3),
              inputs_position=jnp.zeros((4,3),jnp.int32),inputs_segmentation=jnp.ones((4,3),jnp.int32),
              targets_segmentation=jnp.ones((4,3),jnp.int32),targets=jnp.ones((4,3)))
    params=model.init(jax.random.key(1),data['inputs'],data['inputs_position'],
                      data['inputs_segmentation'],data['targets_segmentation'],data['targets'],False)
    state=train_state.TrainState.create(apply_fn=model.apply,params=params,tx=optax.sgd(.001))
    calls.clear()
    for step in (179,180,181,190,200,210,220,221):
      old=state.replace(step=jnp.asarray(step))
      key=jax.random.key(step)
      cfg.train_no_dropout_loss_interval=0
      baseline,_=train.train_step(model,cfg,None,old,dict(data),key)
      cfg.train_no_dropout_loss_interval=200
      # Build a fresh trace so the captured Python configuration remains fixed.
      f=jax.jit(lambda s,d,r:train.train_step(model,cfg,None,s,d,r))
      before=len(calls)
      new,metrics=f(old,dict(data),key)
      jax.block_until_ready((new,metrics));jax.effects_barrier()
      for a,b in zip(jax.tree.leaves(baseline),jax.tree.leaves(new)):
        np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
      valid=step in (180,190,200,210,220)
      self.assertEqual(bool(metrics['scalar']['learning/eval_loss_valid']),valid)
      self.assertEqual(len(calls)-before,int(valid))
      if valid:
        expected=float(jnp.mean((data['inputs'][:2]-1.)**2))
        self.assertAlmostEqual(float(metrics['scalar']['learning/eval_loss']),expected,places=5)
        # Prove this was not the post-update parameter or the eval-batch slice.
        after=float(jnp.mean((data['inputs'][:2]*new.params['params']['weight']-1.)**2))
        self.assertNotAlmostEqual(after,expected,places=3)

  def test_console_and_tb_do_not_publish_unmeasured_zero(self):
    cfg=SimpleNamespace(upload_loss_tb_period=10,upload_param_act_tb_period=25,
        log_period=1000,mtp_num_layers=0,num_experts=1,run_name='test',tensorboard_dir='')
    scalar={'learning/loss':3.2,'learning/total_weights':12,'learning/accuracy':.2,
        'learning/current_learning_rate':.001,'perf/step_time_seconds':.7,
        'perf/per_device_tflops_per_sec':1.,'learning/eval_loss':0.,'learning/eval_loss_valid':False}
    writer=mock.Mock()
    with mock.patch.object(train.max_logging,'log') as logger:
      train.write_metrics_to_tensorboard(writer,{'scalar':scalar},230,cfg)
      self.assertIn('loss: 3.200000',logger.call_args.args[0])
      self.assertIn('eval_loss: unmeasured',logger.call_args.args[0])
    tags=[call.args[0] for call in writer.add_scalar.call_args_list]
    self.assertNotIn('learning/eval_loss',tags)
    scalar.update({'learning/eval_loss':3.,'learning/eval_loss_valid':True})
    writer.reset_mock()
    with mock.patch.object(train.max_logging,'log') as logger:
      train.write_metrics_to_tensorboard(writer,{'scalar':scalar},220,cfg)
      self.assertIn('eval_loss: 3.000000',logger.call_args.args[0])
    self.assertIn(mock.call('learning/eval_loss',np.array(3.),220),writer.add_scalar.call_args_list)


if __name__=='__main__':unittest.main()
