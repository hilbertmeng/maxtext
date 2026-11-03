"""Output-head precision, chunk/GA statistics and configuration regressions."""
import unittest
from types import SimpleNamespace

from flax import linen as nn
from flax.core import unfreeze
import jax
import jax.numpy as jnp
import numpy as np
import yaml

import max_utils
from layers import embeddings
from layers import logits as logits_utils
from layers.models import OutputHead
from train import record_logits_health_metrics


def config(**overrides):
  values = dict(dtype=jnp.bfloat16, weight_dtype=jnp.float32, emb_dim=4, vocab_size=8,
                init_method='006normal', dropout_rate=0., rmt_enabled=True,
                logits_via_embedding=False, normalize_embedding_logits=False,
                final_logits_soft_cap=None, cast_logits_to_fp32=True,
                logits_dot_in_fp32=False, logits_fp32_output_bf16_backward=False,
                record_logits_health_metrics=False, logits_health_sample_size=128)
  values.update(overrides)
  return SimpleNamespace(**values)


class TiedHead(nn.Module):
  config: object

  @nn.compact
  def __call__(self, hidden, targets, mask, chunk_size=1024):
    emb = embeddings.Embed(self.config, self.config.vocab_size, self.config.emb_dim,
                           dtype=self.config.dtype, name='embedding')
    return OutputHead(self.config, emb, None, name='lm_head')(
        hidden, targets, mask, chunk_size=chunk_size)


class LogitsTest(unittest.TestCase):
  def setUp(self):
    self.h = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 4)).astype(jnp.bfloat16)
    self.y = jnp.arange(14).reshape(2, 7) % 8
    self.mask = jnp.ones_like(self.y)

  def head(self, **kwargs):
    cfg = config(**kwargs)
    return OutputHead(cfg, None, None), cfg

  def test_final_upcast_preserves_bf16_values_and_fixes_both_logit_apis(self):
    h = jnp.array([[[1., 1., 1., 1.]]], jnp.bfloat16)
    native, _ = self.head(cast_logits_to_fp32=False)
    variables = native.init(jax.random.PRNGKey(2), h, method=native.project_logits)
    restored, _ = self.head(cast_logits_to_fp32=True)
    low = native.apply(variables, h, method=native.project_logits)
    high = restored.apply(variables, h, method=restored.project_logits)
    self.assertEqual(low.dtype, jnp.bfloat16)
    self.assertEqual(high.dtype, jnp.float32)
    np.testing.assert_array_equal(high, low.astype(jnp.float32))
    direct = restored.apply(variables, h, method=restored.logits_from_hidden_states)
    np.testing.assert_array_equal(direct, high)

  def test_forward_retains_information_lost_by_posthoc_cast(self):
    # 1 + 1/256 is halfway between consecutive BF16 numbers near 1.
    h = jnp.array([[[1., 1/256, 0., 0.]]], jnp.bfloat16)
    old, _ = self.head()
    variables = old.init(jax.random.PRNGKey(2), h, method=old.project_logits)
    variables = unfreeze(nn.unbox(variables))
    variables['params']['logits_dense'] = jnp.ones((4, 8), jnp.float32)
    rounded = old.apply(variables, h, method=old.project_logits)
    for cast in (False, True):
      precise, _ = self.head(cast_logits_to_fp32=cast, logits_fp32_output_bf16_backward=True)
      actual = precise.apply(variables, h, method=precise.project_logits)
      self.assertEqual(actual.dtype, jnp.float32)
      np.testing.assert_array_equal(actual, jnp.full((1, 1, 8), 1+1/256, jnp.float32))
      self.assertTrue(np.any(np.asarray(actual) != np.asarray(rounded)))

  def test_custom_backward_equals_native_for_same_cotangent(self):
    w = jax.random.normal(jax.random.PRNGKey(3), (4, 8)).astype(jnp.bfloat16)
    g = jax.random.normal(jax.random.PRNGKey(4), (2, 7, 8)).astype(jnp.bfloat16)
    _, native_pull = jax.vjp(lambda h,w:jnp.tensordot(h,w,axes=((-1,), (0,))), self.h, w)
    output, pull = jax.vjp(logits_utils.bf16_dot_fp32_output, self.h, w)
    self.assertEqual(output.dtype, jnp.float32)
    for actual, expected in zip(pull(g.astype(jnp.float32)), native_pull(g)):
      self.assertEqual(actual.dtype, jnp.bfloat16)
      np.testing.assert_array_equal(actual, expected)
    with self.assertRaisesRegex(TypeError, 'requires BF16'):
      logits_utils.bf16_dot_fp32_output(self.h.astype(jnp.float32), w)

  def test_restored_fp32_ce_preserves_confident_target_gradient(self):
    h=jnp.array([[[1.,0.,0.,0.]]],jnp.bfloat16)
    y=jnp.zeros((1,1),jnp.int32);mask=jnp.ones_like(y)
    native,_=self.head(cast_logits_to_fp32=False)
    variables=unfreeze(nn.unbox(native.init(jax.random.PRNGKey(8),h,y,mask)))
    variables['params']['logits_dense']=jnp.zeros((4,8),jnp.float32).at[0].set(
        jnp.array([0.,-8.,-100.,-100.,-100.,-100.,-100.,-100.]))
    def objective(params,model):
      return model.apply({'params':params},h,y,mask)[0].mean()
    old=jax.grad(lambda p:objective(p,native))(variables['params'])
    restored,_=self.head()
    fixed=jax.grad(lambda p:objective(p,restored))(variables['params'])
    self.assertEqual(float(old['logits_dense'][0,0]),0.)
    self.assertLess(float(fixed['logits_dense'][0,0]),-.0003)

  def test_chunked_training_ce_matches_manual_reference_and_gradients(self):
    for tied in (False, True):
      for precise in (False, True):
        cfg = config(logits_via_embedding=tied, logits_fp32_output_bf16_backward=precise)
        head = TiedHead(cfg) if tied else OutputHead(cfg, None, None)
        variables = head.init(jax.random.PRNGKey(5), self.h, self.y, self.mask, chunk_size=3)
        def evaluate(params, chunk):
          ce,_,_ = head.apply({'params':params}, self.h, self.y, self.mask, chunk_size=chunk)
          return ce.mean()
        a, ga = jax.value_and_grad(lambda p:evaluate(p, 3))(variables['params'])
        b, gb = jax.value_and_grad(lambda p:evaluate(p, 7))(variables['params'])
        np.testing.assert_allclose(a, b, atol=2e-6)
        for x,y in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
          # Chunked BF16 dW has a different reduction order.
          np.testing.assert_allclose(x, y, atol=6e-4, rtol=.02)
        if not tied:
          logits = head.apply(variables, self.h, method=head.project_logits)
          self.assertEqual(logits.dtype, jnp.float32)
          def reference(h):
            z = head.apply(variables, h, method=head.project_logits)
            return max_utils.cross_entropy_with_logits(z, jax.nn.one_hot(self.y,8), 0.)[0].mean()
          ref,_ = jax.value_and_grad(reference)(self.h)
          np.testing.assert_allclose(a, ref, atol=2e-6)

  def test_health_is_global_across_chunks_masked_and_gradient_neutral(self):
    head, _ = self.head(record_logits_health_metrics=True, logits_health_sample_size=14)
    variables = head.init(jax.random.PRNGKey(6), self.h, self.y, self.mask, chunk_size=3)
    self.assertNotIn('intermediates', variables)
    mask = self.mask.at[0,:2].set(0)
    outputs, intermediate = head.apply(variables, self.h, self.y, mask, chunk_size=3,
                                       mutable=['intermediates'])
    samples = intermediate['intermediates'][logits_utils.HEALTH_SAMPLE_KEY][0]
    self.assertEqual(samples.shape, (14,3))
    expected_logits = np.asarray(head.apply(variables, self.h, method=head.project_logits))
    valid = np.asarray(mask) > 0
    means = expected_logits.mean(-1)[valid]
    stds = expected_logits.std(-1)[valid]
    expected = [np.quantile(np.abs(means),.95), np.median(stds)]
    np.testing.assert_allclose(logits_utils.summarize_health(samples), expected, atol=2e-6)
    _, full = head.apply(variables, self.h, self.y, mask, chunk_size=7, mutable=['intermediates'])
    full_samples = full['intermediates'][logits_utils.HEALTH_SAMPLE_KEY][0]
    np.testing.assert_allclose(logits_utils.summarize_health(full_samples), expected, atol=2e-6)
    plain, _ = self.head()
    def objective(model, params):
      (ce,_,_),extra = model.apply({'params':params}, self.h, self.y, mask,
                                   chunk_size=3, mutable=['intermediates'])
      return ce.mean(), extra
    (_, off), goff = jax.value_and_grad(lambda p:objective(plain,p),has_aux=True)(variables['params'])
    (_, on), gon = jax.value_and_grad(lambda p:objective(head,p),has_aux=True)(variables['params'])
    self.assertNotIn(logits_utils.HEALTH_SAMPLE_KEY, off.get('intermediates',{}))
    for x,y in zip(jax.tree.leaves(goff),jax.tree.leaves(gon)):
      np.testing.assert_array_equal(x,y)

  def test_large_signed_offsets_and_empty_masks(self):
    z = jnp.array([[[200.,202.,198.,200.],[-200.,-198.,-202.,-200.]]],jnp.float32)
    samples = logits_utils.chunk_health_samples(z,jnp.ones((1,2)),0,2,128)
    np.testing.assert_allclose(logits_utils.summarize_health(samples),[200.,np.sqrt(2)],atol=1e-6)
    empty = samples.at[:,2].set(0)
    self.assertTrue(np.isnan(np.asarray(logits_utils.summarize_health(empty))).all())

  def test_sampling_does_not_alias_to_sequence_start_on_real_batch_shape(self):
    positions = logits_utils.health_sample_positions(1024,2048,128)
    self.assertEqual(len(positions),128)
    self.assertGreater(len({p % 2048 for p in positions}),64)
    for i,p in enumerate(positions):
      self.assertGreaterEqual(p,i*1024*2048//128)
      self.assertLess(p,(i+1)*1024*2048//128)

  def test_gradient_accumulation_pools_samples_instead_of_averaging_quantiles(self):
    samples = jnp.array([[[1.,1.,1.],[2.,2.,1.]], [[100.,3.,1.],[200.,4.,0.]]])
    aux = {'total_loss':jnp.array([2.,3.]), 'intermediate_outputs':{'intermediates':{
        'decoder':{'lm_head':{logits_utils.HEALTH_SAMPLE_KEY:(samples,)}}}}}
    merged = logits_utils.reduce_microbatch_aux(aux)
    self.assertEqual(float(merged['total_loss']),5.)
    metrics = {'scalar':{}}
    record_logits_health_metrics(metrics,merged['intermediate_outputs'])
    self.assertEqual(set(metrics['scalar']),set(logits_utils.HEALTH_METRIC_NAMES))
    actual = [metrics['scalar'][k] for k in logits_utils.HEALTH_METRIC_NAMES]
    np.testing.assert_allclose(actual,[np.quantile([1.,2.,100.],.95),2.],rtol=1e-6)
    evaluation = {'scalar':{}}
    record_logits_health_metrics(evaluation,merged['intermediate_outputs'],prefix='evaluation/')
    self.assertEqual(set(evaluation['scalar']),{'evaluation/'+k for k in logits_utils.HEALTH_METRIC_NAMES})

  def test_health_excludes_mtp_and_preserves_parameter_tree(self):
    head, _ = self.head(record_logits_health_metrics=True)
    variables = head.init(jax.random.PRNGKey(7), self.h, self.y, self.mask)
    _, extra = head.apply(variables,self.h,self.y,self.mask,mtp_layer=True,mutable=['intermediates'])
    self.assertNotIn(logits_utils.HEALTH_SAMPLE_KEY,extra.get('intermediates',{}))
    precise, _ = self.head(logits_fp32_output_bf16_backward=True)
    precise_vars = precise.init(jax.random.PRNGKey(7),self.h,self.y,self.mask)
    self.assertEqual(jax.tree.structure(variables),jax.tree.structure(precise_vars))
    for a,b in zip(jax.tree.leaves(variables),jax.tree.leaves(precise_vars)):
      np.testing.assert_array_equal(a,b)

  def test_config_defaults_and_validation(self):
    with open('MaxText/configs/base.yml') as f:
      keys = yaml.safe_load(f)
    self.assertTrue(keys['cast_logits_to_fp32'])
    self.assertFalse(keys['logits_fp32_output_bf16_backward'])
    self.assertFalse(keys['record_logits_health_metrics'])
    logits_utils.validate_config(keys)
    for overrides in [dict(logits_health_sample_size=0),dict(logits_health_sample_size=True),
                      dict(record_logits_health_metrics=1),
                      dict(logits_fp32_output_bf16_backward=True,dtype='float32'),
                      dict(logits_fp32_output_bf16_backward=True,logits_dot_in_fp32=True)]:
      with self.subTest(overrides=overrides),self.assertRaises(ValueError):
        logits_utils.validate_config(dict(keys,**overrides))


if __name__ == '__main__':
  unittest.main()
