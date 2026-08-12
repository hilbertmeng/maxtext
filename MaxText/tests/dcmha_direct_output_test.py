"""CPU contracts for the minimal Plain-to-DCMHA posttrain path."""

from types import SimpleNamespace
import unittest

import flax
import jax
import jax.numpy as jnp

import exp
import max_utils
from layers import attentions, dc


def _config():
  return SimpleNamespace(
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      use_dw_bias=False,
      use_dd_bias=False,
      dc_use_muon=False,
      dc_share_all_dw_hidden=False,
      dc_share_prepost_dw_hidden=False,
      dc_w2_norm=False,
      normalization_layer_epsilon=1e-6,
      direct_scale=True,
      debug=False,
      query_chunk_size=None,
      query_chunk_method="",
      float32_qk_product=False,
      float32_logits=False,
      pre_compose=True,
      post_compose=True,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=2,
      dc_num_groups=None,
      static_proj=False,
      query_wise=True,
      key_wise=False,
      seperate_qk_dw_proj=True,
      dc_hidden_way="qk",
      attention_type="global",
      sw_quant=False,
      dc_gqa_global_heads=True,
  )


def _plain_gqa(query, key, value):
  repeats = query.shape[-2] // key.shape[-2]
  key = jnp.repeat(key, repeats, axis=-2)
  value = jnp.repeat(value, repeats, axis=-2)
  scores = jnp.einsum("btnd,bsnd->bnts", query, key)
  q_len, kv_len = query.shape[1], key.shape[1]
  mask = jnp.arange(kv_len)[None, None, None, :] <= jnp.arange(q_len)[None, None, :, None]
  probs = jax.nn.softmax(jnp.where(mask, scores, jnp.finfo(scores.dtype).min), axis=-1)
  return jnp.einsum("bnts,bsnd->btnd", probs, value)


class MinimalDcmhaPosttrainTest(unittest.TestCase):

  def test_attention_setup_accepts_global_and_local_window_values(self):
    config = _config()
    config.max_target_length = 128
    self.assertTrue(attentions.uses_dcmha_attention(config, None, "dot_product"))
    self.assertTrue(attentions.uses_dcmha_attention(config, 8, "dot_product"))
    self.assertFalse(attentions.uses_dcmha_attention(config, 128, "dot_product"))
    self.assertTrue(attentions.uses_dcmha_attention(config, 128, "dot_product_chunk"))

  def test_step0_metrics_and_all_new_parameter_gradients(self):
    config = _config()
    module = dc.AttentionOp(config=config, num_kv_heads=2)
    query = jax.random.normal(jax.random.key(1), (1, 6, 4, 2)) / jnp.sqrt(2.0)
    key = jax.random.normal(jax.random.key(2), (1, 6, 2, 2))
    value = jax.random.normal(jax.random.key(3), (1, 6, 2, 2))
    controller = jax.random.normal(jax.random.key(4), (1, 6, 8))
    segments = jnp.ones((1, 6), dtype=jnp.int32)
    variables = module.init(
        jax.random.key(5), query, key, value, segments, input_q=controller, input_kv=controller
    )
    output = module.apply(
        variables, query, key, value, segments, input_q=controller, input_kv=controller
    )
    reference = _plain_gqa(query, key, value)

    params = {
        path: value.value if hasattr(value, "value") else value
        for path, value in flax.traverse_util.flatten_dict(variables["params"]).items()
    }
    self.assertTrue(any(path[-1] == "qkw" for path in params))
    self.assertTrue(any(path[-2:] == ("dd", "kernel") for path in params))
    self.assertFalse(any("qkw1" in path or "qkw2" in path for path in params))
    self.assertTrue(all(bool(jnp.any(value != 0)) for value in params.values()))

    projection = jax.random.normal(jax.random.key(8), (8, 13)) / jnp.sqrt(8.0)
    labels = jax.random.randint(jax.random.key(9), (1, 6), 0, 13)

    def loss(attention_output):
      logits = attention_output.reshape(1, 6, 8) @ projection
      nll = -jax.nn.log_softmax(logits)[jnp.arange(1)[:, None], jnp.arange(6), labels]
      return logits, jnp.mean(nll)

    plain_logits, plain_loss = loss(reference)
    dc_logits, dc_loss = loss(output)
    delta = jnp.abs(dc_logits - plain_logits)
    relative_loss_delta = jnp.abs(dc_loss - plain_loss) / jnp.maximum(jnp.abs(plain_loss), 1e-8)
    top1_agreement = jnp.mean(jnp.argmax(dc_logits, axis=-1) == jnp.argmax(plain_logits, axis=-1))
    print(
        "step0_metrics",
        f"relative_loss_delta={float(relative_loss_delta):.8f}",
        f"logits_mean_abs_delta={float(jnp.mean(delta)):.8f}",
        f"logits_max_abs_delta={float(jnp.max(delta)):.8f}",
        f"token_top1_agreement={float(top1_agreement):.8f}",
    )
    self.assertLess(float(relative_loss_delta), 0.05)

    def objective(p):
      direct = module.apply(
          {"params": p}, query, key, value, segments, input_q=controller, input_kv=controller
      )
      return loss(direct)[1]

    grads = {
        path: grad.value if hasattr(grad, "value") else grad
        for path, grad in flax.traverse_util.flatten_dict(jax.grad(objective)(variables["params"])).items()
    }
    self.assertEqual(set(grads), set(params))
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) and bool(jnp.any(g != 0)) for g in grads.values()))

  def test_posttrain_identity_is_plain_initialized(self):
    identity = exp.Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap303e4RerunQueryWiseDCMHAPostTrainV1
    self.assertTrue(identity.train_merge_loaded_params)
    self.assertTrue(identity.dc_gqa_global_heads)
    self.assertEqual(identity.steps, 1001)
    self.assertEqual(identity.eval_interval, 1000)
    self.assertEqual(
        identity.train_load_parameters_path,
        "gs://newproject-1-llm_projects_us-east5/log/qwen3_alignment/"
        "maxtext_qwen3_0_6b_nvarc16_ckpt/0/items",
    )
    for removed in ("dc_identity_preserved", "dc_dw2_zero_init", "dc_implementation"):
      self.assertFalse(hasattr(identity, removed), removed)

  def test_plain_restore_skips_only_new_dcmha_leaves(self):
    params = {
        "decoder": {
            "plain": {"kernel": jnp.zeros((2, 2))},
            "q_dyn_w_proj": {"qkw": jnp.zeros((2, 2)), "dd": {"kernel": jnp.zeros((2, 2))}},
        }
    }
    skipped = max_utils.dcmha_param_paths_to_skip(params)
    self.assertEqual(len(skipped), 2)
    self.assertTrue(all("q_dyn_w_proj" in path for path in skipped))


if __name__ == "__main__":
  unittest.main()
