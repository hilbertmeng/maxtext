"""Six-layer forward probe for compounded BAM RMS precision changes."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import common_types
import max_utils
import pyconfig
from layers import models, normalizations


def relative_l2(actual, expected):
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)


def main():
  run_name = "rms-six-layer-forward"
  os.makedirs(run_name, exist_ok=True)
  cfg = pyconfig.initialize(
      [sys.argv[0], "MaxText/configs/base.yml"],
      exp_class="BamLlama2MediumDirectPLocR256Gelu",
      run_name=run_name,
      enable_checkpointing=False,
      per_device_batch_size=1.0,
      max_target_length=64,
      max_prefill_predict_length=4,
      base_emb_dim=512,
      base_num_query_heads=8,
      base_num_kv_heads=8,
      base_mlp_dim=1408,
      base_num_decoder_layers=6,
      scan_layers=False,
      attention="dot_product",
  )
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  model = models.Transformer(config=cfg, mesh=mesh, quant=None)
  rng, ids_key = jax.random.split(jax.random.PRNGKey(777))
  shape = (1, cfg.max_target_length)
  ids = jax.random.randint(ids_key, shape, 0, cfg.vocab_size, jnp.int32)
  positions = jnp.arange(cfg.max_target_length, dtype=jnp.int32)[None]
  segments = jnp.ones(shape, dtype=jnp.int32)
  variables = model.init(
      {"params": rng, "aqt": rng}, ids, positions, segments,
      enable_dropout=False, model_mode=common_types.MODEL_MODE_TRAIN)

  def forward():
    return jax.jit(lambda: model.apply(
        variables, ids, positions, segments, enable_dropout=False,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={"aqt": rng}))()

  new_rms_norm = normalizations.rms_norm
  new_logits = forward()

  def legacy_bam_rms(x, *, dtype, epsilon=1e-6, axis=-1):
    canonical_axis = axis if axis >= 0 else axis + x.ndim
    is_bam = (
        canonical_axis == x.ndim - 2
        or (x.ndim == 4 and canonical_axis == x.ndim - 1)
        or (x.ndim == 3 and x.shape[-1] == cfg.num_query_heads)
        or (canonical_axis == x.ndim - 1
            and (x.ndim >= 5 or (x.ndim == 3 and x.shape[-1] in (cfg.bam_k, cfg.bam_v))))
    )
    if not is_bam:
      return new_rms_norm(x, dtype=dtype, epsilon=epsilon, axis=axis)
    return jnp.asarray(
        x * jax.lax.rsqrt(
            jnp.mean(x ** 2, axis=axis, keepdims=True) + epsilon), dtype)

  normalizations.rms_norm = legacy_bam_rms
  jax.clear_caches()
  legacy_logits = forward()
  targets = jnp.roll(ids, -1, axis=-1)

  def loss(logits):
    token_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1),
        targets[..., None], axis=-1)
    return jnp.mean(token_loss)

  new_loss = loss(new_logits)
  legacy_loss = loss(legacy_logits)
  print("logits_relative_l2", relative_l2(new_logits, legacy_logits))
  print("logits_max_abs", float(jnp.max(jnp.abs(
      new_logits.astype(jnp.float32) - legacy_logits.astype(jnp.float32)))))
  print("new_loss", float(new_loss))
  print("legacy_loss", float(legacy_loss))
  print("loss_delta_new_minus_legacy", float(new_loss - legacy_loss))


if __name__ == "__main__":
  main()
