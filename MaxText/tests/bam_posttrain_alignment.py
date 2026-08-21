"""CPU contracts for the accepted Plain -> BAM posttrain experiment identities."""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np


PLAIN = "Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileTiedCap303e4Rerun"
BAM = f"{PLAIN}BAM"
ADAPTATION = f"{PLAIN}BAMAdaptation"
ADAPTATION_V2 = f"{PLAIN}BAMAdaptationV2"
POSTNORM = f"{ADAPTATION}PostNorm"
MREAD_SCALE = f"{ADAPTATION}MReadNormScale001"
INVARIANT_KEYS = (
    "dataset_path", "eval_dataset_path", "dataset_type", "tokenizer_path",
    "tokenizer_type", "vocab_size", "base_emb_dim", "base_num_query_heads",
    "base_num_kv_heads", "head_dim", "base_mlp_dim", "base_num_decoder_layers",
    "optimizer", "opt_type", "learning_rate", "adam_b1", "adam_b2", "adam_eps",
    "adam_weight_decay", "gradient_clipping_threshold", "decay_method",
    "steps", "learning_rate_schedule_steps", "warmup_steps_fraction",
    "cosine_learning_rate_final_fraction", "max_target_length", "scan_layers",
    "decoder_block", "train_load_parameters_path", "checkpoint_period",
    "eval_interval", "per_device_batch_size", "eval_per_device_batch_size",
)
STANDARD_ATTENTION_MODULES = frozenset({"query", "key", "value", "out", "qk_norm"})


def _emit(root, identity, output):
  maxtext = os.path.join(root, "MaxText")
  sys.path.insert(0, maxtext)

  import flax
  import jax
  import jax.numpy as jnp
  from jax.sharding import Mesh

  import max_utils
  import pyconfig
  from layers import models

  os.makedirs(os.path.join(os.path.dirname(output), identity), exist_ok=True)
  cfg = pyconfig.initialize(
      [None, os.path.join(maxtext, "configs/base.yml")],
      exp_class=identity,
      train_load_parameters_path="",
      base_output_directory=os.path.dirname(output),
      base_num_decoder_layers=2,
      base_emb_dim=256,
      base_num_query_heads=4,
      base_num_kv_heads=2,
      base_mlp_dim=512,
      max_target_length=4,
      max_prefill_predict_length=2,
      per_device_batch_size=1.0,
      eval_per_device_batch_size=1.0,
      enable_checkpointing=False,
      attention="dot_product",
      dtype="float32",
      weight_dtype="float32",
      log_config=False,
  )
  mesh = Mesh(np.asarray(jax.devices()), ("data",))
  model = models.Transformer(cfg, mesh, None)
  tokens = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
  positions = jnp.arange(4, dtype=jnp.int32)[None]
  mask = jnp.ones_like(tokens)
  segments = jnp.ones_like(tokens)
  args = (tokens, positions, tokens, mask, segments)
  params = model.init(
      {"params": jax.random.key(0), "dropout": jax.random.key(1), "aqt": jax.random.key(2)},
      *args,
      enable_dropout=False,
  )["params"]

  def objective(current_params):
    xent, _, _ = model.apply(
        {"params": current_params}, *args, enable_dropout=False, rngs={"aqt": jax.random.key(3)}
    )
    return jnp.mean(xent)

  loss, grads = jax.value_and_grad(objective)(params)

  def flatten(tree):
    return {
        "/".join(path): np.asarray(value.value if hasattr(value, "value") else value)
        for path, value in flax.traverse_util.flatten_dict(tree).items()
    }

  payload = {
      "config": {key: getattr(cfg, key) for key in INVARIANT_KEYS},
      "bam_enabled": bool(getattr(cfg, "bam_enabled", False)),
      "bam_adaptation": bool(getattr(cfg, "bam_adaptation", False)),
      "bam_adaptation_postnorm": bool(getattr(cfg, "bam_adaptation_postnorm", False)),
      "bam_m_read_norm": getattr(cfg, "bam_m_read_norm", None),
      "bam_m_read_learnable_scale": bool(getattr(cfg, "bam_m_read_learnable_scale", False)),
      "bam_m_read_scale_init": getattr(cfg, "bam_m_read_scale_init", None),
      "train_merge_loaded_params": bool(getattr(cfg, "train_merge_loaded_params", False)),
      "params": flatten(params),
      "logical_axes": {
          "/".join(path): tuple(value.names) if hasattr(value, "names") else None
          for path, value in flax.traverse_util.flatten_dict(params).items()
      },
      "grads": flatten(grads),
      "bam_skip_paths": tuple("/".join(path) for path in max_utils.bam_param_paths_to_skip(params)),
      "loss": np.asarray(loss),
  }
  with open(output, "wb") as stream:
    pickle.dump(payload, stream)


def _is_standard_attention(path):
  parts = path.split("/")
  return "self_attention" in parts and any(part in STANDARD_ATTENTION_MODULES for part in parts)


def _compare(root):
  with tempfile.TemporaryDirectory(prefix="bam-posttrain-alignment-") as tmp:
    outputs = {}
    for identity in (PLAIN, BAM, ADAPTATION, ADAPTATION_V2, POSTNORM, MREAD_SCALE):
      output = os.path.join(tmp, f"{identity}.pkl")
      subprocess.run(
          [sys.executable, __file__, "--emit", identity, "--root", root, "--output", output],
          check=True,
          env={**os.environ, "PYTHONPATH": "", "JAX_PLATFORMS": "cpu"},
      )
      with open(output, "rb") as stream:
        outputs[identity] = pickle.load(stream)

  plain, bam, adaptation, adaptation_v2, postnorm, mread_scale = (
      outputs[PLAIN], outputs[BAM], outputs[ADAPTATION], outputs[ADAPTATION_V2],
      outputs[POSTNORM], outputs[MREAD_SCALE]
  )
  bam_candidates = (bam, adaptation, adaptation_v2, postnorm, mread_scale)
  if not (
      not plain["bam_enabled"] and all(candidate["bam_enabled"] for candidate in bam_candidates)
  ):
    raise AssertionError("BAM enablement does not match the experiment identities")
  if plain["bam_adaptation"] or bam["bam_adaptation"] or not all(
      candidate["bam_adaptation"] for candidate in (adaptation, adaptation_v2, postnorm, mread_scale)
  ):
    raise AssertionError("only the adaptation identity may enable bam_adaptation")
  if (
      any(
          candidate["bam_adaptation_postnorm"]
          for candidate in (plain, bam, adaptation, adaptation_v2, mread_scale)
      )
      or not postnorm["bam_adaptation_postnorm"]
  ):
    raise AssertionError("only the postnorm identity may enable bam_adaptation_postnorm")
  if not all(candidate["train_merge_loaded_params"] for candidate in bam_candidates):
    raise AssertionError("BAM posttrain identities must merge the Plain checkpoint")
  if not all(plain["config"] == candidate["config"] for candidate in (bam, adaptation, postnorm, mread_scale)):
    raise AssertionError("accepted Plain training/runtime invariants changed")
  v2_expected_config = dict(plain["config"], steps=1001)
  if adaptation_v2["config"] != v2_expected_config:
    raise AssertionError("V2 must stop after step-1000 eval without changing the matched LR schedule/runtime")
  if mread_scale["bam_m_read_norm"] != "rms" or not mread_scale["bam_m_read_learnable_scale"]:
    raise AssertionError("MReadNormScale001 must enable RMS-normalized reads with a learnable scale")
  if mread_scale["bam_m_read_scale_init"] != 0.01:
    raise AssertionError("MReadNormScale001 scale initializer changed")

  standard = {path for path in plain["params"] if _is_standard_attention(path)}
  if not standard:
    raise AssertionError("no standard attention parameters found")
  for candidate in bam_candidates:
    candidate_standard = {path for path in candidate["params"] if _is_standard_attention(path)}
    if candidate_standard != standard:
      raise AssertionError("standard Q/K/V/O/QK-norm parameter paths changed")
    for path in standard:
      if candidate["params"][path].shape != plain["params"][path].shape:
        raise AssertionError(f"standard attention shape changed: {path}")
      np.testing.assert_array_equal(candidate["params"][path], plain["params"][path], err_msg=path)

  bam_only = set(bam["params"]) - set(plain["params"])
  adaptation_only = set(adaptation["params"]) - set(bam["params"])
  adaptation_v2_only = set(adaptation_v2["params"]) - set(bam["params"])
  postnorm_only = set(postnorm["params"]) - set(adaptation["params"])
  mread_scale_only = set(mread_scale["params"]) - set(adaptation["params"])
  if set(bam["bam_skip_paths"]) != bam_only:
    raise AssertionError("Plain restore skip set is not exactly the default BAM additions")
  if set(adaptation["bam_skip_paths"]) != bam_only | adaptation_only:
    raise AssertionError("Plain restore skip set is not exactly the adaptation BAM additions")
  if set(adaptation_v2["bam_skip_paths"]) != bam_only | adaptation_v2_only:
    raise AssertionError("Plain restore skip set is not exactly the V2 BAM additions")
  if set(postnorm["bam_skip_paths"]) != bam_only | adaptation_only | postnorm_only:
    raise AssertionError("Plain restore skip set is not exactly the postnorm BAM additions")
  if set(mread_scale["bam_skip_paths"]) != bam_only | adaptation_only | mread_scale_only:
    raise AssertionError("Plain restore skip set is not exactly the MReadNormScale001 BAM additions")
  expected_adaptation_suffixes = {"P_agg_u"}
  matched_adaptation_suffixes = {
      suffix
      for path in adaptation_only
      for suffix in expected_adaptation_suffixes
      if path.endswith(suffix)
  }
  if len(adaptation_only) != len(expected_adaptation_suffixes) or matched_adaptation_suffixes != expected_adaptation_suffixes:
    raise AssertionError(f"unexpected adaptation-only paths: {sorted(adaptation_only)}")
  expected_v2_suffixes = {"P_agg_u", "local_q_decoder", "local_k_decoder"}
  matched_v2_suffixes = {
      suffix
      for path in adaptation_v2_only
      for suffix in expected_v2_suffixes
      if path.endswith(suffix)
  }
  if len(adaptation_v2_only) != len(expected_v2_suffixes) or matched_v2_suffixes != expected_v2_suffixes:
    raise AssertionError(f"unexpected V2-only paths: {sorted(adaptation_v2_only)}")
  for suffix in ("abs_v_row_decoder", "local_q_decoder", "local_k_decoder"):
    path = next(path for path in adaptation_v2["params"] if path.endswith(suffix))
    axes = adaptation_v2["logical_axes"][path]
    if axes is None or "embed" not in axes:
      raise AssertionError(f"V2 decoder contraction axis is not FSDP-shardable: {path}={axes}")
  expected_postnorm_suffixes = {"rms_norm_q/scale", "rms_norm_k/scale", "rms_norm_o/scale"}
  matched_postnorm_suffixes = {
      suffix for path in postnorm_only for suffix in expected_postnorm_suffixes if path.endswith(suffix)
  }
  if len(postnorm_only) != len(expected_postnorm_suffixes) or matched_postnorm_suffixes != expected_postnorm_suffixes:
    raise AssertionError(f"unexpected postnorm-only paths: {sorted(postnorm_only)}")
  for suffix in expected_postnorm_suffixes:
    path = next(path for path in postnorm_only if path.endswith(suffix))
    np.testing.assert_array_equal(
        postnorm["params"][path], np.full_like(postnorm["params"][path], 0.001), err_msg=path
    )
  if len(mread_scale_only) != 1 or not next(iter(mread_scale_only)).endswith("m_read_scale"):
    raise AssertionError(f"unexpected MReadNormScale001-only paths: {sorted(mread_scale_only)}")
  scale_path = next(iter(mread_scale_only))
  np.testing.assert_array_equal(
      mread_scale["params"][scale_path],
      np.full_like(mread_scale["params"][scale_path], 0.01),
      err_msg=scale_path,
  )
  for identity, candidate in outputs.items():
    if not np.all(np.isfinite(candidate["loss"])):
      raise AssertionError(f"non-finite loss for {identity}")
    for path, grad in candidate["grads"].items():
      if not np.all(np.isfinite(grad)):
        raise AssertionError(f"non-finite gradient for {identity}/{path}")

  print(
      "BAM posttrain alignment passed:",
      f"standard_attention={len(standard)}",
      f"bam_only={len(bam_only)}",
      f"adaptation_only={sorted(adaptation_only)}",
      f"adaptation_v2_only={sorted(adaptation_v2_only)}",
      f"postnorm_only={sorted(postnorm_only)}",
      f"mread_scale_only={sorted(mread_scale_only)}",
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
  parser.add_argument("--emit", choices=(PLAIN, BAM, ADAPTATION, ADAPTATION_V2, POSTNORM, MREAD_SCALE))
  parser.add_argument("--output")
  args = parser.parse_args()
  if args.emit:
    _emit(args.root, args.emit, args.output)
  else:
    _compare(args.root)


if __name__ == "__main__":
  main()
