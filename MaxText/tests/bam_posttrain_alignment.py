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
      "train_merge_loaded_params": bool(getattr(cfg, "train_merge_loaded_params", False)),
      "params": flatten(params),
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
    for identity in (PLAIN, BAM, ADAPTATION):
      output = os.path.join(tmp, f"{identity}.pkl")
      subprocess.run(
          [sys.executable, __file__, "--emit", identity, "--root", root, "--output", output],
          check=True,
          env={**os.environ, "PYTHONPATH": "", "JAX_PLATFORMS": "cpu"},
      )
      with open(output, "rb") as stream:
        outputs[identity] = pickle.load(stream)

  plain, bam, adaptation = outputs[PLAIN], outputs[BAM], outputs[ADAPTATION]
  if not (not plain["bam_enabled"] and bam["bam_enabled"] and adaptation["bam_enabled"]):
    raise AssertionError("BAM enablement does not match the three experiment identities")
  if plain["bam_adaptation"] or bam["bam_adaptation"] or not adaptation["bam_adaptation"]:
    raise AssertionError("only the adaptation identity may enable bam_adaptation")
  if not bam["train_merge_loaded_params"] or not adaptation["train_merge_loaded_params"]:
    raise AssertionError("BAM posttrain identities must merge the Plain checkpoint")
  if plain["config"] != bam["config"] or bam["config"] != adaptation["config"]:
    raise AssertionError("accepted Plain training/runtime invariants changed")

  standard = {path for path in plain["params"] if _is_standard_attention(path)}
  if not standard:
    raise AssertionError("no standard attention parameters found")
  for candidate in (bam, adaptation):
    candidate_standard = {path for path in candidate["params"] if _is_standard_attention(path)}
    if candidate_standard != standard:
      raise AssertionError("standard Q/K/V/O/QK-norm parameter paths changed")
    for path in standard:
      if candidate["params"][path].shape != plain["params"][path].shape:
        raise AssertionError(f"standard attention shape changed: {path}")
      np.testing.assert_array_equal(candidate["params"][path], plain["params"][path], err_msg=path)

  bam_only = set(bam["params"]) - set(plain["params"])
  adaptation_only = set(adaptation["params"]) - set(bam["params"])
  if set(bam["bam_skip_paths"]) != bam_only:
    raise AssertionError("Plain restore skip set is not exactly the default BAM additions")
  if set(adaptation["bam_skip_paths"]) != bam_only | adaptation_only:
    raise AssertionError("Plain restore skip set is not exactly the adaptation BAM additions")
  if len(adaptation_only) != 1 or not next(iter(adaptation_only)).endswith("P_agg_u"):
    raise AssertionError(f"unexpected adaptation-only paths: {sorted(adaptation_only)}")
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
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
  parser.add_argument("--emit", choices=(PLAIN, BAM, ADAPTATION))
  parser.add_argument("--output")
  args = parser.parse_args()
  if args.emit:
    _emit(args.root, args.emit, args.output)
  else:
    _compare(args.root)


if __name__ == "__main__":
  main()
