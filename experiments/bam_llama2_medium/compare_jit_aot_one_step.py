"""Compare native-JIT and loaded-AOT train steps on identical TPU inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MAXTEXT_ROOT = REPO_ROOT / "MaxText"
sys.path.insert(0, str(MAXTEXT_ROOT))

from flax.linen import partitioning as nn_partitioning  # pylint: disable=wrong-import-position
import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
import tensorflow as tf  # pylint: disable=wrong-import-position

import max_utils  # pylint: disable=wrong-import-position
import maxtext_utils  # pylint: disable=wrong-import-position
import pyconfig  # pylint: disable=wrong-import-position
import train  # pylint: disable=wrong-import-position


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", required=True)
  args, config_args = parser.parse_known_args()
  if config_args[:1] == ["--"]:
    config_args = config_args[1:]
  if not config_args:
    parser.error("MaxText config arguments are required after --")
  return args, [sys.argv[0], *config_args]


def _copy_tree(tree: Any) -> Any:
  copied = jax.tree.map(lambda x: jnp.array(x, copy=True), tree)
  return jax.block_until_ready(copied)


def _tree_stats(lhs: Any, rhs: Any) -> dict[str, Any]:
  pairs = list(zip(jax.tree.leaves(lhs), jax.tree.leaves(rhs), strict=True))
  if not pairs:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return {"l2": zero, "ref_l2": zero, "relative_l2": zero, "max_abs": zero}
  squared = []
  ref_squared = []
  max_abs = []
  for left, right in pairs:
    left32 = jnp.asarray(left, dtype=jnp.float32)
    right32 = jnp.asarray(right, dtype=jnp.float32)
    diff = left32 - right32
    squared.append(jnp.sum(jnp.square(diff)))
    ref_squared.append(jnp.sum(jnp.square(right32)))
    max_abs.append(jnp.max(jnp.abs(diff)))
  l2 = jnp.sqrt(jnp.sum(jnp.stack(squared)))
  ref_l2 = jnp.sqrt(jnp.sum(jnp.stack(ref_squared)))
  return {
      "l2": l2,
      "ref_l2": ref_l2,
      "relative_l2": l2 / jnp.maximum(ref_l2, jnp.finfo(jnp.float32).tiny),
      "max_abs": jnp.max(jnp.stack(max_abs)),
  }


def _per_leaf_stats(lhs: Any, rhs: Any) -> Any:
  def compare(left: Any, right: Any) -> jax.Array:
    left32 = jnp.asarray(left, dtype=jnp.float32)
    right32 = jnp.asarray(right, dtype=jnp.float32)
    diff = left32 - right32
    l2 = jnp.sqrt(jnp.sum(jnp.square(diff)))
    ref_l2 = jnp.sqrt(jnp.sum(jnp.square(right32)))
    return jnp.stack(
        [
            jnp.max(jnp.abs(diff)),
            l2,
            l2 / jnp.maximum(ref_l2, jnp.finfo(jnp.float32).tiny),
        ]
    )

  return jax.tree.map(compare, lhs, rhs)


def _path_name(path: tuple[Any, ...]) -> str:
  parts = []
  for entry in path:
    if hasattr(entry, "key"):
      parts.append(str(entry.key))
    elif hasattr(entry, "idx"):
      parts.append(str(entry.idx))
    elif hasattr(entry, "name"):
      parts.append(str(entry.name))
    else:
      parts.append(str(entry))
  return "/".join(parts)


def _top_leaf_differences(stats_tree: Any, limit: int = 40) -> list[dict[str, Any]]:
  rows = []
  for path, value in jax.tree_util.tree_flatten_with_path(stats_tree)[0]:
    max_abs, l2, relative_l2 = [float(x) for x in value]
    rows.append(
        {
            "path": _path_name(path),
            "max_abs": max_abs,
            "l2": l2,
            "relative_l2": relative_l2,
        }
    )
  rows.sort(key=lambda row: (row["l2"], row["max_abs"]), reverse=True)
  return rows[:limit]


def _scalar_metric_differences(jit_metrics: Any, aot_metrics: Any) -> list[dict[str, Any]]:
  jit_flat = {
      _path_name(path): value
      for path, value in jax.tree_util.tree_flatten_with_path(jit_metrics)[0]
  }
  aot_flat = {
      _path_name(path): value
      for path, value in jax.tree_util.tree_flatten_with_path(aot_metrics)[0]
  }
  rows = []
  for path in sorted(jit_flat.keys() & aot_flat.keys()):
    jit_value = jnp.asarray(jit_flat[path])
    aot_value = jnp.asarray(aot_flat[path])
    if jit_value.size != 1 or aot_value.size != 1:
      continue
    jit_scalar = float(jit_value.reshape(()))
    aot_scalar = float(aot_value.reshape(()))
    rows.append(
        {
            "path": path,
            "jit": jit_scalar,
            "aot": aot_scalar,
            "aot_minus_jit": aot_scalar - jit_scalar,
            "relative_abs": abs(aot_scalar - jit_scalar) / max(abs(jit_scalar), 1e-30),
        }
    )
  rows.sort(key=lambda row: abs(row["aot_minus_jit"]), reverse=True)
  return rows


def _hlo_text(compiled: Any) -> str:
  if hasattr(compiled, "as_text"):
    text = compiled.as_text()
    if isinstance(text, str):
      return text
  executable = compiled.runtime_executable()
  return "\n\n".join(module.to_string() for module in executable.hlo_modules())


def main() -> None:
  args, config_argv = _parse_args()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "")
        + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  config = pyconfig.initialize(config_argv)
  train.validate_train_config(config)
  if not config.compiled_trainstep_file:
    raise ValueError("compiled_trainstep_file must point to the AOT executable")

  (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      _,
      data_iterator,
      _,
      state,
  ) = train.setup_train_loop(config)
  del writer, checkpoint_manager

  batch = train.load_next_batch(data_iterator, None, config)
  train.check_example_batch(config, example_batch=batch)
  step = train.get_first_step(state)
  rng = jax.jit(jax.random.fold_in)(init_rng, step)
  (
      functional_train,
      in_shardings,
      out_shardings,
      static_argnums,
      donate_argnums,
  ) = maxtext_utils.get_functional_train_with_signature(
      train.train_step, mesh, state_mesh_shardings, model, config
  )

  jit_input = _copy_tree(state)
  p_jit = jax.jit(
      functional_train,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      static_argnums=static_argnums,
      donate_argnums=donate_argnums,
  )
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    jit_lowered = p_jit.lower(jit_input, batch, rng)
    jit_compiled = jit_lowered.compile()
    p_aot = maxtext_utils.load_compiled(config, functional_train, state, mesh)
    jit_state, jit_metrics = jit_compiled(jit_input, batch, rng)
    aot_state, aot_metrics = p_aot(state, batch, rng)
  jax.block_until_ready((jit_state, jit_metrics, aot_state, aot_metrics))

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_stats = jax.jit(_tree_stats)(jit_state, aot_state)
    param_stats = jax.jit(_tree_stats)(jit_state.params, aot_state.params)
    opt_stats = jax.jit(_tree_stats)(jit_state.opt_state, aot_state.opt_state)
    per_param = jax.jit(_per_leaf_stats)(jit_state.params, aot_state.params)
  state_stats, param_stats, opt_stats, per_param = jax.device_get(
      (state_stats, param_stats, opt_stats, per_param)
  )
  jit_metrics, aot_metrics = jax.device_get((jit_metrics, aot_metrics))

  output_dir = Path(args.output_dir)
  if jax.process_index() == 0:
    output_dir.mkdir(parents=True, exist_ok=True)
    jit_hlo = _hlo_text(jit_compiled)
    aot_hlo = _hlo_text(p_aot)
    (output_dir / "jit_optimized_hlo.txt").write_text(jit_hlo, encoding="utf-8")
    (output_dir / "aot_optimized_hlo.txt").write_text(aot_hlo, encoding="utf-8")

    def host_stats(values: dict[str, Any]) -> dict[str, float]:
      return {key: float(value) for key, value in values.items()}

    report = {
        "step": int(step),
        "jit_hlo_sha256": hashlib.sha256(jit_hlo.encode()).hexdigest(),
        "aot_hlo_sha256": hashlib.sha256(aot_hlo.encode()).hexdigest(),
        "jit_hlo_bytes": len(jit_hlo.encode()),
        "aot_hlo_bytes": len(aot_hlo.encode()),
        "state_difference": host_stats(state_stats),
        "parameter_difference": host_stats(param_stats),
        "optimizer_difference": host_stats(opt_stats),
        "top_parameter_differences": _top_leaf_differences(per_param),
        "metric_differences": _scalar_metric_differences(jit_metrics, aot_metrics),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
