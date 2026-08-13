"""CPU alignment of the minimal BAM V2 port against tmp/maxtext.

Run from the target repository root:

  PYTHONPATH=MaxText python MaxText/tests/bam_v2_reference_alignment.py \
    --reference-root=/home/mengqy/Projects/shared_projects/tmp/maxtext

The comparison covers the resolved config, initialized parameters, forward
output, matrix-state update, scalar loss, and every parameter-gradient leaf.
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np


CONFIG_KEYS = (
    "bam_enabled", "bam_layer_modes", "bam_read_sides", "bam_k", "bam_v",
    "bam_n_f", "bam_write_form", "bam_write_eps", "bam_lambda_decay",
    "bam_read_key_mode", "bam_read_key_scale", "bam_read_key_epsilon",
    "bam_read_rms_statistics_dtype", "bam_read_gate_init",
    "bam_local_qk_key_mode", "bam_factorized_head_output_layout",
    "bam_pack_factorized_local_qk", "bam_local_qk_injection",
    "bam_force_activation_dtype", "bam_shared_fetch_mode",
    "bam_fetch_diagonal_one", "bam_read_implementation", "bam_m_read_norm",
    "bam_abs_v_compression_dim", "bam_abs_v_row_output", "bam_write_u_proj",
    "bam_write_v_mode", "bam_write_rms_statistics_dtype",
    "bam_write_v_bottleneck_dim", "bam_write_v_bottleneck_activation",
    "bam_write_outer_implementation", "float32_logits", "scan_layers", "record_internal_nn_metrics",
    "steps", "max_to_keep", "keep_period", "wd_mults",
)


def _emit(root, implementation, output):
  maxtext = os.path.join(root, "MaxText")
  sys.path.insert(0, maxtext)

  import flax
  import jax
  import jax.numpy as jnp
  from jax.sharding import Mesh

  import common_types
  import pyconfig

  run_dir = os.path.join(os.path.dirname(output), implementation)
  os.makedirs(run_dir, exist_ok=True)
  config_overrides = {"bam_adaptation": True} if implementation == "adaptation" else {}
  cfg = pyconfig.initialize(
      [None, os.path.join(maxtext, "configs/base.yml")],
      exp_class="BamLlama2MediumV2",
      base_output_directory=os.path.dirname(output),
      run_name=implementation,
      base_num_decoder_layers=2,
      base_emb_dim=320,
      base_num_query_heads=5,
      base_num_kv_heads=5,
      head_dim=64,
      base_mlp_dim=640,
      max_target_length=4,
      max_prefill_predict_length=2,
      per_device_batch_size=1.0,
      eval_per_device_batch_size=1.0,
      enable_checkpointing=False,
      attention="dot_product",
      dtype="float32",
      weight_dtype="float32",
      log_config=False,
      **config_overrides,
  )
  mesh = Mesh(np.asarray(jax.devices()), ("data",))
  common_kwargs = dict(
      config=cfg,
      num_query_heads=5,
      num_kv_heads=5,
      head_dim=64,
      max_target_length=4,
      max_prefill_predict_length=2,
      mesh=mesh,
      attention_kernel="dot_product",
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      sliding_window_size=4,
      name="self_attention",
  )
  if implementation == "reference":
    from layers.attentions import BamAttention
    module = BamAttention(
        **common_kwargs, layer_mode="local_qk+full", read_side="both", bam_k=32, bam_v=32
    )
  else:
    from layers.bam_v2 import BamV2Attention
    module = BamV2Attention(**common_kwargs, bam_k=32, bam_v=32)

  x = jax.random.normal(jax.random.key(1), (1, 4, 320))
  positions = jnp.arange(4, dtype=jnp.int32)[None]
  segments = jnp.ones((1, 4), dtype=jnp.int32)
  matrix = jax.random.normal(jax.random.key(2), (1, 4, 32, 32))
  call_kwargs = dict(
      decoder_segment_ids=segments,
      model_mode=common_types.MODEL_MODE_TRAIN,
      M_in=matrix,
  )
  params = module.init(jax.random.key(3), x, x, positions, **call_kwargs)["params"]

  # Compare the initializer tree itself, then exercise gradients at a shared
  # nonzero point so zero-initialized read kernels cannot hide path mistakes.
  leaves, tree_def = jax.tree.flatten(params)
  eval_params = jax.tree.unflatten(
      tree_def,
      [
          0.02 * jax.random.normal(jax.random.fold_in(jax.random.key(11), index), leaf.shape)
          for index, leaf in enumerate(leaves)
      ],
  )

  def objective(current_params):
    y, matrix_out = module.apply(
        {"params": current_params}, x, x, positions, **call_kwargs
    )
    loss = jnp.mean(jnp.square(y)) + 0.01 * jnp.mean(jnp.square(matrix_out))
    return loss, (y, matrix_out)

  (loss, (y, matrix_out)), grads = jax.value_and_grad(objective, has_aux=True)(eval_params)
  flatten = lambda tree: {
      "/".join(path): np.asarray(value.value if hasattr(value, "value") else value)
      for path, value in flax.traverse_util.flatten_dict(tree).items()
  }
  payload = dict(
      config={key: getattr(cfg, key) for key in CONFIG_KEYS},
      bam_adaptation=getattr(cfg, "bam_adaptation", False),
      params=flatten(params),
      grads=flatten(grads),
      output=np.asarray(y),
      matrix_out=np.asarray(matrix_out),
      loss=np.asarray(loss),
  )
  with open(output, "wb") as stream:
    pickle.dump(payload, stream)


def _compare(reference_root, target_root):
  with tempfile.TemporaryDirectory(prefix="bam-v2-alignment-") as tmp:
    outputs = {}
    for implementation, root in (
        ("reference", reference_root),
        ("target", target_root),
        ("adaptation", target_root),
    ):
      output = os.path.join(tmp, f"{implementation}.pkl")
      subprocess.run(
          [sys.executable, __file__, "--emit", implementation, "--root", root, "--output", output],
          check=True,
          env={**os.environ, "PYTHONPATH": ""},
      )
      with open(output, "rb") as stream:
        outputs[implementation] = pickle.load(stream)

    reference, target = outputs["reference"], outputs["target"]
    adaptation = outputs["adaptation"]
    if target["bam_adaptation"] or not adaptation["bam_adaptation"]:
      raise AssertionError("bam_adaptation must default to false and enable only by override")
    if reference["config"] != target["config"]:
      raise AssertionError(f"resolved config differs: {reference['config']} != {target['config']}")
    for tree_name in ("params", "grads"):
      if reference[tree_name].keys() != target[tree_name].keys():
        raise AssertionError(f"{tree_name} paths differ")
      for path in reference[tree_name]:
        if not np.all(np.isfinite(target[tree_name][path])):
          raise AssertionError(f"non-finite {tree_name}/{path}")
        np.testing.assert_allclose(
            target[tree_name][path], reference[tree_name][path], rtol=1e-6, atol=1e-6,
            err_msg=f"{tree_name}/{path}",
        )
    for name in ("output", "matrix_out", "loss"):
      np.testing.assert_allclose(target[name], reference[name], rtol=1e-6, atol=1e-6, err_msg=name)

    # The adaptation path intentionally adds only the projected-U kernel.  The
    # projected row read reuses abs_v_row_decoder, which is retained in the
    # default V2 checkpoint tree for reference compatibility.
    extra_params = set(adaptation["params"]) - set(target["params"])
    if len(extra_params) != 1 or not next(iter(extra_params)).endswith("P_agg_u"):
      raise AssertionError(f"unexpected adaptation parameter paths: {sorted(extra_params)}")
    if set(target["params"]) - set(adaptation["params"]):
      raise AssertionError("adaptation removed default parameter paths")
    for name in ("output", "matrix_out", "loss"):
      if not np.all(np.isfinite(adaptation[name])):
        raise AssertionError(f"non-finite adaptation {name}")
    for suffix in ("P_agg_u", "abs_v_row_decoder"):
      paths = [path for path in adaptation["grads"] if path.endswith(suffix)]
      if len(paths) != 1 or not np.all(np.isfinite(adaptation["grads"][paths[0]])):
        raise AssertionError(f"invalid adaptation gradient for {suffix}: {paths}")
      if not np.any(adaptation["grads"][paths[0]] != 0):
        raise AssertionError(f"adaptation gradient is identically zero for {suffix}")
    print(
        "BAM V2 alignment passed:",
        f"params={len(target['params'])}",
        f"grads={len(target['grads'])}",
        f"adaptation_params={len(adaptation['params'])}",
        "rtol=atol=1e-6",
    )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--reference-root")
  parser.add_argument("--target-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
  parser.add_argument("--emit", choices=("reference", "target", "adaptation"))
  parser.add_argument("--root")
  parser.add_argument("--output")
  args = parser.parse_args()
  if args.emit:
    _emit(args.root, args.emit, args.output)
  else:
    _compare(args.reference_root, args.target_root)


if __name__ == "__main__":
  main()
