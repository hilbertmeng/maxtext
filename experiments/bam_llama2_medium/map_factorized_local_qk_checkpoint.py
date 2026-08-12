"""Map an unpacked BAM parameter checkpoint to the packed LocalQK tree."""

from __future__ import annotations

import argparse

from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

import checkpointing


def _restore_params(path):
  checkpointer = ocp.PyTreeCheckpointer()
  metadata = checkpointer.metadata(path).item_metadata.tree["params"]
  abstract = jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), metadata)
  restore_args = ocp.checkpoint_utils.construct_restore_args(abstract)
  restored = checkpointer.restore(
      path,
      item={"params": abstract},
      transforms={},
      restore_args={"params": restore_args},
  )
  return restored["params"]


def _map_params(source, template):
  source_flat = flatten_dict(source)
  template_flat = flatten_dict(template)
  mapped = {
      key: source_flat[key]
      if key in source_flat and source_flat[key].shape == value.shape
      else value
      for key, value in template_flat.items()
  }

  packed_count = 0
  for key, value in tuple(template_flat.items()):
    if key[-2:] != ("W_local_qk_packed", "kernel"):
      continue
    prefix = key[:-2]
    q_mix = source_flat[prefix + ("W_lq_head_mix", "kernel")]
    k_mix = source_flat[prefix + ("W_lk_head_mix", "kernel")]
    segments = (
        source_flat[prefix + ("W_lq", "kernel")],
        source_flat[prefix + ("W_lq_gate", "kernel")],
        q_mix.reshape(q_mix.shape[0], -1),
        source_flat[prefix + ("W_lk", "kernel")],
        source_flat[prefix + ("W_lk_gate", "kernel")],
        k_mix.reshape(k_mix.shape[0], -1),
    )
    packed = jnp.concatenate(segments, axis=-1)
    if packed.shape != value.shape:
      raise ValueError(f"packed shape mismatch at {'/'.join(key)}: {packed.shape} != {value.shape}")
    mapped[key] = packed
    mapped[prefix + ("W_lq_bias",)] = source_flat[prefix + ("W_lq", "bias")]
    mapped[prefix + ("W_lk_bias",)] = source_flat[prefix + ("W_lk", "bias")]
    packed_count += 1

  if packed_count != 24:
    raise ValueError(f"expected 24 packed layer projections, mapped {packed_count}")
  return unflatten_dict(mapped), packed_count


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", required=True)
  parser.add_argument("--template", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()

  source = _restore_params(args.source)
  template = _restore_params(args.template)
  mapped, packed_count = _map_params(source, template)
  checkpointing.save_params_to_path(args.output, mapped)
  print(f"mapped_packed_layers={packed_count} output={args.output}", flush=True)


if __name__ == "__main__":
  main()
