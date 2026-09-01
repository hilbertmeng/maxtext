"""Run BAM diagnostics with an external whole-M read normalization ablation.

This launcher deliberately monkey-patches ``BamAttention._matrix_for_read`` so
the diagnostic can test historical commits without adding experiment-only
branches to the production attention implementation.

Environment:
  BAM_MNORM_MODE    ``none``, ``unit``, or ``shared_rescale``.
  BAM_MNORM_SCALE   Shared post-normalization RMS for ``shared_rescale``.
  BAM_MNORM_TARGET  Python script to execute after installing the patch.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

import jax
import jax.numpy as jnp


_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "MaxText"))

from layers import attentions  # pylint: disable=wrong-import-position


def _install_patch() -> None:
  mode = os.environ.get("BAM_MNORM_MODE", "unit")
  if mode not in ("none", "unit", "shared_rescale"):
    raise ValueError(f"unsupported BAM_MNORM_MODE={mode!r}")
  scale = float(os.environ.get("BAM_MNORM_SCALE", "1"))
  if mode == "shared_rescale" and scale <= 0:
    raise ValueError(f"BAM_MNORM_SCALE must be positive, got {scale}")

  def matrix_for_read(self, matrix):
    if matrix is None or mode == "none":
      return matrix
    epsilon = float(getattr(
        self, "_rms_epsilon", self.config.normalization_layer_epsilon))
    normalized = matrix * jax.lax.rsqrt(
        jnp.mean(matrix ** 2, axis=(-2, -1), keepdims=True) + epsilon)
    if mode == "shared_rescale":
      normalized = normalized * jnp.asarray(scale, normalized.dtype)
    return normalized

  attentions.BamAttention._matrix_for_read = matrix_for_read
  print(f"BAM_MNORM_PATCH mode={mode} scale={scale}", flush=True)


def main() -> None:
  target = os.environ.get("BAM_MNORM_TARGET")
  if not target:
    raise ValueError("BAM_MNORM_TARGET is required")
  target_path = Path(target)
  if not target_path.is_absolute():
    target_path = _REPO / target_path
  if not target_path.is_file():
    raise FileNotFoundError(target_path)
  _install_patch()
  runpy.run_path(str(target_path), run_name="__main__")


if __name__ == "__main__":
  main()
