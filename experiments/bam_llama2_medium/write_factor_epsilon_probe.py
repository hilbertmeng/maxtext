"""Capture Direct PLoc's raw write-V output alongside BAM diagnostic tensors."""

from absl import app

import bam_diagnostics
import exp
import numpy as np


class BamDirectPackedWriteFactorDiagnostics(
    exp.BamLlama2MediumDirectPLocR256GeluPackedLocalQK
):
  """Random-init Direct-Packed probe with raw BAM tensors enabled."""

  bam_diagnostics = True
  eval_per_device_batch_size = 1.0
  eval_shuffle_buffer_size = 32768
  tensorboard_dir = "/tmp/write_factor_eps_tb/"


exp.BamDirectPackedWriteFactorDiagnostics = BamDirectPackedWriteFactorDiagnostics


def _factor_stats(value):
  mean_square = np.mean(np.square(np.asarray(value, np.float32)), axis=-1)
  return {
      "rms": bam_diagnostics._stats(np.sqrt(mean_square)),  # pylint: disable=protected-access
      "rms_squared": bam_diagnostics._stats(mean_square),  # pylint: disable=protected-access
      "fraction_rms_squared_lt_1e-4": float(np.mean(mean_square < 1.0e-4)),
      "normalized_rms_eps_1e-6": bam_diagnostics._stats(  # pylint: disable=protected-access
          np.sqrt(mean_square / (mean_square + 1.0e-6))
      ),
      "normalized_rms_eps_1e-4": bam_diagnostics._stats(  # pylint: disable=protected-access
          np.sqrt(mean_square / (mean_square + 1.0e-4))
      ),
  }


def _write_factor_summary(layer_raw, unused_positions, unused_segments, unused_decay):
  y_std = np.asarray(layer_raw["y_std"], np.float32)
  u2 = np.asarray(layer_raw["read_key_P_loc_up"], np.float32)
  return {
      "u1": _factor_stats(y_std[..., :32]),
      "u2": _factor_stats(u2),
      "write_gate": bam_diagnostics._stats(layer_raw["write_gate"]),  # pylint: disable=protected-access
      "read": {"combined_to_standard": {"p50": float("nan")}},
  }


bam_diagnostics._layer_summary = _write_factor_summary  # pylint: disable=protected-access


# Capture the output of the final P_loc factor projection without adding diagnostic
# branches to BamAttention itself.  bam_diagnostics labels captured module outputs as
# ``read_key_<module>``; the name is retained to reuse its existing raw export path.
bam_diagnostics._READ_PROJECTION_NAMES = (  # pylint: disable=protected-access
    bam_diagnostics._READ_PROJECTION_NAMES | frozenset(("P_loc_up",))
)


if __name__ == "__main__":
  app.run(bam_diagnostics.main)
