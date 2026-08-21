"""V2 C256 cross-layer BAM gate diagnostic on the completed checkpoint."""

from pathlib import Path
import sys

from absl import app


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MaxText"))

import bam_gate_diagnostics
import exp


class BamLlama2MediumV2C256RopeGateDiagnostics(
    exp.BamV2C256FetchScheduleBase
):
  """Read-only randomized-Pile gate probe for RoPE layer selection."""

  model_name = "BamLlama2MediumV2C256RopeGateDiagnostics"
  only_eval = True
  load_parameters_path = (
      "gs://newproject-1-llm_base_models_us-central1/log/"
      "BamLlama2MediumV2/checkpoints/13250/items"
  )
  eval_per_device_batch_size = 32.0
  eval_shuffle_buffer_size = 32768
  tensorboard_dir = "/tmp/bam_v2_c256_rope_gate_diag_tb/"


exp.BamLlama2MediumV2C256RopeGateDiagnostics = (
    BamLlama2MediumV2C256RopeGateDiagnostics
)


if __name__ == "__main__":
  app.run(bam_gate_diagnostics.main)
