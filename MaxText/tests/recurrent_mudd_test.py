# Copyright 2026
"""Focused contracts for recurrent full-history MUDD execution."""

from types import SimpleNamespace
import unittest

import checkpointing
import flax
import max_utils
from layers import attentions
from layers import models


class RecurrentMuddTest(unittest.TestCase):

  def test_abbc_full_layer_order(self):
    self.assertEqual(
        models.build_recurrent_layer_order(28, 7, 21, 2),
        list(range(7)) + list(range(7, 21)) * 2 + list(range(21, 28)),
    )

  def test_aab_full_layer_order(self):
    self.assertEqual(
        models.build_recurrent_layer_order(28, 0, 14, 2),
        list(range(14)) * 2 + list(range(14, 28)),
    )

  def test_special_restore_uses_physical_layer_count(self):
    config = SimpleNamespace(
        recurrent_mudd_virtual_state=True,
        recurrent_physical_num_layers=28,
        num_decoder_layers=42,
    )
    self.assertEqual(checkpointing.decoder_layers_to_restore(config), 28)

  def test_legacy_restore_behavior_is_unchanged(self):
    config = SimpleNamespace(
        recurrent_mudd_virtual_state=False,
        recurrent_physical_num_layers=28,
        num_decoder_layers=42,
    )
    self.assertEqual(checkpointing.decoder_layers_to_restore(config), 42)
    self.assertIsNone(checkpointing.decoder_layers_to_restore(None))

  def test_virtual_cache_names_are_isolated_and_legacy_names_are_stable(self):
    self.assertEqual(attentions.AttentionOp._cache_var_name("ar_key", None), "ar_key")
    self.assertEqual(
        attentions.AttentionOp._cache_var_name("ar_key", "virtual_007"),
        "ar_key__virtual_007",
    )

  def test_qwen_prenorm_maps_to_three_shared_core_norms(self):
    source = ("decoder", "layers_7", "pre_self_attention_layer_norm", "scale")
    self.assertEqual(
        max_utils._recurrent_mudd_target_paths_for_source(source),
        [
            ("decoder", "layers_7", "block", "mudd_qkvnorm", f"pre_self_attention_layer_norm_{letter}", "scale")
            for letter in "qkv"
        ],
    )

  def test_audit_params_unwraps_real_train_state_collection_shape(self):
    model_tree = flax.core.freeze(
        {
            "token_embedder": {"embedding": "embedding"},
            "decoder": {"layers_0": {"block": "layer"}},
        }
    )
    wrapped = flax.core.freeze({"params": model_tree})
    self.assertEqual(max_utils._unwrap_recurrent_mudd_model_params(model_tree), flax.core.unfreeze(model_tree))
    self.assertEqual(max_utils._unwrap_recurrent_mudd_model_params(wrapped), flax.core.unfreeze(model_tree))


if __name__ == "__main__":
  unittest.main()
