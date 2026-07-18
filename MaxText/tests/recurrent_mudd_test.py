# Copyright 2026
"""Focused contracts for recurrent full-history MUDD execution."""

from types import SimpleNamespace
import unittest

import checkpointing
import exp
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

  def test_abbc_early_full_layer_order(self):
    self.assertEqual(
        models.build_recurrent_layer_order(28, 3, 17, 2),
        list(range(3)) + list(range(3, 17)) * 2 + list(range(17, 28)),
    )

  def test_abbc_early_config_preserves_physical_and_virtual_depth(self):
    config = exp.Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedMuddNormKVshiftIdentityPreservedGGropeRecurABBCEarly
    self.assertEqual(config.recurrent_physical_num_layers, 28)
    self.assertEqual(config.recurrent_total_layers, 42)
    self.assertEqual(config.recurrent_layer_start, 3)
    self.assertEqual(config.recurrent_layer_end, 17)
    self.assertEqual(config.recurrent_block_repeats, 2)
    self.assertEqual(
        checkpointing.decoder_layers_to_restore(
            SimpleNamespace(
                recurrent_mudd_virtual_state=config.recurrent_mudd_virtual_state,
                recurrent_physical_num_layers=config.recurrent_physical_num_layers,
                num_decoder_layers=config.recurrent_total_layers,
            )
        ),
        28,
    )

  def test_existing_recurrent_windows_are_unchanged(self):
    abbc = exp.Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedMuddNormKVshiftIdentityPreservedGGropeRecurABBC
    aab = exp.Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedMuddNormKVshiftIdentityPreservedGGropeRecurAAB
    self.assertEqual((abbc.recurrent_layer_start, abbc.recurrent_layer_end), (7, 21))
    self.assertEqual((aab.recurrent_layer_start, aab.recurrent_layer_end), (0, 14))

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

  def test_recurrent_mudd_sharding_guard_is_narrowly_raised(self):
    self.assertEqual(
        exp.Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedMuddNormKVshiftIdentityPreservedGGropeRecurABBC.sharding_tolerance,
        0.03,
    )
    self.assertEqual(
        exp.Qwen3LargeArcPostTrainFullNVARC16ShuffleOneFileCosine3e4Cap30TiedMuddNormKVshiftIdentityPreservedGGropeRecurAAB.sharding_tolerance,
        0.03,
    )


if __name__ == "__main__":
  unittest.main()
