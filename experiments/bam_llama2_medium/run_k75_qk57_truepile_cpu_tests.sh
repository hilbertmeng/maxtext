#!/usr/bin/env bash
set -euo pipefail

REPO=/data0/xd/mediumprop-k75-qk57-truepile
PYTHON=/data0/xd/conda/envs/maxtext-cpu/bin/python
cd "$REPO"
env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 \
  PYTHONPATH=MaxText:MaxText/tests "$PYTHON" -m unittest \
  bam_local_v_modes_test.LocalVModeTest.test_v_only_forward_and_gradients_do_not_require_qk_or_o
"$PYTHON" - <<'PY'
import runpy

exp = runpy.run_path('MaxText/exp.py')
old = exp['BamMediumPropK75EmbedVOnlyQK57']
new = exp['BamMediumPropK75EmbedVOnlyQK57TruePile']
old_values = {key: getattr(old, key) for key in dir(old) if not key.startswith('__')}
new_values = {key: getattr(new, key) for key in dir(new) if not key.startswith('__')}
changed = {key for key in old_values.keys() | new_values.keys()
           if old_values.get(key) != new_values.get(key)}
assert changed == {'model_name', 'dataset_path', 'compare_runs', 'jax_cache_dir'}, changed
assert new.dataset_path.endswith('pythia_pile_idxmaps_tfrecord_4096')
assert new.max_target_length == 4096 and new.per_device_batch_size == 16.0
print('K75_TRUEPILE_CONFIG_OK')
PY
