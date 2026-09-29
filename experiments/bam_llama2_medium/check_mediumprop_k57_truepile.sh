#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import ast
import sys
from pathlib import Path

root = Path('/data0/xd/mediumprop-k57-truepile')
source = (root / 'MaxText/exp.py').read_text()
tree = ast.parse(source)
classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
name = 'BamLlama2MediumPropK57SharedRank4MLPPerLayerTruePile'
target = classes[name]
parent = 'BamLlama2MediumPropK57SharedRank4MLPPerLayer'
assert len(target.bases) == 1 and isinstance(target.bases[0], ast.Name)
assert target.bases[0].id == parent
assignments = {}
for node in target.body:
    if isinstance(node, ast.Assign):
        for lhs in node.targets:
            if isinstance(lhs, ast.Name):
                assignments[lhs.id] = ast.literal_eval(node.value)
assert set(assignments) == {'model_name', 'DATASET_VARIANT', 'compare_runs', 'jax_cache_dir'}
assert assignments['model_name'] == name
assert assignments['DATASET_VARIANT'] == 'truepile4096'
assert 'BamMediumPropK75EmbedVOnlyQK57TruePile' in assignments['compare_runs']

sys.path.insert(0, '/home/xd/projects/xd_tpu_scripts')
from dataset_paths import declared_variant, resolve_dataset_path
variant = declared_variant(source, name)
path = resolve_dataset_path('us-east5-a', name, class_variant=variant)
assert path == 'gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord_4096', path
print('CONFIG_AND_UE5A_TRUEPILE_ROUTE_OK')
PY
