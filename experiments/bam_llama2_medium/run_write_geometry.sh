#!/usr/bin/env bash
set -euo pipefail
REPO=${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}
PYTHON=${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}
OUTPUT=${GEOMETRY_OUTPUT:?}
GCS=${GEOMETRY_GCS:?}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUTPUT/maxtext-output/write-geometry"
gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz /tmp/pile_eval_cohort.npz
cd "$REPO"
("$PYTHON" -c 'import sklearn,scipy; print(sklearn.__version__,scipy.__version__)'
env HARDWARE=tpu "$PYTHON" experiments/bam_llama2_medium/write_geometry.py MaxText/configs/base.yml exp_class=GeometryProbe run_name=write-geometry only_eval=True enable_checkpointing=True async_checkpointing=False base_output_directory="$OUTPUT/maxtext-output/" tensorboard_dir="$OUTPUT/tb" > "$OUTPUT/probe.log" 2>&1
"$PYTHON" experiments/bam_llama2_medium/benchmark_write_geometry.py "$OUTPUT" > "$OUTPUT/cpu_benchmark.log" 2>&1
"$PYTHON" experiments/bam_llama2_medium/analyze_write_geometry.py "$OUTPUT" --n "${GEOMETRY_N:-64}" --workers 8 > "$OUTPUT/analysis.log" 2>&1
echo GEOMETRY_ALL_DONE > "$OUTPUT/DONE") &
job=$!
while kill -0 "$job" 2>/dev/null; do
 sleep 30
 gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS" || true
done
result=0
wait "$job" || result=$?
gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS"
exit "$result"
