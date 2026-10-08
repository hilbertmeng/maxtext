#!/usr/bin/env bash
# DirectC10 vs MHA paired train-step profile on retained v6e-1. Read-only w.r.t. other checkouts.
set -u
PY=/home/lishengping/miniconda3/bin/python
ROOT=$HOME/xd-diag
OUT=$ROOT/prof-dc10
GCS=gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix/ca4491a/xlprop-directc10-v6e1
mkdir -p $OUT; rm -rf $OUT/DONE $OUT/dc10-b* $OUT/mha-b*
exec >> $OUT/driver.log 2>&1
echo "START $(date -u)"
cd $ROOT/maxtext
git fetch -q origin ca4491a839b4a0db8708f1e71e78581363518c0d e30c1b8d4b809164b537b5683d55dba5962ebc56 || git fetch -q origin
for c in ca4491a839b4a0db8708f1e71e78581363518c0d e30c1b8d4b809164b537b5683d55dba5962ebc56; do
  [ -d $ROOT/wt-${c:0:7} ] || git worktree add -f --detach $ROOT/wt-${c:0:7} $c
  git -C $ROOT/wt-${c:0:7} log -1 --format='WT %H'
done
run(){ # name exp commit batch
  local name=$1 exp=$2 wt=$ROOT/wt-$3 bs=$4
  if pgrep -f '[p]ython.*MaxText/train.py' >/dev/null; then echo "BUSY before $name"; return 9; fi
  mkdir -p $OUT/$name; cd $wt
  timeout 3600 env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off $PY MaxText/train.py MaxText/configs/base.yml \
    exp_class=$exp run_name=$name steps=40 dataset_type=synthetic \
    base_output_directory=$OUT tensorboard_dir=$OUT/$name/tb \
    enable_checkpointing=False async_checkpointing=False profiler=xplane \
    skip_first_n_steps_for_profiler=10 profiler_steps=5 profile_cleanly=True \
    upload_all_profiler_results=False per_device_batch_size=$bs jax_cache_dir= \
    > $OUT/$name.log 2>&1
  local rc=$?; echo "RC $name $rc $(date -u)"; return $rc
}
B=2
run dc10-b$B BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile ca4491a $B
if [ $? -ne 0 ] && grep -q -i -E "RESOURCE_EXHAUSTED|out of memory|Ran out of memory" $OUT/dc10-b$B.log; then
  B=1; run dc10-b$B BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdDirectC10TruePile ca4491a $B
fi
run mha-b$B Llama2XLPropTruePileMHA e30c1b8 $B
gsutil -m -q rsync -r $OUT $GCS && echo "UPLOADED $GCS"
echo "B=$B" > $OUT/DONE; echo "DONE $(date -u)"
