#!/usr/bin/env bash
# usage: bench_pair.sh LOOM_DIR MHA_DIR OUT_PREFIX GPU REPEATS [bench.py args...]
# Alternates LoomFormer / MHA processes REPEATS times on one GPU; summarize.py takes medians.
set -u
loom=$1 mha=$2 out=$3 gpu=$4 reps=$5; shift 5
here=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/xd/venvs/bam-infer/bin/python}
for r in $(seq 1 $reps); do
  for kind in loom mha; do
    dir=$([ $kind = loom ] && echo $loom || echo $mha)
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu $PY $here/bench.py $dir --out $out-$kind-$r.json "$@" \
      > $out-$kind-$r.log 2>&1 || echo "BENCH_FAIL $kind $r"
  done
done
$PY $here/summarize.py $out
