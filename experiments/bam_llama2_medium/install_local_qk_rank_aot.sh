#!/usr/bin/env bash
set -euo pipefail

REGION=${1:?region is required}
COMMIT=8235ccd561eedd6f65befdafc795ed58f33fa13d
BOOTSTRAP=gs://newproject-1-conda_script_${REGION}/maxtext_aot_bootstrap
AOT_ROOT=gs://newproject-1-llm_base_models_us-central1/log/diagnostics/local_qk_rank/8235ccd/aot_v5p32
REPO=/home/lishengping/xd/projects/maxtext
PROFILE_ROOT=/home/lishengping/xd/profile_outputs/local_qk_rank
LABEL=v5p32_xl_full24
PROFILE_DURATION=

case "$(basename "$0")" in
  *_xl16r4.sh) EXP=BamXL16V2LocalQKRank4MulReduceFullLayerProfile ;;
  *_xl32r1.sh) EXP=BamXL32V2LocalQKRankControlFullLayerProfile ;;
  *_xl32r2.sh) EXP=BamXL32V2LocalQKRank2MulReduceFullLayerProfile ;;
  *_xl32r4.sh)
    EXP=BamXL32V2LocalQKRank4MulReduceFullLayerProfile
    PROFILE_DURATION=1
    PROFILE_SKIP=0
    PROFILE_PERIOD=10
    ;;
  *) EXP= ;;
esac

gsutil -q cp "$BOOTSTRAP/install_xd_maxtext_jax081_fast.sh" /tmp/install_xd_maxtext_jax081_fast.sh
if [[ -n "$EXP" ]]; then
  gsutil -q cp "$AOT_ROOT/$EXP.pickle" /tmp/ &
else
  gsutil -m -q cp "$AOT_ROOT/*.pickle" /tmp/ &
fi
aot_download_pid=$!
bash /tmp/install_xd_maxtext_jax081_fast.sh "$REGION"
wait "$aot_download_pid"

cd "$REPO"
git fetch -q origin refactor-bam
git reset --hard -q
git clean -ffdq
git checkout --detach -q "$COMMIT"
test "$(git rev-parse HEAD)" = "$COMMIT"

gsutil -q cp "$BOOTSTRAP/run_train_smoke_compiled.sh" /tmp/run_train_smoke_compiled.sh
chmod 755 /tmp/run_train_smoke_compiled.sh

if [[ -n "$EXP" ]]; then
  RUN="LQRank${COMMIT:0:7}_${LABEL}_0_${EXP}"
  OUTPUT="$PROFILE_ROOT/$RUN"
  rm -rf "$OUTPUT"
  mkdir -p "$OUTPUT"
  sudo rm -f /tmp/libtpu_lockfile
  cd "$REPO"
  nohup env SMOKE_OUTPUT="$OUTPUT" PROFILE_DURATION="$PROFILE_DURATION" \
    PROFILE_SKIP="${PROFILE_SKIP:-}" PROFILE_PERIOD="${PROFILE_PERIOD:-}" \
    /tmp/run_train_smoke_compiled.sh \
    "$EXP" "$RUN" "/tmp/$EXP.pickle" 20 \
    >"/home/lishengping/train_$RUN.log" 2>&1 </dev/null &
fi
