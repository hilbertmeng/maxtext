#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
  echo "usage: $0 TPU ZONE COMMIT LABEL EXP_CLASS..." >&2
  exit 2
fi

tpu=$1
zone=$2
commit=$3
label=$4
shift 4

project=${TPU_PROJECT:-newproject-1-451205}
account=${TPU_ACCOUNT:-626151558586-compute@developer.gserviceaccount.com}
configuration=${TPU_GCLOUD_CONFIGURATION:-xd-tpu}
root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
repo=${MAXTEXT_REPO:-$root/maxtext}
log_root=${PROFILE_LOG_ROOT:-$root/logs}
remote_root=${PROFILE_REMOTE_ROOT:-/home/lishengping/xd/profile_outputs/profile_matrix}
gcs_root=${PROFILE_GCS_ROOT:-gs://newproject-1-llm_base_models_us-central1/log/diagnostics/profile_matrix}
collector=${XPLANE_COLLECTOR:-$root/collect_xplane.sh}
smoke=${PROFILE_SMOKE:-/home/lishengping/run_train_smoke.sh}
compiled_smoke=${PROFILE_COMPILED_SMOKE:-/home/lishengping/run_train_smoke_compiled.sh}
preflight=${TPU_AG_PREFLIGHT:-$root/tpu_ag_preflight.sh}
aot_root=${AOT_ROOT:-}
profile_steps=${PROFILE_STEPS:-20}
done_step=${PROFILE_DONE_STEP:-15}
profile_skip=${PROFILE_SKIP:-10}
profile_period=${PROFILE_PERIOD:-1000}
profile_duration=${PROFILE_DURATION:-5}
trace_count=${PROFILE_TRACE_COUNT:-}
matrix_id=${PROFILE_MATRIX_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}

[[ "$commit" =~ ^[0-9a-f]{7,40}$ ]] || { echo "ERROR: invalid commit: $commit" >&2; exit 2; }
[[ "$label" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "ERROR: invalid label: $label" >&2; exit 2; }
[[ "$matrix_id" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "ERROR: invalid matrix id: $matrix_id" >&2; exit 2; }
for exp in "$@"; do
  [[ "$exp" =~ ^[A-Za-z0-9_]+$ ]] || { echo "ERROR: invalid exp class: $exp" >&2; exit 2; }
done

# Bash may read a script lazily. Execute a private snapshot so later edits cannot corrupt a live run.
if [[ ${PROFILE_RUNNER_SEALED:-0} != 1 ]]; then
  mkdir -p "$log_root"
  sealed="$log_root/.run-profile-${commit:0:7}-${label}-$(date -u +%Y%m%dT%H%M%SZ)-$$.sh"
  install -m 700 "$0" "$sealed"
  echo "SEALED_RUNNER path=$sealed sha256=$(sha256sum "$sealed" | awk '{print $1}')"
  exec env PROFILE_RUNNER_SEALED=1 PROFILE_MATRIX_ID="$matrix_id" \
    "$sealed" "$tpu" "$zone" "$commit" "$label" "$@"
fi
trap 'rm -f -- "$0"' EXIT

"$preflight"
export CLOUDSDK_ACTIVE_CONFIG_NAME="$configuration"
gcloud_base=(gcloud --configuration="$configuration" --account="$account")
manifest="$log_root/profile-matrix-${commit:0:7}-${label}-${matrix_id}.tsv"
printf 'utc\tarm\tstate\tdetail\n' >"$manifest"
record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" >>"$manifest"
}
record MATRIX PREFLIGHT_OK "tpu=$tpu zone=$zone commit=$commit"

[[ -x "$collector" ]] || { echo "ERROR: missing collector: $collector" >&2; exit 1; }
accelerator=$("${gcloud_base[@]}" compute tpus tpu-vm describe "$tpu" --zone="$zone" \
  --project="$project" --format='value(acceleratorType)' 2>/dev/null)
if [[ "$accelerator" == v5p-* && -z "$aot_root" && ${ALLOW_TARGET_JIT:-0} != 1 ]]; then
  echo "ERROR: full target profile on $accelerator requires AOT_ROOT" >&2
  exit 1
fi
if [[ -z "$trace_count" ]]; then
  trace_count=$([[ -n "$aot_root" ]] && echo 1 || echo 2)
fi

"${gcloud_base[@]}" compute tpus tpu-vm ssh --internal-ip "$tpu" --zone="$zone" \
  --project="$project" --worker=all --command="cd '$repo' && \
  git fetch origin refactor-bam && git reset --hard && git clean -ffd && \
  git checkout --detach '$commit' && \
  test \"\$(git rev-parse HEAD)\" = \"\$(git rev-parse '$commit^{commit}')\"" </dev/null

aot_dir="/tmp/xd-profile-aot/${commit:0:7}-$label"
if [[ -n "$aot_root" ]]; then
  for exp in "$@"; do
    gsutil stat "${aot_root%/}/$exp.pickle" >/dev/null
  done
  for exp in "$@"; do
    "${gcloud_base[@]}" compute tpus tpu-vm ssh --internal-ip "$tpu" --zone="$zone" \
      --project="$project" --worker=all --command="mkdir -p '$aot_dir' && \
      gsutil -q cp '${aot_root%/}/$exp.pickle' '$aot_dir/$exp.pickle'" </dev/null
  done
fi

collector_pid=
current_run=
cleanup() {
  if [[ -n "$collector_pid" ]]; then
    kill "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
  if [[ -n "$current_run" ]]; then
    "${gcloud_base[@]}" compute tpus tpu-vm ssh --internal-ip "$tpu" --zone="$zone" \
      --project="$project" --worker=all --command="pkill -KILL -f \
      '[M]axText/train.py.*run_name=$current_run' 2>/dev/null || true; \
      sudo rm -f /tmp/libtpu_lockfile" </dev/null 2>/dev/null || true
  fi
}
finalize() {
  cleanup
  [[ ${PROFILE_RUNNER_SEALED:-0} == 1 ]] && rm -f -- "$0"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

index=0
for exp in "$@"; do
  current_run="Profile${commit:0:7}_${label}_${matrix_id}_${index}_${exp}"
  remote_run="$remote_root/$current_run"
  train_log="/home/lishengping/train_${current_run}.log"
  collector_log="$log_root/${current_run}-collector.log"
  gcs_run="${gcs_root%/}/${commit:0:7}/$label/$current_run"

  if [[ -n "$aot_root" ]]; then
    smoke_command="env PROFILE_SKIP='$profile_skip' PROFILE_PERIOD='$profile_period' \
      PROFILE_DURATION='$profile_duration' '$compiled_smoke' '$exp' '$current_run' \
      '$aot_dir/$exp.pickle' '$profile_steps'"
  else
    smoke_command="'$smoke' '$exp' '$current_run' '$profile_steps'"
  fi

  "${gcloud_base[@]}" compute tpus tpu-vm ssh --internal-ip "$tpu" --zone="$zone" \
    --project="$project" --worker=all --command="rm -rf '$remote_run'; mkdir -p '$remote_run'; \
    sudo rm -f /tmp/libtpu_lockfile; cd '$repo'; nohup env SMOKE_OUTPUT='$remote_run' \
    $smoke_command >'$train_log' 2>&1 </dev/null &" </dev/null

  "$collector" "$tpu" "$zone" "$remote_run/tensorboard" "$gcs_run" \
    "$project" "$trace_count" auto >"$collector_log" 2>&1 &
  collector_pid=$!
  echo "$current_run launch_utc=$(date -u +%FT%TZ)"
  record "$current_run" LAUNCHED "trace_count=$trace_count"

  deadline=$((SECONDS + ${PROFILE_ARM_TIMEOUT_SECONDS:-2400}))
  first=false
  done=false
  while (( SECONDS < deadline )); do
    state=$("${gcloud_base[@]}" compute tpus tpu-vm ssh --internal-ip "$tpu" --zone="$zone" \
      --project="$project" --worker=0 --command="if grep -q 'completed step: $done_step,' \
      '$train_log' 2>/dev/null; then echo DONE; elif grep -q 'completed step: 0,' \
      '$train_log' 2>/dev/null; then echo FIRST; elif pgrep -f \
      '[M]axText/train.py.*run_name=$current_run' >/dev/null; then echo RUNNING; \
      elif grep -Eq 'ValueError:|AssertionError:|TypeError:|Traceback' '$train_log' \
      2>/dev/null; then echo ERROR; else echo EXITED; fi" </dev/null 2>/dev/null | tail -1)
    case "$state" in
      DONE) done=true; break ;;
      FIRST)
        if ! $first; then
          first=true
          echo "$current_run first_step_utc=$(date -u +%FT%TZ)"
          record "$current_run" FIRST_STEP step=0
        fi
        ;;
      ERROR|EXITED)
        echo "$current_run $state" >&2
        record "$current_run" "$state" "before_step=$done_step"
        "${gcloud_base[@]}" compute tpus tpu-vm ssh --internal-ip "$tpu" --zone="$zone" \
          --project="$project" --worker=0 --command="tail -100 '$train_log'" </dev/null >&2 || true
        break
        ;;
    esac
    sleep 5
  done

  if $done; then
    wait "$collector_pid"
    collector_pid=
    record "$current_run" TRACE_VERIFIED "gcs=$gcs_run"
  else
    echo "ERROR: arm failed before step $done_step: $current_run" >&2
    exit 1
  fi
  cleanup
  "${gcloud_base[@]}" compute tpus tpu-vm ssh --internal-ip "$tpu" --zone="$zone" \
    --project="$project" --worker=0 --command="grep 'completed step:' '$train_log' | tail -6" \
    </dev/null || true
  current_run=
  index=$((index + 1))
done
record MATRIX COMPLETE "arms=$index"
echo "PROFILE_MATRIX_OK manifest=$manifest"
