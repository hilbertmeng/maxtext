#!/usr/bin/env bash
# Cancel an owned TPU queue or standalone VM and verify that both are absent.
set -u

TPU_NAME="${1:?usage: delete_tpu_xd.sh TPU_NAME ZONE PROJECT_ID}"
ZONE="${2:?usage: delete_tpu_xd.sh TPU_NAME ZONE PROJECT_ID}"
PROJECT_ID="${3:?usage: delete_tpu_xd.sh TPU_NAME ZONE PROJECT_ID}"
ATTEMPTS="${RELEASE_ATTEMPTS:-30}"
WAIT_SECONDS="${RELEASE_WAIT_SECONDS:-10}"
GCLOUD_DELETE_TIMEOUT_SECONDS="${GCLOUD_DELETE_TIMEOUT_SECONDS:-120}"
ROOT="${TPU_AG_PROJECTS_DIR:-/home/lishengping/xd/projects}"
CREATOR_PID_FILE="$ROOT/logs/${TPU_NAME}-create.pid"

# Names may be identical across regions during migration. Never stop or unlink
# the destination creator when releasing the source region.
if [[ -s "$CREATOR_PID_FILE" ]]; then
    creator_pid=$(<"$CREATOR_PID_FILE")
    remove_creator_pid=false
    if [[ "$creator_pid" =~ ^[0-9]+$ ]]; then
        if kill -0 "$creator_pid" 2>/dev/null; then
            creator_args=()
            creator_env=()
            mapfile -d '' -t creator_args <"/proc/$creator_pid/cmdline" 2>/dev/null || true
            mapfile -d '' -t creator_env <"/proc/$creator_pid/environ" 2>/dev/null || true
            creator_project=newproject-1-451205
            for entry in "${creator_env[@]}"; do
                [[ "$entry" == TPU_PROJECT=* ]] && creator_project="${entry#TPU_PROJECT=}"
            done
            for ((i=0; i<${#creator_args[@]}; i++)); do
                if [[ "${creator_args[i]##*/}" == create_standalone_tpu.sh ]] &&
                   [[ "${creator_args[i+1]:-}" == "$TPU_NAME" ]] &&
                   [[ "${creator_args[i+3]:-}" == "$ZONE" ]] &&
                   [[ "$creator_project" == "$PROJECT_ID" ]]; then
                    kill "$creator_pid" 2>/dev/null || true
                    for _ in {1..20}; do
                        kill -0 "$creator_pid" 2>/dev/null || break
                        sleep 0.25
                    done
                    remove_creator_pid=true
                    break
                fi
            done
        else
            remove_creator_pid=true
        fi
    else
        remove_creator_pid=true
    fi
    # A concurrent creator may have replaced the file during the wait above.
    if $remove_creator_pid && [[ -f "$CREATOR_PID_FILE" ]] &&
       [[ "$(<"$CREATOR_PID_FILE")" == "$creator_pid" ]]; then
        rm -f "$CREATOR_PID_FILE"
    fi
fi

describe_resource() {
    local kind="$1" output rc
    if [ "$kind" = node ]; then
        output=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID" --format='value(state)' 2>&1)
    else
        output=$(gcloud alpha compute tpus queued-resources describe "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID" --format='value(state)' 2>&1)
    fi
    rc=$?
    if [ "$rc" -eq 0 ]; then
        printf '%s' "$output"
        return 0
    fi
    if [[ "$output" == *NOT_FOUND* ]]; then
        return 0
    fi
    echo "ERROR: failed to inspect $kind for $TPU_NAME: $output" >&2
    return 1
}

for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
    if ! node_state=$(describe_resource node); then
        sleep "$WAIT_SECONDS"
        continue
    fi
    if ! queued_state=$(describe_resource queued-resource); then
        sleep "$WAIT_SECONDS"
        continue
    fi

    if [ -z "$node_state" ] && [ -z "$queued_state" ]; then
        echo "Verified absent: TPU=$TPU_NAME queued-resource=$TPU_NAME"
        exit 0
    fi

    echo "Release attempt $attempt/$ATTEMPTS: node=${node_state:-absent} queued=${queued_state:-absent}"
    if [ -n "$queued_state" ]; then
        # Cancel the queue itself, including a racing provisioning attempt.
        # Deleting its VM alone lets PROVISIONING recreate it indefinitely.
        timeout "$GCLOUD_DELETE_TIMEOUT_SECONDS" \
            gcloud alpha compute tpus queued-resources delete "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID" --force --quiet || true
    elif [ -n "$node_state" ]; then
        # Standalone V2 TPU VMs have no queued-resource to cancel.
        timeout "$GCLOUD_DELETE_TIMEOUT_SECONDS" \
            gcloud compute tpus tpu-vm delete "$TPU_NAME" \
            --zone="$ZONE" --project="$PROJECT_ID" --quiet || true
    fi
    sleep "$WAIT_SECONDS"
done

node_state=$(describe_resource node 2>/dev/null || echo unknown)
queued_state=$(describe_resource queued-resource 2>/dev/null || echo unknown)
echo "ERROR: release unverified after $ATTEMPTS attempts: node=${node_state:-absent} queued=${queued_state:-absent}" >&2
exit 1
