#!/bin/bash

REMOTE_PROJECTS_DIR=/home/lishengping/projects

# ================= 参数 =================
tpu_suffix=$1
direction=$2
file_name=$3

echo "file_name: $file_name"
echo "direction: $direction"
echo "tpu_name: llm-jax-$tpu_suffix"

# ================= 参数检查 =================
if [ $# -ne 3 ]; then
  echo "Usage: $0 <tpu_suffix> <direction:0|1> <file_path>"
  exit 1
fi

if [ ! -e "$file_name" ] && [ "$direction" -eq 0 ]; then
  echo "Error: File does not exist: $file_name"
  exit 1
fi

# ================= 绝对路径 =================
abs_path=$(readlink -f "$file_name")
echo "Abs path: $abs_path"

# ================= 路径匹配 =================
names=("paxml" "praxis" "mesh_easy_jax" "DCFormer" "maxtext")

IFS='/' read -ra path_parts <<< "$abs_path"
matched_paths=()

for name in "${names[@]}"; do
    for ((i=0; i<${#path_parts[@]}; i++)); do
        if [[ "${path_parts[i]}" == "$name" ]]; then
            matched_paths=()
            for ((j=i; j<${#path_parts[@]}; j++)); do
                matched_paths+=("${path_parts[j]}")
            done
            break 2
        fi
    done
done

if [ ${#matched_paths[@]} -gt 0 ]; then
    MATCH_PATH=$(echo "${matched_paths[*]}" | tr ' ' '/')
else
    echo "Warning: No project root matched, using filename only"
    MATCH_PATH=$(basename "$file_name")
fi

echo "Matched path: $MATCH_PATH"

# ================= TPU 类型判断 =================
if [[ $tpu_suffix == *7x* ]]; then
  zone="us-central1-c"
  tpu_type="v7x"

elif [[ $tpu_suffix == *v3* ]]; then
  zone="us-central1-a"
  tpu_type="v3"

elif [[ $tpu_suffix == *v4* ]]; then
  zone="us-central2-b"
  tpu_type="v4"

elif [[ $tpu_suffix == *v6e* ]]; then
  zone="us-east5-a"
  tpu_type="v6e"

else
  zone="us-central1-a"
  tpu_type="v5p"
fi

echo "TPU type: $tpu_type"
echo "Zone: $zone"

# ================= 远程路径 =================
remote_path=$REMOTE_PROJECTS_DIR/$MATCH_PATH
echo "Remote path: $remote_path"

VM_NAME="llm-jax-${tpu_suffix}"
PROJECT_ID="newproject-1-451205"

# ================= 文件传输 =================
if [ "$direction" -eq 0 ]; then
  # 本地 -> 远程

  if [[ "$tpu_type" == "v7x" ]]; then
    echo "[Using compute scp for v7x]"
    gcloud compute scp "$file_name" "${VM_NAME}:$remote_path" \
      --zone "$zone" --project="$PROJECT_ID"
  else
    echo "[Using tpu-vm scp]"
    gcloud compute tpus tpu-vm scp "$file_name" "${VM_NAME}:$remote_path" \
      --worker all --zone "$zone" --project="$PROJECT_ID"
  fi

  echo "Upload completed → $remote_path"

elif [ "$direction" -eq 1 ]; then
  # 远程 -> 本地

  if [[ "$tpu_type" == "v7x" ]]; then
    echo "[Using compute scp for v7x]"
    gcloud compute scp "${VM_NAME}:$remote_path" ./ \
      --zone "$zone" --project="$PROJECT_ID"
  else
    echo "[Using tpu-vm scp]"
    gcloud compute tpus tpu-vm scp "${VM_NAME}:$remote_path" ./ \
      --worker all --zone "$zone"
  fi

  echo "Download completed → ./"

else
  echo "Error: direction must be 0 (upload) or 1 (download)"
  exit 1
fi
