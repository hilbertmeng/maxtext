#!/usr/bin/env bash
# Publish one complete immutable source archive, suitable for fresh TPU gsutil.
set -euo pipefail
repo=${1:?repository}
revision=${2:?commit}
destination=${3:?gs://destination.tar.gz}
gcloud=${GCLOUD_BIN:-/home/xd/google-cloud-sdk/bin/gcloud}
commit=$(git -C "$repo" rev-parse "$revision^{commit}")
archive_dir=$(mktemp -d /tmp/xd-diag-source.XXXXXX)
archive="$archive_dir/$commit.tar.gz"
trap 'rm -f "$archive"; rmdir "$archive_dir"' EXIT
git -C "$repo" archive --format=tar.gz --output="$archive" "$commit"
gzip -t "$archive"
sha256sum "$archive"
env CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED=False \
  "$gcloud" storage cp "$archive" "$destination"
echo "SOURCE_READY commit=$commit uri=$destination"
