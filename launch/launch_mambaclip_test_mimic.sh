#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint.pth> [extra options...]" >&2
  exit 2
fi
checkpoint="$1"
shift
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_path_vars MIMIC_ANNOTATION MIMIC_IMAGES
cd "$PROJECT_ROOT"
exec "$PYTHON" -u train_downstream.py \
  --test \
  --dataset mimic_cxr \
  --annotation "$MIMIC_ANNOTATION" \
  --base_dir "$MIMIC_IMAGES" \
  --vision_model "$checkpoint" \
  --delta_file "$checkpoint" \
  --savedmodel_path outputs/mimic_cxr/test \
  "${common_model_args[@]}" \
  "$@"
