#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <profile.yaml> <checkpoint.pth> [extra options...]" >&2
  exit 2
fi

PROFILE="$1"
CHECKPOINT="$2"
shift 2
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ "$PROFILE" == *chexpert* ]]; then
  require_path_vars CHEXPERT_ANNOTATION CHEXPERT_IMAGES
  annotation="$CHEXPERT_ANNOTATION"
  base_dir="$CHEXPERT_IMAGES"
elif [[ "$PROFILE" == *mimic* ]]; then
  require_path_vars MIMIC_ANNOTATION MIMIC_IMAGES
  annotation="$MIMIC_ANNOTATION"
  base_dir="$MIMIC_IMAGES"
else
  require_path_vars IU_ANNOTATION IU_IMAGES
  annotation="$IU_ANNOTATION"
  base_dir="$IU_IMAGES"
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" -u train_downstream.py \
  --config "$PROFILE" \
  --test \
  --devices 1 \
  --strategy auto \
  --annotation "$annotation" \
  --base_dir "$base_dir" \
  --vision_model "$CHECKPOINT" \
  --delta_file "$CHECKPOINT" \
  "${common_model_args[@]}" \
  "$@"
