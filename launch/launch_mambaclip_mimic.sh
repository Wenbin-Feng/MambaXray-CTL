#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_path_vars MIMIC_ANNOTATION MIMIC_IMAGES STAGE3_VISION_MODEL
cd "$PROJECT_ROOT"
exec "$PYTHON" -u train_downstream.py \
  --dataset mimic_cxr \
  --annotation "$MIMIC_ANNOTATION" \
  --base_dir "$MIMIC_IMAGES" \
  --vision_model "$STAGE3_VISION_MODEL" \
  --savedmodel_path outputs/mimic_cxr/stage3 \
  "${common_model_args[@]}" \
  "$@"
