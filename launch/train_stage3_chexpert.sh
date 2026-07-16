#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_path_vars CHEXPERT_ANNOTATION CHEXPERT_IMAGES STAGE3_VISION_MODEL
cd "$PROJECT_ROOT"
exec "$PYTHON" -u train_downstream.py \
  --config configs/profiles/chexpert_plus_stage3.yaml \
  --annotation "$CHEXPERT_ANNOTATION" \
  --base_dir "$CHEXPERT_IMAGES" \
  --vision_model "$STAGE3_VISION_MODEL" \
  "${common_model_args[@]}" \
  "$@"
