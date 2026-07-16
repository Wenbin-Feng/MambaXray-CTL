#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require_path_vars IU_ANNOTATION IU_IMAGES STAGE2_VISION_MODEL
cd "$PROJECT_ROOT"
exec "$PYTHON" -u train_clip.py \
  --config configs/profiles/iu_xray_stage2.yaml \
  --annotation "$IU_ANNOTATION" \
  --base_dir "$IU_IMAGES" \
  --vision_model "$STAGE2_VISION_MODEL" \
  "${common_model_args[@]}" \
  "$@"
