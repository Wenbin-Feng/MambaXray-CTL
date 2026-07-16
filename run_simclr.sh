#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch/common.sh"
require_path_vars IU_IMAGES STAGE2_VISION_MODEL
SIMCLR_CONFIG="${SIMCLR_CONFIG:-$PROJECT_ROOT/config.yaml}"
cd "$PROJECT_ROOT"
exec "$PYTHON" -u main_simclr.py \
  --config "$SIMCLR_CONFIG" \
  --data_dir "$IU_IMAGES" \
  --pretrained_checkpoint "$STAGE2_VISION_MODEL" \
  --batch_size 60 \
  --dataset hippo \
  -e 100 \
  "$@"
