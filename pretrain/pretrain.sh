#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/launch/common.sh"
require_path_vars IU_IMAGES
cd "$PROJECT_ROOT"
exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  --master_port=4399 \
  -m pretrain.main_pretrain \
  --batch_size 12 \
  --input_size 192 \
  --model arm_large_pz16 \
  --norm_pix_loss \
  --epochs 101 \
  --warmup_epochs 5 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --data_path "$IU_IMAGES" \
  --output_dir pretrain/outputs/test/ \
  --log_dir pretrain/outputs/test/ \
  "$@"
