#!/usr/bin/env bash

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$LAUNCH_DIR/.." && pwd)"
PATHS_FILE="${MAMBA_XRAY_PATHS_FILE:-$LAUNCH_DIR/paths.env}"
CALLER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

if [[ ! -f "$PATHS_FILE" ]]; then
  echo "Path configuration not found: $PATHS_FILE" >&2
  echo "Copy .env.example to launch/paths.env and fill in the paths." >&2
  exit 2
fi

set -a
source "$PATHS_FILE"
set +a

required_path_vars=(PYTHON QWEN_MODEL LLAMA_MODEL BIO_CLINICALBERT IMAGE_PROCESSOR)
for variable in "${required_path_vars[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing $variable in $PATHS_FILE" >&2
    exit 2
  fi
done

if [[ -n "$CALLER_CUDA_VISIBLE_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$CALLER_CUDA_VISIBLE_DEVICES"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

common_model_args=(
  --qwen_model_path "$QWEN_MODEL"
  --llama_model_path "$LLAMA_MODEL"
  --bio_clinicalbert_path "$BIO_CLINICALBERT"
  --image_processor_path "$IMAGE_PROCESSOR"
)

require_path_vars() {
  local variable
  for variable in "$@"; do
    if [[ -z "${!variable:-}" ]]; then
      echo "Missing $variable in $PATHS_FILE" >&2
      exit 2
    fi
  done
}
