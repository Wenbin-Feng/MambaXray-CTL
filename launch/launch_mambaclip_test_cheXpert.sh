#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint.pth> [extra options...]" >&2
  exit 2
fi
checkpoint="$1"
shift
launch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$launch_dir/evaluate.sh" "$launch_dir/../configs/profiles/chexpert_plus_stage3.yaml" "$checkpoint" "$@"
