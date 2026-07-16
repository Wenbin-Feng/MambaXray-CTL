#!/usr/bin/env python3
"""Read-only environment and configuration check for MambaXray-CTL."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import parser as training_parser  # noqa: E402


REQUIRED_MODULES = (
    "torch",
    "pytorch_lightning",
    "transformers",
    "timm",
    "mamba_ssm",
    "peft",
    "sklearn",
    "pydantic",
    "yaml",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/profiles/iu_xray_stage3.yaml"),
    )
    args = parser.parse_args()

    missing_modules = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing_modules:
        print("Missing Python modules: " + ", ".join(missing_modules))
        return 1

    runtime = training_parser.parse_args(["--config", args.config])
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Profile: {args.config}")
    print(f"Dataset: {runtime.dataset}")
    print(f"Annotation: {runtime.annotation}")
    print(f"Images: {runtime.base_dir}")
    print(f"Vision checkpoint: {runtime.vision_model}")
    print(f"Devices: {runtime.devices}")
    print("Environment and configured paths are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
