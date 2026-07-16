"""CLI-compatible, validated configuration for MambaXray-CTL.

The original project exposed a module-level ``argparse`` parser.  This module
keeps that public API so existing launch commands continue to work, while a
Pydantic model validates the parsed values.  A YAML profile can be supplied
with ``--config`` and every value can still be overridden on the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got: {value!r}")


class RuntimeConfig(BaseModel):
    """Validated runtime settings shared by training, validation, and testing."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    config: str | None = None
    seed: int = 42
    validate_paths: bool = True

    detection: bool = False
    test: bool = False
    validate_only: bool = Field(default=False, validation_alias="validate", serialization_alias="validate")
    dataset: Literal["iu_xray", "mimic_cxr", "chinese"] = "iu_xray"
    annotation: str
    base_dir: str
    batch_size: int = Field(default=6, gt=0)
    val_batch_size: int = Field(default=16, gt=0)
    test_batch_size: int = Field(default=16, gt=0)
    prefetch_factor: int = Field(default=4, gt=0)
    num_workers: int = Field(default=8, ge=0)
    input_size: int = Field(default=224, gt=0)

    type: Literal["base", "large"] = "large"
    text_encoder_type: str = "Bio_ClinicalBERT"
    projection_dim: int = Field(default=2048, gt=0)
    vision_model: str = "None"
    llama_model: str | None = None
    qwen_model_path: str
    llama_model_path: str
    bio_clinicalbert_path: str
    image_processor_path: str
    deepseek_model_path: str
    freeze_vm: bool = True
    llm_use_lora: bool = False
    llm_r: int = Field(default=16, gt=0)
    llm_alpha: int = Field(default=16, gt=0)
    vis_use_lora: bool = False
    vis_r: int = Field(default=16, gt=0)
    vis_alpha: int = Field(default=16, gt=0)
    lora_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    global_only: bool = False
    low_resource: bool = False
    end_sym: str = "</s>"

    contrastive_loss_weight: float = Field(default=0.4, ge=0.0)
    contrastive_temperature: float = Field(default=0.07, gt=0.0)

    savedmodel_path: str
    epoch_save_ckpt: int = Field(default=3, ge=0)
    ckpt_file: str | None = None
    delta_file: str | None = None
    weights: list[float] = Field(default_factory=lambda: [0.8, 0.2])
    scorer_types: list[str] = Field(default_factory=lambda: ["Bleu_4", "CIDEr"])

    optimizer: str = "adam"
    weight_decay: float = Field(default=0.001, ge=0.0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    gradient_clip_val: float | None = Field(default=None, gt=0.0)

    beam_size: int = Field(default=3, gt=0)
    do_sample: bool = True
    no_repeat_ngram_size: int = Field(default=2, ge=0)
    num_beam_groups: int = Field(default=1, gt=0)
    min_new_tokens: int = Field(default=80, ge=0)
    max_new_tokens: int = Field(default=120, gt=0)
    max_length: int = Field(default=100, gt=0)
    repetition_penalty: float = Field(default=2.0, gt=0.0)
    length_penalty: float = 2.0
    diversity_penalty: float = Field(default=0.0, ge=0.0)
    temperature: float = Field(default=0.7, gt=0.0)

    devices: int = Field(default=1, gt=0)
    num_nodes: int = Field(default=1, gt=0)
    accelerator: Literal["cpu", "gpu", "tpu", "ipu", "hpu", "mps"] = "gpu"
    strategy: str = "deepspeed"
    precision: str = "bf16-mixed"
    limit_val_batches: float = Field(default=1.0, gt=0.0)
    limit_test_batches: float = Field(default=1.0, gt=0.0)
    limit_train_batches: float = Field(default=1.0, gt=0.0)
    max_epochs: int = Field(default=3, gt=0)
    every_n_train_steps: int = Field(default=0, ge=0)
    val_check_interval: float = Field(default=1.0, gt=0.0)
    accumulate_grad_batches: int = Field(default=1, gt=0)
    num_sanity_val_steps: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_consistency(self) -> "RuntimeConfig":
        if self.test and self.validate_only:
            raise ValueError("--test and --validate are mutually exclusive")
        if len(self.weights) != len(self.scorer_types):
            raise ValueError("weights and scorer_types must contain the same number of values")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")
        return self

    def missing_paths(self) -> list[str]:
        if not self.validate_paths:
            return []
        candidates: dict[str, str | None] = {
            "annotation": self.annotation,
            "base_dir": self.base_dir,
            "qwen_model_path": self.qwen_model_path if self.dataset == "iu_xray" else None,
            "llama_model_path": self.llama_model_path if self.dataset != "iu_xray" else None,
            "bio_clinicalbert_path": self.bio_clinicalbert_path,
            "image_processor_path": self.image_processor_path,
        }
        if self.vision_model not in {"", "None", None}:
            candidates["vision_model"] = self.vision_model
        if self.delta_file:
            candidates["delta_file"] = self.delta_file
        return [
            f"{name}={value or '<not provided>'}"
            for name, value in candidates.items()
            if value is not None and (not value or not Path(value).exists())
        ]


class ValidatedArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args: list[str] | None = None, namespace: argparse.Namespace | None = None):
        raw_args = args
        profile_parser = argparse.ArgumentParser(add_help=False)
        profile_parser.add_argument("--config")
        profile_args, _ = profile_parser.parse_known_args(raw_args)
        if profile_args.config:
            profile_path = Path(profile_args.config).expanduser()
            if not profile_path.is_file():
                self.error(f"configuration file does not exist: {profile_path}")
            profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
            if not isinstance(profile, dict):
                self.error("configuration file must contain a YAML mapping")
            known = {action.dest for action in self._actions}
            unknown = sorted(set(profile) - known)
            if unknown:
                self.error(f"unknown configuration keys: {', '.join(unknown)}")
            self.set_defaults(**profile)

        parsed = super().parse_args(raw_args, namespace)
        try:
            validated = RuntimeConfig.model_validate(vars(parsed))
        except ValidationError as exc:
            self.error(str(exc))
        missing = validated.missing_paths()
        if missing:
            self.error("missing required paths:\n  " + "\n  ".join(missing))
        return argparse.Namespace(**validated.model_dump(by_alias=True))


parser = ValidatedArgumentParser(description="MambaXray-CTL training and evaluation")
parser.add_argument("--config", help="YAML profile; explicit CLI options override profile values")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validate_paths", type=str_to_bool, default=True)
parser.add_argument("--detection", type=str_to_bool, default=False)
parser.add_argument("--test", action="store_true")
parser.add_argument("--validate", action="store_true")
parser.add_argument("--dataset", choices=["iu_xray", "mimic_cxr", "chinese"], default="iu_xray")
parser.add_argument("--annotation", default="")
parser.add_argument("--base_dir", default="")
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--val_batch_size", type=int, default=16)
parser.add_argument("--test_batch_size", type=int, default=16)
parser.add_argument("--prefetch_factor", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--input-size", dest="input_size", type=int, default=224)
parser.add_argument("--type", choices=["base", "large"], default="large")
parser.add_argument("--text_encoder_type", default="Bio_ClinicalBERT")
parser.add_argument("--projection_dim", type=int, default=2048)
parser.add_argument("--vision_model", default="None")
parser.add_argument("--llama_model")
parser.add_argument("--qwen_model_path", default="")
parser.add_argument("--llama_model_path", default="")
parser.add_argument("--bio_clinicalbert_path", default="")
parser.add_argument("--image_processor_path", default="")
parser.add_argument("--deepseek_model_path", default="")
parser.add_argument("--freeze_vm", type=str_to_bool, default=True)
parser.add_argument("--llm_use_lora", type=str_to_bool, default=False)
parser.add_argument("--llm_r", type=int, default=16)
parser.add_argument("--llm_alpha", type=int, default=16)
parser.add_argument("--vis_use_lora", type=str_to_bool, default=False)
parser.add_argument("--vis_r", type=int, default=16)
parser.add_argument("--vis_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--global_only", type=str_to_bool, default=False)
parser.add_argument("--low_resource", type=str_to_bool, default=False)
parser.add_argument("--end_sym", default="</s>")
parser.add_argument("--contrastive_loss_weight", type=float, default=0.4)
parser.add_argument("--contrastive_temperature", type=float, default=0.07)
parser.add_argument("--savedmodel_path", default=str(PROJECT_ROOT / "outputs/default"))
parser.add_argument("--epoch_save_ckpt", type=int, default=3)
parser.add_argument("--ckpt_file")
parser.add_argument("--delta_file")
parser.add_argument("--weights", type=float, nargs="+", default=[0.8, 0.2])
parser.add_argument("--scorer_types", nargs="+", default=["Bleu_4", "CIDEr"])
parser.add_argument("--optimizer", default="adam")
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--gradient_clip_val", type=float)
parser.add_argument("--beam_size", type=int, default=3)
parser.add_argument("--do_sample", type=str_to_bool, default=True)
parser.add_argument("--no_repeat_ngram_size", type=int, default=2)
parser.add_argument("--num_beam_groups", type=int, default=1)
parser.add_argument("--min_new_tokens", type=int, default=80)
parser.add_argument("--max_new_tokens", type=int, default=120)
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--repetition_penalty", type=float, default=2.0)
parser.add_argument("--length_penalty", type=float, default=2.0)
parser.add_argument("--diversity_penalty", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--accelerator", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"], default="gpu")
parser.add_argument("--strategy", default="deepspeed")
parser.add_argument("--precision", default="bf16-mixed")
parser.add_argument("--limit_val_batches", type=float, default=1.0)
parser.add_argument("--limit_test_batches", type=float, default=1.0)
parser.add_argument("--limit_train_batches", type=float, default=1.0)
parser.add_argument("--max_epochs", type=int, default=3)
parser.add_argument("--every_n_train_steps", type=int, default=0)
parser.add_argument("--val_check_interval", type=float, default=1.0)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--num_sanity_val_steps", type=int, default=2)
