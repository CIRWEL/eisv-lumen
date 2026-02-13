"""LoRA training configuration for the teacher model.

Provides a dataclass-based configuration with sensible defaults for
fine-tuning Qwen3-4B on EISV trajectory-expression data.
Supports loading overrides from YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TrainingConfig:
    """Configuration for LoRA teacher model training.

    All fields have sensible defaults for EISV trajectory-expression
    fine-tuning on Qwen3-4B.
    """

    model_name: str = "Qwen/Qwen3-4B"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_seq_length: int = 512
    weight_decay: float = 0.01
    fp16: bool = True
    seed: int = 42
    output_dir: str = "outputs/teacher_lora"


def load_config(yaml_path: Optional[str] = None) -> TrainingConfig:
    """Load a TrainingConfig, optionally overriding from a YAML file.

    Parameters
    ----------
    yaml_path:
        Path to a YAML configuration file. If *None*, returns the
        default configuration. Any fields present in the YAML override
        the defaults; unspecified fields keep their default values.

    Returns
    -------
    A :class:`TrainingConfig` instance.
    """
    if yaml_path is None:
        return TrainingConfig()

    with open(yaml_path, "r") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    return TrainingConfig(**data)
