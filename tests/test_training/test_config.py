"""Tests for eisv_lumen.training.config — LoRA training configuration."""

from __future__ import annotations

import os

import pytest

from eisv_lumen.training.config import TrainingConfig, load_config


# ---------------------------------------------------------------------------
# TestTrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        """Default config should have sensible values."""
        config = TrainingConfig()
        assert config.model_name == "Qwen/Qwen3-4B"
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.learning_rate == 2e-4
        assert config.num_epochs == 5
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 4
        assert config.warmup_steps == 100
        assert config.max_seq_length == 512
        assert config.weight_decay == 0.01
        assert config.fp16 is True
        assert config.seed == 42
        assert config.output_dir == "outputs/teacher_lora"

    def test_lora_target_modules(self):
        """Default LoRA target modules should cover attention + MLP projections."""
        config = TrainingConfig()
        assert config.lora_target_modules == [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = TrainingConfig(
            model_name="custom/model",
            lora_rank=8,
            num_epochs=10,
            seed=99,
        )
        assert config.model_name == "custom/model"
        assert config.lora_rank == 8
        assert config.num_epochs == 10
        assert config.seed == 99
        # Unchanged defaults
        assert config.lora_alpha == 32
        assert config.batch_size == 4


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_default(self):
        """load_config with no path should return default TrainingConfig."""
        config = load_config()
        assert isinstance(config, TrainingConfig)
        assert config.model_name == "Qwen/Qwen3-4B"
        assert config.lora_rank == 16

    def test_load_from_yaml(self, tmp_path):
        """load_config should override defaults from a YAML file."""
        yaml_content = """\
model_name: "custom/test-model"
lora_rank: 8
num_epochs: 3
seed: 123
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        config = load_config(str(yaml_file))
        assert config.model_name == "custom/test-model"
        assert config.lora_rank == 8
        assert config.num_epochs == 3
        assert config.seed == 123
        # Defaults for unspecified fields
        assert config.lora_alpha == 32
        assert config.batch_size == 4
        assert config.fp16 is True
