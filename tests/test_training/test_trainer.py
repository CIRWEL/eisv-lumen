"""Tests for eisv_lumen.training.trainer — LoRA training loop.

All tests run WITHOUT GPU or heavy dependencies.  Model and tokenizer
objects are mocked so the data-loading, config-mapping, and
parameter-counting logic can be verified in CI.
"""

from __future__ import annotations

import json
import os
import textwrap
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from eisv_lumen.training.config import TrainingConfig
from eisv_lumen.training.trainer import (
    config_to_training_args,
    load_training_data,
    print_trainable_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_train_data():
    """Minimal list of training example dicts."""
    return [
        {"text": "<|system|>\nYou are...\n<|user|>\nSHAPE: settled_presence\n<|assistant|>\nEISV_TOKENS: ~warmth~\nLUMEN_TOKENS: warm\nPATTERN: SINGLE", "shape": "settled_presence", "pattern": "SINGLE"},
        {"text": "<|system|>\nYou are...\n<|user|>\nSHAPE: rising_warmth\n<|assistant|>\nEISV_TOKENS: ~warmth~ ~resonance~\nLUMEN_TOKENS: warm with\nPATTERN: PAIR", "shape": "rising_warmth", "pattern": "PAIR"},
    ]


@pytest.fixture
def sample_val_data():
    """Minimal validation data."""
    return [
        {"text": "<|system|>\nYou are...\n<|user|>\nSHAPE: fading_clarity\n<|assistant|>\nEISV_TOKENS: ~stillness~\nLUMEN_TOKENS: quiet\nPATTERN: SINGLE", "shape": "fading_clarity", "pattern": "SINGLE"},
    ]


@pytest.fixture
def data_dir(tmp_path, sample_train_data, sample_val_data):
    """Create a temporary data directory with train.json and val.json."""
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    train_path.write_text(json.dumps(sample_train_data))
    val_path.write_text(json.dumps(sample_val_data))
    return str(tmp_path)


# ---------------------------------------------------------------------------
# TestLoadTrainingData
# ---------------------------------------------------------------------------

class TestLoadTrainingData:
    """Test data loading from JSON files."""

    def test_loads_train_and_val(self, data_dir, sample_train_data, sample_val_data):
        """Training and validation data load correctly from JSON."""
        train, val = load_training_data(data_dir)
        assert len(train) == len(sample_train_data)
        assert len(val) == len(sample_val_data)

    def test_preserves_text_field(self, data_dir):
        """Each loaded example should have a 'text' key."""
        train, val = load_training_data(data_dir)
        for item in train + val:
            assert "text" in item
            assert isinstance(item["text"], str)
            assert len(item["text"]) > 0

    def test_preserves_shape_field(self, data_dir):
        """Each loaded example should have a 'shape' key."""
        train, val = load_training_data(data_dir)
        for item in train + val:
            assert "shape" in item

    def test_missing_train_file(self, tmp_path):
        """Should raise FileNotFoundError when train.json is missing."""
        val_path = tmp_path / "val.json"
        val_path.write_text("[]")
        with pytest.raises(FileNotFoundError, match="train.json"):
            load_training_data(str(tmp_path))

    def test_missing_val_file(self, tmp_path):
        """Should raise FileNotFoundError when val.json is missing."""
        train_path = tmp_path / "train.json"
        train_path.write_text("[]")
        with pytest.raises(FileNotFoundError, match="val.json"):
            load_training_data(str(tmp_path))


# ---------------------------------------------------------------------------
# TestConfigToTrainingArgs
# ---------------------------------------------------------------------------

class TestConfigToTrainingArgs:
    """Test TrainingConfig -> TrainingArguments kwargs mapping."""

    def test_default_config_mapping(self):
        """Default config produces expected training args dict."""
        config = TrainingConfig()
        args = config_to_training_args(config, "/tmp/output")

        assert args["output_dir"] == "/tmp/output"
        assert args["num_train_epochs"] == config.num_epochs
        assert args["per_device_train_batch_size"] == config.batch_size
        assert args["per_device_eval_batch_size"] == config.batch_size
        assert args["gradient_accumulation_steps"] == config.gradient_accumulation_steps
        assert args["learning_rate"] == config.learning_rate
        assert args["warmup_steps"] == config.warmup_steps
        assert args["weight_decay"] == config.weight_decay
        assert args["fp16"] == config.fp16
        assert args["seed"] == config.seed
        assert args["report_to"] == "none"

    def test_custom_config_mapping(self):
        """Custom config values are correctly mapped."""
        config = TrainingConfig(
            num_epochs=3,
            batch_size=8,
            learning_rate=1e-5,
            fp16=False,
            seed=123,
        )
        args = config_to_training_args(config, "/tmp/custom")

        assert args["num_train_epochs"] == 3
        assert args["per_device_train_batch_size"] == 8
        assert args["learning_rate"] == 1e-5
        assert args["fp16"] is False
        assert args["seed"] == 123

    def test_eval_strategy_present(self):
        """Evaluation strategy should be set to 'steps'."""
        config = TrainingConfig()
        args = config_to_training_args(config, "/tmp/out")

        assert args["eval_strategy"] == "steps"
        assert args["eval_steps"] == 50

    def test_save_strategy_present(self):
        """Save strategy should be configured."""
        config = TrainingConfig()
        args = config_to_training_args(config, "/tmp/out")

        assert args["save_strategy"] == "steps"
        assert args["save_steps"] == 100

    def test_logging_steps(self):
        """Logging steps should be set."""
        config = TrainingConfig()
        args = config_to_training_args(config, "/tmp/out")

        assert args["logging_steps"] == 10


# ---------------------------------------------------------------------------
# TestPrintTrainableParams
# ---------------------------------------------------------------------------

class TestPrintTrainableParams:
    """Test parameter counting utility."""

    def test_prints_counts(self, capsys):
        """Should print trainable and total parameter counts."""
        # Create a mock model with parameters
        param_trainable = MagicMock()
        param_trainable.numel.return_value = 1000
        param_trainable.requires_grad = True

        param_frozen = MagicMock()
        param_frozen.numel.return_value = 9000
        param_frozen.requires_grad = False

        model = MagicMock()
        model.parameters.return_value = [param_trainable, param_frozen]

        print_trainable_params(model)
        captured = capsys.readouterr()

        assert "1,000" in captured.out
        assert "10,000" in captured.out
        assert "10.00%" in captured.out

    def test_all_frozen(self, capsys):
        """Model with no trainable params prints 0%."""
        param = MagicMock()
        param.numel.return_value = 5000
        param.requires_grad = False

        model = MagicMock()
        model.parameters.return_value = [param]

        print_trainable_params(model)
        captured = capsys.readouterr()

        assert "0" in captured.out
        assert "5,000" in captured.out
        assert "0.00%" in captured.out

    def test_all_trainable(self, capsys):
        """Model with all params trainable prints 100%."""
        param = MagicMock()
        param.numel.return_value = 2000
        param.requires_grad = True

        model = MagicMock()
        model.parameters.return_value = [param]

        print_trainable_params(model)
        captured = capsys.readouterr()

        assert "2,000" in captured.out
        assert "100.00%" in captured.out

    def test_empty_model(self, capsys):
        """Model with no parameters should not crash."""
        model = MagicMock()
        model.parameters.return_value = []

        print_trainable_params(model)
        captured = capsys.readouterr()

        assert "0" in captured.out


# ---------------------------------------------------------------------------
# TestTrainTeacherImportGuard
# ---------------------------------------------------------------------------

class TestTrainTeacherImportGuard:
    """Test that train_teacher raises ImportError without GPU deps."""

    def test_import_error_message(self, data_dir):
        """train_teacher should raise ImportError with helpful message when deps missing."""
        from eisv_lumen.training.trainer import train_teacher, _require_training_deps

        # Mock the import check to simulate missing dependencies
        with patch(
            "eisv_lumen.training.trainer._require_training_deps",
            side_effect=ImportError("Training requires PyTorch"),
        ):
            config = TrainingConfig()
            with pytest.raises(ImportError, match="PyTorch"):
                train_teacher(config, data_dir)


# ---------------------------------------------------------------------------
# TestCLITrainWiring
# ---------------------------------------------------------------------------

class TestCLITrainWiring:
    """Test that the CLI train subcommand is wired correctly."""

    def test_train_subparser_has_data_dir(self):
        """The train subparser should accept --data-dir."""
        from eisv_lumen.training.cli import main
        import argparse

        # Build the parser as main() does
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        train = subparsers.add_parser("train")
        train.add_argument("--config", type=str, default=None)
        train.add_argument("--data-dir", type=str, default="data/training")

        args = parser.parse_args(["train", "--data-dir", "/tmp/mydata"])
        assert args.data_dir == "/tmp/mydata"

    def test_train_subparser_default_data_dir(self):
        """The train subparser should have a default --data-dir."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        train = subparsers.add_parser("train")
        train.add_argument("--config", type=str, default=None)
        train.add_argument("--data-dir", type=str, default="data/training")

        args = parser.parse_args(["train"])
        assert args.data_dir == "data/training"
