"""LoRA fine-tuning trainer for the EISV-Lumen teacher model.

Wraps Hugging Face Transformers + PEFT to fine-tune a causal LM (default: Qwen3-4B)
with LoRA on trajectory-expression training data.  All heavy imports
(torch, transformers, peft) are deferred so that non-GPU environments can
still import the module for testing.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, TYPE_CHECKING

from eisv_lumen.training.config import TrainingConfig

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

_MISSING_DEPS_MSG = (
    "Training requires PyTorch, Transformers, and PEFT.\n"
    "Install them with:\n"
    "  pip install torch transformers peft datasets\n"
)


def _require_training_deps():
    """Import and return heavy training dependencies, or raise with a message."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(_MISSING_DEPS_MSG) from exc

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "TaskType": TaskType,
        "Dataset": Dataset,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_training_data(data_dir: str):
    """Load train.json and val.json from *data_dir*.

    Returns
    -------
    Tuple of (train_data, val_data) where each is a list of dicts
    with at least a ``text`` key.
    """
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "val.json")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    with open(train_path, "r") as f:
        train_data: List[Dict[str, Any]] = json.load(f)
    with open(val_path, "r") as f:
        val_data: List[Dict[str, Any]] = json.load(f)

    return train_data, val_data


# ---------------------------------------------------------------------------
# Config -> TrainingArguments mapping
# ---------------------------------------------------------------------------


def config_to_training_args(config: TrainingConfig, output_dir: str):
    """Convert a :class:`TrainingConfig` to HuggingFace TrainingArguments kwargs.

    Returns a plain dict so tests can inspect the mapping without importing
    transformers.
    """
    return {
        "output_dir": output_dir,
        "num_train_epochs": config.num_epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "fp16": config.fp16,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 100,
        "seed": config.seed,
        "report_to": "none",
    }


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def print_trainable_params(model) -> None:
    """Print the number of trainable vs total parameters."""
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({pct:.2f}% trainable)"
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train_teacher(config: TrainingConfig, data_dir: str) -> str:
    """Fine-tune a causal LM (default: Qwen3-4B) with LoRA.

    Parameters
    ----------
    config:
        :class:`TrainingConfig` with all hyperparameters.
    data_dir:
        Path to directory containing ``train.json`` and ``val.json``
        produced by the ``prepare`` CLI step.

    Returns
    -------
    Path to the saved LoRA adapter directory.
    """
    deps = _require_training_deps()
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]
    Dataset = deps["Dataset"]

    # 1. Load data ---------------------------------------------------------
    print(f"Loading training data from {data_dir} ...")
    train_data, val_data = load_training_data(data_dir)
    print(f"  Train examples: {len(train_data)}")
    print(f"  Val examples:   {len(val_data)}")

    # 2. Load base model ---------------------------------------------------
    # MPS (Apple Silicon) works best with float32; CUDA can use fp16
    is_mps = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    if is_mps:
        dtype = torch.float32
        print(f"Loading base model: {config.model_name} (dtype=float32, device=mps) ...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
        )
        model = model.to("mps")
    else:
        dtype = torch.float16 if config.fp16 else torch.float32
        print(f"Loading base model: {config.model_name} (dtype={dtype}) ...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    # 3. Load tokenizer ----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Create LoRA config ------------------------------------------------
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Apply LoRA --------------------------------------------------------
    model = get_peft_model(model, lora_config)
    print_trainable_params(model)

    # 6. Tokenize ----------------------------------------------------------
    def _apply_chat_template_safe(messages: List[Dict[str, str]]) -> str:
        """Apply the tokenizer's chat template, falling back to generic format."""
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
        # Fallback: use the generic text format
        return None

    def tokenize_examples(examples: List[Dict[str, Any]]) -> Dict[str, list]:
        # Prefer the model's native chat template (ChatML for Qwen3)
        # over our generic <|role|> format for better training signal.
        texts = []
        for ex in examples:
            if "messages" in ex:
                templated = _apply_chat_template_safe(ex["messages"])
                texts.append(templated if templated else ex["text"])
            else:
                texts.append(ex["text"])

        encodings = tokenizer(
            texts,
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        # Labels are the same as input_ids for causal LM
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    print("Tokenizing training data ...")
    train_encodings = tokenize_examples(train_data)
    val_encodings = tokenize_examples(val_data)

    # 7. Build HuggingFace Datasets ----------------------------------------
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)

    # 8. Training arguments ------------------------------------------------
    output_dir = config.output_dir
    args_dict = config_to_training_args(config, output_dir)
    # MPS doesn't support fp16 training; disable it
    if is_mps:
        args_dict["fp16"] = False
        args_dict["use_mps_device"] = True
    training_args = TrainingArguments(**args_dict)

    # 9. Trainer -----------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # 10. Train! -----------------------------------------------------------
    print("Starting training ...")
    trainer.train()
    print("Training complete.")

    # 11. Save adapter -----------------------------------------------------
    adapter_path = os.path.join(output_dir, "final_adapter")
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to: {adapter_path}")

    return adapter_path
