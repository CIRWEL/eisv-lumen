"""Inference utilities for the fine-tuned EISV-Lumen teacher model.

Provides functions to load a LoRA-adapted model, generate expressions
from trajectory inputs, and run evaluation on a test set.  Heavy imports
are deferred so non-GPU environments can still import the module.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from eisv_lumen.training.chat_format import SYSTEM_PROMPT
from eisv_lumen.training.teacher_train import parse_model_output, OutputParseResult
from eisv_lumen.training.teacher_eval import evaluate_predictions, EvalResults

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

_MISSING_DEPS_MSG = (
    "Inference requires PyTorch, Transformers, and PEFT.\n"
    "Install them with:\n"
    "  pip install torch transformers peft\n"
)


def _require_inference_deps():
    """Import and return heavy inference dependencies, or raise with a message."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(_MISSING_DEPS_MSG) from exc

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "PeftModel": PeftModel,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_teacher_model(
    adapter_path: str,
    base_model: str = "Qwen/Qwen3-4B",
):
    """Load the fine-tuned teacher model for inference.

    Loads the base model and applies the saved LoRA adapter on top.

    Parameters
    ----------
    adapter_path:
        Path to the saved LoRA adapter directory (output of training).
    base_model:
        HuggingFace model identifier for the base model.

    Returns
    -------
    Tuple of (model, tokenizer).
    """
    deps = _require_inference_deps()
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    PeftModel = deps["PeftModel"]

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Chat formatting helpers
# ---------------------------------------------------------------------------


def _format_inference_messages(trajectory_input: str) -> List[Dict[str, str]]:
    """Build chat messages for inference (system + user, no assistant)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": trajectory_input},
    ]


def _extract_assistant_response(full_text: str, prompt_text: str) -> str:
    """Extract only the assistant's generated response from full decoded text.

    Tries multiple strategies:
    1. Strip the prompt prefix from the full output.
    2. Look for an assistant role marker in the text.
    3. Fall back to the raw text if no markers found.
    """
    # Strategy 1: remove the prompt prefix
    if full_text.startswith(prompt_text):
        response = full_text[len(prompt_text):]
        return response.strip()

    # Strategy 2: look for common assistant markers (covers Qwen3, Llama, generic)
    for marker in (
        "<|im_start|>assistant",   # Qwen3 ChatML format
        "<|assistant|>",           # Generic chat format
        "<|start_header_id|>assistant<|end_header_id|>",  # Llama format
    ):
        if marker in full_text:
            parts = full_text.rsplit(marker, 1)
            if len(parts) == 2:
                # Strip any trailing end-of-turn tokens
                response = parts[1].strip()
                for end_token in ("<|im_end|>", "<|eot_id|>", "</s>"):
                    response = response.replace(end_token, "")
                return response.strip()

    # Strategy 3: fall back to raw text
    return full_text.strip()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_expression(
    model,
    tokenizer,
    trajectory_input: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate an expression from a trajectory input text.

    Parameters
    ----------
    model:
        The loaded (and possibly LoRA-adapted) model.
    tokenizer:
        The corresponding tokenizer.
    trajectory_input:
        Formatted trajectory text (from :func:`format_trajectory_input`).
    max_new_tokens:
        Maximum number of tokens to generate.

    Returns
    -------
    The generated text (assistant response only).
    """
    deps = _require_inference_deps()
    torch = deps["torch"]

    messages = _format_inference_messages(trajectory_input)

    # Apply chat template if the tokenizer supports it
    # Qwen3 supports enable_thinking — we disable it for direct structured output
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizers don't support enable_thinking kwarg
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    else:
        # Fallback: manual formatting
        parts = []
        for msg in messages:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        parts.append("<|assistant|>")
        prompt_text = "\n".join(parts)

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return _extract_assistant_response(full_text, prompt_text)


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------


def evaluate_on_test_set(
    adapter_path: str,
    test_data_path: str,
    base_model: str = "Qwen/Qwen3-4B",
    max_new_tokens: int = 64,
) -> EvalResults:
    """Run evaluation on a test set and return results.

    Loads the fine-tuned model, generates predictions for each test
    example, parses outputs, and evaluates with
    :func:`evaluate_predictions`.

    Parameters
    ----------
    adapter_path:
        Path to the saved LoRA adapter directory.
    test_data_path:
        Path to test.json file from the prepare step.
    base_model:
        HuggingFace model identifier for the base model.
    max_new_tokens:
        Maximum tokens per generation.

    Returns
    -------
    An :class:`EvalResults` instance.
    """
    # Load test data
    with open(test_data_path, "r") as f:
        test_data: List[Dict[str, Any]] = json.load(f)

    print(f"Evaluating on {len(test_data)} test examples ...")

    # Load model
    model, tokenizer = load_teacher_model(adapter_path, base_model)

    predictions: List[Dict[str, Any]] = []
    for i, example in enumerate(test_data):
        # Extract trajectory input from the text field
        # The text field has format: <|system|>\n...\n<|user|>\n...\n<|assistant|>\n...
        # We need the user part (the trajectory input)
        trajectory_input = _extract_trajectory_from_text(example["text"])
        shape = example["shape"]

        # Generate
        generated = generate_expression(
            model, tokenizer, trajectory_input, max_new_tokens=max_new_tokens
        )

        # Parse
        parsed = parse_model_output(generated)

        predictions.append({
            "shape": shape,
            "parsed": parsed,
            "expected_pattern": example.get("pattern"),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(test_data)}] generated ...")

    print(f"Evaluation complete. Computing metrics ...")
    return evaluate_predictions(predictions)


def _extract_trajectory_from_text(text: str) -> str:
    """Extract the user (trajectory) portion from a formatted text example.

    The text follows the format::

        <|system|>
        ...
        <|user|>
        SHAPE: ...
        ...
        <|assistant|>
        ...

    We extract everything between ``<|user|>`` and ``<|assistant|>``.
    """
    user_marker = "<|user|>"
    assistant_marker = "<|assistant|>"

    user_start = text.find(user_marker)
    assistant_start = text.find(assistant_marker)

    if user_start == -1:
        # No markers — return the whole text as-is
        return text

    start = user_start + len(user_marker)
    end = assistant_start if assistant_start != -1 else len(text)
    return text[start:end].strip()
