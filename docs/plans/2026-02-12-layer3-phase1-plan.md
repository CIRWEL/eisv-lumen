# Layer 3 Phase 1: Teacher Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fine-tune Llama-3.2-1B-Instruct with LoRA to map EISV trajectories to hierarchical expression tokens (EISV tokens + Lumen primitives + pattern class).

**Architecture:** Text-serialized trajectory windows as input, structured hierarchical output (EISV_TOKENS, LUMEN_TOKENS, PATTERN). LoRA rank 16 on attention projections. Training data from Layer 2 rule-based generator + synthetic augmentation.

**Tech Stack:** PyTorch, transformers, peft (LoRA), datasets (HuggingFace), existing eisv-lumen infrastructure.

---

### Task 1: Training Data Formatter

**Files:**
- Create: `eisv_lumen/training/__init__.py` (already exists, empty)
- Create: `eisv_lumen/training/data_prep.py`
- Test: `tests/test_training/test_data_prep.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_data_prep.py
"""Tests for training data preparation."""

import pytest

from eisv_lumen.training.data_prep import (
    format_trajectory_input,
    format_expression_output,
    build_training_example,
    TrainingExample,
)
from eisv_lumen.synthetic.trajectory_generator import generate_trajectory
from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.shapes.shape_classes import classify_trajectory


class TestFormatTrajectoryInput:
    """Test trajectory -> text serialization."""

    def test_basic_format(self):
        """Input text contains shape, mean EISV, derivatives."""
        states = generate_trajectory("settled_presence", n_points=20, seed=42)
        window = compute_trajectory_window(states)
        shape = classify_trajectory(window)
        text = format_trajectory_input(shape.value, window)
        assert "shape: settled_presence" in text
        assert "eisv_mean:" in text
        assert "derivatives:" in text
        assert "second_derivatives:" in text

    def test_contains_numeric_values(self):
        """Input includes actual numeric EISV values."""
        states = generate_trajectory("rising_entropy", n_points=20, seed=42)
        window = compute_trajectory_window(states)
        shape = classify_trajectory(window)
        text = format_trajectory_input(shape.value, window)
        # Should contain E= with some float
        assert "E=" in text
        assert "dS=" in text

    def test_window_metadata(self):
        """Input includes window size and duration."""
        states = generate_trajectory("convergence", n_points=20, dt=2.0, seed=42)
        window = compute_trajectory_window(states)
        shape = classify_trajectory(window)
        text = format_trajectory_input(shape.value, window)
        assert "window:" in text
        assert "20 states" in text


class TestFormatExpressionOutput:
    """Test expression -> structured output formatting."""

    def test_basic_format(self):
        """Output contains EISV_TOKENS, LUMEN_TOKENS, PATTERN."""
        text = format_expression_output(
            eisv_tokens=["~warmth~", "~stillness~"],
            lumen_tokens=["warm", "quiet"],
            pattern="PAIR",
        )
        assert "EISV_TOKENS: ~warmth~ ~stillness~" in text
        assert "LUMEN_TOKENS: warm quiet" in text
        assert "PATTERN: PAIR" in text

    def test_single_token(self):
        """Single-token expression formats correctly."""
        text = format_expression_output(
            eisv_tokens=["~stillness~"],
            lumen_tokens=["quiet"],
            pattern="SINGLE",
        )
        assert "EISV_TOKENS: ~stillness~" in text
        assert "PATTERN: SINGLE" in text

    def test_triple_token(self):
        """Triple-token expression formats correctly."""
        text = format_expression_output(
            eisv_tokens=["~curiosity~", "~ripple~", "~questioning~"],
            lumen_tokens=["why", "busy", "what"],
            pattern="TRIPLE",
        )
        assert "~curiosity~ ~ripple~ ~questioning~" in text
        assert "why busy what" in text


class TestBuildTrainingExample:
    """Test full training example construction."""

    def test_returns_training_example(self):
        """build_training_example returns a TrainingExample."""
        states = generate_trajectory("settled_presence", n_points=20, seed=42)
        window = compute_trajectory_window(states)
        shape = classify_trajectory(window)
        example = build_training_example(shape.value, window, seed=42)
        assert isinstance(example, TrainingExample)
        assert example.shape == "settled_presence"
        assert len(example.eisv_tokens) >= 1
        assert len(example.lumen_tokens) >= 1
        assert example.pattern in ("SINGLE", "PAIR", "TRIPLE", "REPETITION", "QUESTION")
        assert len(example.input_text) > 0
        assert len(example.output_text) > 0

    def test_deterministic_with_seed(self):
        """Same seed produces same example."""
        states = generate_trajectory("rising_entropy", n_points=20, seed=42)
        window = compute_trajectory_window(states)
        shape = classify_trajectory(window)
        ex1 = build_training_example(shape.value, window, seed=100)
        ex2 = build_training_example(shape.value, window, seed=100)
        assert ex1.eisv_tokens == ex2.eisv_tokens
        assert ex1.output_text == ex2.output_text

    def test_lumen_tokens_match_bridge(self):
        """Lumen tokens come from bridge translation of EISV tokens."""
        from eisv_lumen.bridge.lumen_bridge import translate_expression
        states = generate_trajectory("basin_transition_up", n_points=20, seed=42)
        window = compute_trajectory_window(states)
        shape = classify_trajectory(window)
        example = build_training_example(shape.value, window, seed=42)
        expected_lumen = translate_expression(example.eisv_tokens)
        assert example.lumen_tokens == expected_lumen
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_training/test_data_prep.py -v`
Expected: FAIL with ImportError (data_prep module doesn't exist yet)

**Step 3: Write minimal implementation**

```python
# eisv_lumen/training/data_prep.py
"""Training data preparation for Layer 3 teacher model.

Converts trajectory windows + expression labels into text-formatted
training examples for Llama-3.2-1B-Instruct LoRA fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from eisv_lumen.bridge.lumen_bridge import translate_expression
from eisv_lumen.shapes.expression_generator import ExpressionGenerator, ExpressionPattern


@dataclass
class TrainingExample:
    """A single training example for the teacher model."""
    shape: str
    eisv_tokens: List[str]
    lumen_tokens: List[str]
    pattern: str
    input_text: str
    output_text: str


def format_trajectory_input(shape: str, window: Dict[str, Any]) -> str:
    """Format a trajectory window as text input for the teacher model.

    Parameters
    ----------
    shape:
        Trajectory shape name.
    window:
        Output of compute_trajectory_window() with keys:
        states, derivatives, second_derivatives.

    Returns
    -------
    Formatted text string for model input.
    """
    states = window["states"]
    derivs = window["derivatives"]
    second_derivs = window["second_derivatives"]

    n_states = len(states)

    # Compute mean EISV values
    e_vals = [s["E"] for s in states]
    i_vals = [s["I"] for s in states]
    s_vals = [s["S"] for s in states]
    v_vals = [s["V"] for s in states]

    e_mean = sum(e_vals) / len(e_vals)
    i_mean = sum(i_vals) / len(i_vals)
    s_mean = sum(s_vals) / len(s_vals)
    v_mean = sum(v_vals) / len(v_vals)

    # Compute mean derivatives
    de_vals = [d["dE"] for d in derivs]
    di_vals = [d["dI"] for d in derivs]
    ds_vals = [d["dS"] for d in derivs]
    dv_vals = [d["dV"] for d in derivs]

    de_mean = sum(de_vals) / len(de_vals)
    di_mean = sum(di_vals) / len(di_vals)
    ds_mean = sum(ds_vals) / len(ds_vals)
    dv_mean = sum(dv_vals) / len(dv_vals)

    # Compute mean second derivatives
    d2e_vals = [d["d2E"] for d in second_derivs]
    d2i_vals = [d["d2I"] for d in second_derivs]
    d2s_vals = [d["d2S"] for d in second_derivs]
    d2v_vals = [d["d2V"] for d in second_derivs]

    d2e_mean = sum(d2e_vals) / len(d2e_vals)
    d2i_mean = sum(d2i_vals) / len(d2i_vals)
    d2s_mean = sum(d2s_vals) / len(d2s_vals)
    d2v_mean = sum(d2v_vals) / len(d2v_vals)

    # Compute window duration
    t_start = states[0]["t"]
    t_end = states[-1]["t"]
    duration = t_end - t_start

    return (
        f"TRAJECTORY:\n"
        f"shape: {shape}\n"
        f"window: {n_states} states over {duration:.0f}s\n"
        f"eisv_mean: E={e_mean:.2f}, I={i_mean:.2f}, S={s_mean:.2f}, V={v_mean:.2f}\n"
        f"derivatives: dE={de_mean:+.3f}, dI={di_mean:+.3f}, dS={ds_mean:+.3f}, dV={dv_mean:+.3f}\n"
        f"second_derivatives: d2E={d2e_mean:+.3f}, d2I={d2i_mean:+.3f}, d2S={d2s_mean:+.3f}, d2V={d2v_mean:+.3f}"
    )


def format_expression_output(
    eisv_tokens: List[str],
    lumen_tokens: List[str],
    pattern: str,
) -> str:
    """Format expression as structured output text.

    Parameters
    ----------
    eisv_tokens:
        EISV-Lumen token list.
    lumen_tokens:
        Lumen primitive token list.
    pattern:
        Expression pattern name (SINGLE, PAIR, TRIPLE, REPETITION, QUESTION).

    Returns
    -------
    Formatted text string for model output.
    """
    eisv_str = " ".join(eisv_tokens)
    lumen_str = " ".join(lumen_tokens)
    return (
        f"EISV_TOKENS: {eisv_str}\n"
        f"LUMEN_TOKENS: {lumen_str}\n"
        f"PATTERN: {pattern}"
    )


def _infer_pattern(tokens: List[str]) -> str:
    """Infer the expression pattern from a token list."""
    if len(tokens) == 1:
        return "SINGLE"
    if len(tokens) == 2 and tokens[0] == tokens[1]:
        return "REPETITION"
    if len(tokens) == 2:
        # Check if last token is an inquiry token
        if tokens[-1] in ("~questioning~", "~curiosity~"):
            return "QUESTION"
        return "PAIR"
    if len(tokens) == 3:
        return "TRIPLE"
    return "SINGLE"  # fallback


def build_training_example(
    shape: str,
    window: Dict[str, Any],
    seed: Optional[int] = None,
) -> TrainingExample:
    """Build a complete training example from a trajectory window.

    Uses the rule-based ExpressionGenerator to create labels,
    then translates via the Lumen bridge.

    Parameters
    ----------
    shape:
        Trajectory shape name.
    window:
        Output of compute_trajectory_window().
    seed:
        Random seed for reproducible expression generation.

    Returns
    -------
    A TrainingExample with input/output text and metadata.
    """
    gen = ExpressionGenerator(seed=seed)
    eisv_tokens = gen.generate(shape)
    lumen_tokens = translate_expression(eisv_tokens)
    pattern = _infer_pattern(eisv_tokens)

    input_text = format_trajectory_input(shape, window)
    output_text = format_expression_output(eisv_tokens, lumen_tokens, pattern)

    return TrainingExample(
        shape=shape,
        eisv_tokens=eisv_tokens,
        lumen_tokens=lumen_tokens,
        pattern=pattern,
        input_text=input_text,
        output_text=output_text,
    )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_training/test_data_prep.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add eisv_lumen/training/data_prep.py tests/test_training/test_data_prep.py
git commit -m "feat: add training data formatter for Layer 3 teacher model"
```

---

### Task 2: Dataset Builder (Augmentation + Splits)

**Files:**
- Create: `eisv_lumen/training/dataset_builder.py`
- Test: `tests/test_training/test_dataset_builder.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_dataset_builder.py
"""Tests for training dataset construction."""

import pytest

from eisv_lumen.training.dataset_builder import (
    build_training_dataset,
    split_dataset,
    DatasetStats,
)
from eisv_lumen.training.data_prep import TrainingExample
from eisv_lumen.shapes.shape_classes import TrajectoryShape


class TestBuildTrainingDataset:
    """Test dataset construction from real + synthetic trajectories."""

    def test_returns_list_of_examples(self):
        """build_training_dataset returns TrainingExample list."""
        # Use synthetic-only (no real db needed)
        examples = build_training_dataset(
            real_records=[],
            min_per_shape=5,
            seed=42,
        )
        assert isinstance(examples, list)
        assert all(isinstance(e, TrainingExample) for e in examples)

    def test_all_shapes_represented(self):
        """Every shape class has at least min_per_shape examples."""
        examples = build_training_dataset(
            real_records=[],
            min_per_shape=3,
            seed=42,
        )
        shapes = {e.shape for e in examples}
        for shape in TrajectoryShape:
            assert shape.value in shapes, f"Missing shape: {shape.value}"

    def test_min_per_shape_enforced(self):
        """Each shape has at least min_per_shape examples."""
        examples = build_training_dataset(
            real_records=[],
            min_per_shape=5,
            seed=42,
        )
        from collections import Counter
        counts = Counter(e.shape for e in examples)
        for shape in TrajectoryShape:
            assert counts[shape.value] >= 5

    def test_real_records_included(self):
        """Real trajectory records are converted to training examples."""
        from eisv_lumen.synthetic.trajectory_generator import generate_trajectory
        from eisv_lumen.extract.derivatives import compute_trajectory_window
        from eisv_lumen.shapes.shape_classes import classify_trajectory

        states = generate_trajectory("rising_entropy", n_points=20, seed=42)
        window = compute_trajectory_window(states)
        shape = classify_trajectory(window)

        real_records = [{
            "shape": shape.value,
            "states": window["states"],
            "derivatives": window["derivatives"],
            "second_derivatives": window["second_derivatives"],
            "t_start": states[0]["t"],
            "t_end": states[-1]["t"],
            "provenance": "lumen_real",
        }]

        examples = build_training_dataset(
            real_records=real_records,
            min_per_shape=1,
            seed=42,
        )
        assert len(examples) >= 9  # at least 1 per shape


class TestSplitDataset:
    """Test train/val/test splitting."""

    def test_split_ratios(self):
        """Default 80/10/10 split."""
        examples = build_training_dataset(
            real_records=[], min_per_shape=10, seed=42,
        )
        train, val, test = split_dataset(examples, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(examples)
        # Allow ±2 tolerance for rounding
        assert abs(len(train) / total - 0.8) < 0.05
        assert abs(len(val) / total - 0.1) < 0.05
        assert abs(len(test) / total - 0.1) < 0.05

    def test_no_overlap(self):
        """No examples appear in multiple splits."""
        examples = build_training_dataset(
            real_records=[], min_per_shape=10, seed=42,
        )
        train, val, test = split_dataset(examples, seed=42)
        train_texts = {e.input_text for e in train}
        val_texts = {e.input_text for e in val}
        test_texts = {e.input_text for e in test}
        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0

    def test_deterministic(self):
        """Same seed produces same split."""
        examples = build_training_dataset(
            real_records=[], min_per_shape=5, seed=42,
        )
        t1, v1, te1 = split_dataset(examples, seed=42)
        t2, v2, te2 = split_dataset(examples, seed=42)
        assert [e.input_text for e in t1] == [e.input_text for e in t2]

    def test_stratified_by_shape(self):
        """Each split contains all shapes."""
        examples = build_training_dataset(
            real_records=[], min_per_shape=20, seed=42,
        )
        train, val, test = split_dataset(examples, seed=42)
        all_shapes = {s.value for s in TrajectoryShape}
        assert {e.shape for e in train} == all_shapes
        assert {e.shape for e in val} == all_shapes
        assert {e.shape for e in test} == all_shapes
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_training/test_dataset_builder.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# eisv_lumen/training/dataset_builder.py
"""Build training datasets for Layer 3 teacher model.

Combines real trajectory records with synthetic augmentation,
constructs training examples, and provides stratified splitting.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.shapes.shape_classes import TrajectoryShape, classify_trajectory
from eisv_lumen.synthetic.trajectory_generator import generate_trajectory
from eisv_lumen.training.data_prep import TrainingExample, build_training_example


@dataclass
class DatasetStats:
    """Statistics about the built dataset."""
    total: int
    per_shape: Dict[str, int]
    real_count: int
    synthetic_count: int


def build_training_dataset(
    real_records: List[Dict[str, Any]],
    min_per_shape: int = 50,
    seed: int = 42,
) -> List[TrainingExample]:
    """Build training examples from real records + synthetic augmentation.

    Parameters
    ----------
    real_records:
        Trajectory records from assembler (each has shape, states,
        derivatives, second_derivatives keys).
    min_per_shape:
        Minimum examples per shape class. Synthetic trajectories fill gaps.
    seed:
        Random seed for reproducible generation.

    Returns
    -------
    List of TrainingExample, one per trajectory.
    """
    examples: List[TrainingExample] = []
    shape_counts: Dict[str, int] = Counter()
    rng = random.Random(seed)

    # Convert real records to training examples
    for i, record in enumerate(real_records):
        shape = record["shape"]
        window = {
            "states": record["states"],
            "derivatives": record["derivatives"],
            "second_derivatives": record["second_derivatives"],
        }
        example_seed = seed + i
        example = build_training_example(shape, window, seed=example_seed)
        examples.append(example)
        shape_counts[shape] += 1

    # Generate synthetic examples for underrepresented shapes
    offset = len(real_records)
    for shape in TrajectoryShape:
        deficit = min_per_shape - shape_counts.get(shape.value, 0)
        if deficit <= 0:
            continue
        for j in range(deficit):
            synth_seed = seed + offset + j
            states = generate_trajectory(
                shape.value, n_points=20, dt=2.0, seed=synth_seed,
            )
            window = compute_trajectory_window(states)
            example = build_training_example(shape.value, window, seed=synth_seed)
            examples.append(example)
            offset += 1

    return examples


def split_dataset(
    examples: List[TrainingExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
    """Stratified train/val/test split preserving shape distribution.

    Parameters
    ----------
    examples:
        Full list of training examples.
    train_ratio:
        Fraction for training set (default 0.8).
    val_ratio:
        Fraction for validation set (default 0.1).
        Test set gets the remainder.
    seed:
        Random seed for reproducible splitting.

    Returns
    -------
    Tuple of (train, val, test) example lists.
    """
    rng = random.Random(seed)

    # Group by shape for stratified splitting
    by_shape: Dict[str, List[TrainingExample]] = defaultdict(list)
    for ex in examples:
        by_shape[ex.shape].append(ex)

    train, val, test = [], [], []

    for shape in sorted(by_shape.keys()):
        group = by_shape[shape]
        rng.shuffle(group)

        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        # Ensure at least 1 in test if enough examples
        n_test = n - n_train - n_val
        if n_test < 1 and n >= 3:
            n_train -= 1
            n_test = 1

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_training/test_dataset_builder.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add eisv_lumen/training/dataset_builder.py tests/test_training/test_dataset_builder.py
git commit -m "feat: add training dataset builder with augmentation and stratified splits"
```

---

### Task 3: Chat Template Formatter

**Files:**
- Create: `eisv_lumen/training/chat_format.py`
- Test: `tests/test_training/test_chat_format.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_chat_format.py
"""Tests for chat template formatting."""

import pytest

from eisv_lumen.training.chat_format import (
    SYSTEM_PROMPT,
    format_chat_messages,
    format_for_tokenizer,
)
from eisv_lumen.training.data_prep import TrainingExample


class TestFormatChatMessages:
    """Test conversion to chat message format."""

    def test_returns_message_list(self):
        """Returns list of role/content dicts."""
        ex = TrainingExample(
            shape="settled_presence",
            eisv_tokens=["~stillness~"],
            lumen_tokens=["quiet"],
            pattern="SINGLE",
            input_text="TRAJECTORY:\nshape: settled_presence\n...",
            output_text="EISV_TOKENS: ~stillness~\nLUMEN_TOKENS: quiet\nPATTERN: SINGLE",
        )
        messages = format_chat_messages(ex)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_system_prompt_present(self):
        """System message contains the trajectory mapper instruction."""
        ex = TrainingExample(
            shape="settled_presence",
            eisv_tokens=["~stillness~"],
            lumen_tokens=["quiet"],
            pattern="SINGLE",
            input_text="INPUT",
            output_text="OUTPUT",
        )
        messages = format_chat_messages(ex)
        assert "trajectory" in messages[0]["content"].lower()
        assert "expression" in messages[0]["content"].lower()

    def test_user_contains_input(self):
        """User message contains the trajectory input text."""
        ex = TrainingExample(
            shape="rising_entropy",
            eisv_tokens=["~curiosity~"],
            lumen_tokens=["why"],
            pattern="SINGLE",
            input_text="TRAJECTORY:\nshape: rising_entropy",
            output_text="OUTPUT",
        )
        messages = format_chat_messages(ex)
        assert "rising_entropy" in messages[1]["content"]

    def test_assistant_contains_output(self):
        """Assistant message contains the expression output text."""
        ex = TrainingExample(
            shape="convergence",
            eisv_tokens=["~stillness~", "~resonance~"],
            lumen_tokens=["quiet", "with"],
            pattern="PAIR",
            input_text="INPUT",
            output_text="EISV_TOKENS: ~stillness~ ~resonance~\nLUMEN_TOKENS: quiet with\nPATTERN: PAIR",
        )
        messages = format_chat_messages(ex)
        assert "~stillness~ ~resonance~" in messages[2]["content"]


class TestFormatForTokenizer:
    """Test HuggingFace tokenizer-ready formatting."""

    def test_returns_dict_with_text(self):
        """format_for_tokenizer returns dict with 'text' key."""
        ex = TrainingExample(
            shape="settled_presence",
            eisv_tokens=["~stillness~"],
            lumen_tokens=["quiet"],
            pattern="SINGLE",
            input_text="INPUT",
            output_text="OUTPUT",
        )
        result = format_for_tokenizer(ex)
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0

    def test_metadata_preserved(self):
        """Result includes shape and pattern metadata."""
        ex = TrainingExample(
            shape="rising_entropy",
            eisv_tokens=["~curiosity~"],
            lumen_tokens=["why"],
            pattern="SINGLE",
            input_text="INPUT",
            output_text="OUTPUT",
        )
        result = format_for_tokenizer(ex)
        assert result["shape"] == "rising_entropy"
        assert result["pattern"] == "SINGLE"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_training/test_chat_format.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# eisv_lumen/training/chat_format.py
"""Chat template formatting for Llama-3.2-1B-Instruct fine-tuning.

Converts TrainingExample objects into chat message format compatible
with the Llama-3 instruction template.
"""

from __future__ import annotations

from typing import Any, Dict, List

from eisv_lumen.training.data_prep import TrainingExample


SYSTEM_PROMPT = (
    "You are Lumen's trajectory-to-expression mapper. "
    "Given an EISV trajectory, output the appropriate expression tokens.\n\n"
    "Output format:\n"
    "EISV_TOKENS: <space-separated EISV-Lumen tokens>\n"
    "LUMEN_TOKENS: <space-separated Lumen primitive tokens>\n"
    "PATTERN: <SINGLE|PAIR|TRIPLE|REPETITION|QUESTION>"
)


def format_chat_messages(example: TrainingExample) -> List[Dict[str, str]]:
    """Convert a TrainingExample to chat message format.

    Parameters
    ----------
    example:
        A training example with input/output text.

    Returns
    -------
    List of message dicts with 'role' and 'content' keys,
    suitable for tokenizer.apply_chat_template().
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example.input_text},
        {"role": "assistant", "content": example.output_text},
    ]


def format_for_tokenizer(example: TrainingExample) -> Dict[str, Any]:
    """Convert a TrainingExample to a tokenizer-ready dict.

    Returns a dict with:
    - 'text': the full chat-formatted text (for use with tokenizer)
    - 'shape': shape label (metadata)
    - 'pattern': pattern label (metadata)
    - 'messages': the raw message list

    Parameters
    ----------
    example:
        A training example.

    Returns
    -------
    Dict ready for HuggingFace Dataset.from_list().
    """
    messages = format_chat_messages(example)

    # Build a simple text representation (will be properly tokenized during training)
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|end|>")
    text = "\n".join(parts)

    return {
        "text": text,
        "shape": example.shape,
        "pattern": example.pattern,
        "messages": messages,
    }
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_training/test_chat_format.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add eisv_lumen/training/chat_format.py tests/test_training/test_chat_format.py
git commit -m "feat: add chat template formatter for Llama-3 instruction tuning"
```

---

### Task 4: LoRA Training Config

**Files:**
- Create: `eisv_lumen/training/configs/teacher_lora.yaml`
- Create: `eisv_lumen/training/config.py`
- Test: `tests/test_training/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_config.py
"""Tests for training configuration."""

import pytest

from eisv_lumen.training.config import TrainingConfig, load_config


class TestTrainingConfig:
    """Test training config dataclass."""

    def test_defaults(self):
        """Default config has sensible values."""
        cfg = TrainingConfig()
        assert cfg.model_name == "meta-llama/Llama-3.2-1B-Instruct"
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.learning_rate == 2e-4
        assert cfg.num_epochs == 5
        assert cfg.batch_size >= 1
        assert cfg.warmup_steps == 100

    def test_lora_target_modules(self):
        """LoRA targets attention projections."""
        cfg = TrainingConfig()
        assert "q_proj" in cfg.lora_target_modules
        assert "k_proj" in cfg.lora_target_modules
        assert "v_proj" in cfg.lora_target_modules
        assert "o_proj" in cfg.lora_target_modules

    def test_custom_values(self):
        """Config accepts custom values."""
        cfg = TrainingConfig(lora_rank=8, num_epochs=3)
        assert cfg.lora_rank == 8
        assert cfg.num_epochs == 3


class TestLoadConfig:
    """Test loading config from YAML."""

    def test_load_default(self):
        """load_config() with no args returns defaults."""
        cfg = load_config()
        assert isinstance(cfg, TrainingConfig)
        assert cfg.model_name == "meta-llama/Llama-3.2-1B-Instruct"

    def test_load_from_yaml(self, tmp_path):
        """load_config() reads a YAML file."""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "model_name: test-model\n"
            "lora_rank: 8\n"
            "num_epochs: 2\n"
        )
        cfg = load_config(str(yaml_path))
        assert cfg.model_name == "test-model"
        assert cfg.lora_rank == 8
        assert cfg.num_epochs == 2
        # Unset fields keep defaults
        assert cfg.lora_alpha == 32
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_training/test_config.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```yaml
# eisv_lumen/training/configs/teacher_lora.yaml
# Layer 3 Phase 1: Teacher model LoRA configuration
model_name: meta-llama/Llama-3.2-1B-Instruct
lora_rank: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
lora_dropout: 0.05
learning_rate: 0.0002
num_epochs: 5
batch_size: 4
gradient_accumulation_steps: 4
warmup_steps: 100
max_seq_length: 512
weight_decay: 0.01
fp16: true
seed: 42
output_dir: outputs/teacher_lora
```

```python
# eisv_lumen/training/config.py
"""Training configuration for Layer 3 teacher model."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning."""

    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
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
    """Load training config from YAML file, falling back to defaults.

    Parameters
    ----------
    yaml_path:
        Path to YAML config file. If None, returns defaults.

    Returns
    -------
    TrainingConfig with values from file merged over defaults.
    """
    if yaml_path is None:
        return TrainingConfig()

    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    # Only pass keys that exist in the dataclass
    valid_keys = {f.name for f in fields(TrainingConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}

    return TrainingConfig(**filtered)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_training/test_config.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
mkdir -p eisv_lumen/training/configs
git add eisv_lumen/training/config.py eisv_lumen/training/configs/teacher_lora.yaml tests/test_training/test_config.py
git commit -m "feat: add training config with LoRA defaults and YAML loading"
```

---

### Task 5: Teacher Training Script

**Files:**
- Create: `eisv_lumen/training/teacher_train.py`
- Test: `tests/test_training/test_teacher_train.py`

**Step 1: Write the failing test**

Note: Training tests are lightweight — they test the training setup/pipeline without actually running training (which requires GPU + model download). Tests mock the model/tokenizer.

```python
# tests/test_training/test_teacher_train.py
"""Tests for teacher model training pipeline.

These tests validate the training setup without requiring GPU or
model downloads. They test data pipeline, config validation, and
output parsing — not actual training.
"""

import pytest
from unittest.mock import MagicMock, patch

from eisv_lumen.training.teacher_train import (
    prepare_training_data,
    parse_model_output,
    validate_output,
    OutputParseResult,
)
from eisv_lumen.training.data_prep import TrainingExample


class TestPrepareTrainingData:
    """Test the data preparation pipeline for training."""

    def test_returns_hf_dataset_dicts(self):
        """prepare_training_data returns train/val/test HF-compatible dicts."""
        train, val, test = prepare_training_data(
            real_records=[],
            min_per_shape=3,
            seed=42,
        )
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        # Each item should have 'text', 'shape', 'messages'
        assert "text" in train[0]
        assert "shape" in train[0]
        assert "messages" in train[0]

    def test_all_shapes_in_train(self):
        """Training set has all 9 shapes."""
        train, _, _ = prepare_training_data(
            real_records=[], min_per_shape=5, seed=42,
        )
        shapes = {d["shape"] for d in train}
        assert len(shapes) == 9


class TestParseModelOutput:
    """Test parsing of model-generated text."""

    def test_valid_output(self):
        """Parse well-formatted model output."""
        text = (
            "EISV_TOKENS: ~warmth~ ~stillness~\n"
            "LUMEN_TOKENS: warm quiet\n"
            "PATTERN: PAIR"
        )
        result = parse_model_output(text)
        assert result.eisv_tokens == ["~warmth~", "~stillness~"]
        assert result.lumen_tokens == ["warm", "quiet"]
        assert result.pattern == "PAIR"
        assert result.valid is True

    def test_single_token(self):
        """Parse single-token output."""
        text = (
            "EISV_TOKENS: ~curiosity~\n"
            "LUMEN_TOKENS: why\n"
            "PATTERN: SINGLE"
        )
        result = parse_model_output(text)
        assert result.eisv_tokens == ["~curiosity~"]
        assert result.valid is True

    def test_malformed_output(self):
        """Gracefully handle malformed output."""
        result = parse_model_output("garbage text here")
        assert result.valid is False
        assert result.eisv_tokens == []

    def test_partial_output(self):
        """Handle output missing some fields."""
        text = "EISV_TOKENS: ~warmth~\nPATTERN: SINGLE"
        result = parse_model_output(text)
        assert result.eisv_tokens == ["~warmth~"]
        # Missing LUMEN_TOKENS should be empty
        assert result.lumen_tokens == []
        assert result.valid is False  # incomplete = invalid


class TestValidateOutput:
    """Test output validation against vocabulary."""

    def test_valid_tokens(self):
        """All tokens in vocabulary passes validation."""
        result = OutputParseResult(
            eisv_tokens=["~warmth~", "~curiosity~"],
            lumen_tokens=["warm", "why"],
            pattern="PAIR",
            valid=True,
        )
        assert validate_output(result) is True

    def test_invalid_eisv_token(self):
        """Unknown EISV token fails validation."""
        result = OutputParseResult(
            eisv_tokens=["~warmth~", "~unknown~"],
            lumen_tokens=["warm", "???"],
            pattern="PAIR",
            valid=True,
        )
        assert validate_output(result) is False

    def test_invalid_pattern(self):
        """Unknown pattern fails validation."""
        result = OutputParseResult(
            eisv_tokens=["~warmth~"],
            lumen_tokens=["warm"],
            pattern="INVALID",
            valid=True,
        )
        assert validate_output(result) is False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_training/test_teacher_train.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# eisv_lumen/training/teacher_train.py
"""Teacher model training pipeline for Layer 3.

Handles data preparation, output parsing, and validation.
Actual training requires GPU and model download — run via CLI:
    python -m eisv_lumen.training.teacher_train --config configs/teacher_lora.yaml

This module separates the testable logic (data prep, parsing, validation)
from the GPU-dependent training loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eisv_lumen.eval.baseline import ALL_TOKENS
from eisv_lumen.bridge.lumen_bridge import LUMEN_TOKENS
from eisv_lumen.training.chat_format import format_for_tokenizer
from eisv_lumen.training.data_prep import TrainingExample
from eisv_lumen.training.dataset_builder import build_training_dataset, split_dataset


VALID_PATTERNS = {"SINGLE", "PAIR", "TRIPLE", "REPETITION", "QUESTION"}
EISV_TOKEN_SET = set(ALL_TOKENS)
LUMEN_TOKEN_SET = set(LUMEN_TOKENS)


@dataclass
class OutputParseResult:
    """Parsed model output."""
    eisv_tokens: List[str] = field(default_factory=list)
    lumen_tokens: List[str] = field(default_factory=list)
    pattern: str = ""
    valid: bool = False


def prepare_training_data(
    real_records: List[Dict[str, Any]],
    min_per_shape: int = 50,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Build and split training data into tokenizer-ready format.

    Parameters
    ----------
    real_records:
        Trajectory records from dataset assembly.
    min_per_shape:
        Minimum examples per shape (synthetic fills gaps).
    seed:
        Random seed.

    Returns
    -------
    Tuple of (train, val, test) where each is a list of dicts
    with 'text', 'shape', 'pattern', 'messages' keys.
    """
    examples = build_training_dataset(real_records, min_per_shape, seed)
    train_ex, val_ex, test_ex = split_dataset(examples, seed=seed)

    train = [format_for_tokenizer(e) for e in train_ex]
    val = [format_for_tokenizer(e) for e in val_ex]
    test = [format_for_tokenizer(e) for e in test_ex]

    return train, val, test


def parse_model_output(text: str) -> OutputParseResult:
    """Parse structured model output into tokens and pattern.

    Expected format:
        EISV_TOKENS: ~token1~ ~token2~
        LUMEN_TOKENS: prim1 prim2
        PATTERN: PAIR

    Parameters
    ----------
    text:
        Raw model output string.

    Returns
    -------
    OutputParseResult with parsed fields and validity flag.
    """
    result = OutputParseResult()

    # Parse EISV_TOKENS
    eisv_match = re.search(r"EISV_TOKENS:\s*(.+?)(?:\n|$)", text)
    if eisv_match:
        tokens_str = eisv_match.group(1).strip()
        result.eisv_tokens = [t for t in tokens_str.split() if t]

    # Parse LUMEN_TOKENS
    lumen_match = re.search(r"LUMEN_TOKENS:\s*(.+?)(?:\n|$)", text)
    if lumen_match:
        tokens_str = lumen_match.group(1).strip()
        result.lumen_tokens = [t for t in tokens_str.split() if t]

    # Parse PATTERN
    pattern_match = re.search(r"PATTERN:\s*(\w+)", text)
    if pattern_match:
        result.pattern = pattern_match.group(1).strip()

    # Valid only if all three fields are present
    result.valid = bool(
        result.eisv_tokens and result.lumen_tokens and result.pattern
    )

    return result


def validate_output(result: OutputParseResult) -> bool:
    """Validate parsed output against known vocabularies.

    Parameters
    ----------
    result:
        A parsed model output.

    Returns
    -------
    True if all tokens are in vocabulary and pattern is valid.
    """
    if not result.valid:
        return False

    # Check EISV tokens
    for token in result.eisv_tokens:
        if token not in EISV_TOKEN_SET:
            return False

    # Check Lumen tokens
    for token in result.lumen_tokens:
        if token not in LUMEN_TOKEN_SET:
            return False

    # Check pattern
    if result.pattern not in VALID_PATTERNS:
        return False

    return True
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_training/test_teacher_train.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add eisv_lumen/training/teacher_train.py tests/test_training/test_teacher_train.py
git commit -m "feat: add teacher training pipeline with output parsing and validation"
```

---

### Task 6: Teacher Evaluation Script

**Files:**
- Create: `eisv_lumen/training/teacher_eval.py`
- Test: `tests/test_training/test_teacher_eval.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_teacher_eval.py
"""Tests for teacher model evaluation."""

import pytest

from eisv_lumen.training.teacher_eval import (
    evaluate_predictions,
    EvalResults,
    check_gate1,
)
from eisv_lumen.training.teacher_train import OutputParseResult


class TestEvaluatePredictions:
    """Test evaluation of parsed model predictions."""

    def test_perfect_predictions(self):
        """All predictions match shape affinity = high coherence."""
        # settled_presence affinities: ~stillness~, ~holding~, ~resonance~, ~deep_listening~
        predictions = [
            {
                "shape": "settled_presence",
                "parsed": OutputParseResult(
                    eisv_tokens=["~stillness~", "~holding~"],
                    lumen_tokens=["quiet", "here"],
                    pattern="PAIR",
                    valid=True,
                ),
            }
        ] * 10
        results = evaluate_predictions(predictions)
        assert isinstance(results, EvalResults)
        assert results.mean_coherence == 1.0
        assert results.valid_rate == 1.0

    def test_mixed_predictions(self):
        """Mix of valid and invalid predictions."""
        predictions = [
            {
                "shape": "settled_presence",
                "parsed": OutputParseResult(
                    eisv_tokens=["~stillness~"],
                    lumen_tokens=["quiet"],
                    pattern="SINGLE",
                    valid=True,
                ),
            },
            {
                "shape": "rising_entropy",
                "parsed": OutputParseResult(
                    eisv_tokens=[],
                    lumen_tokens=[],
                    pattern="",
                    valid=False,
                ),
            },
        ]
        results = evaluate_predictions(predictions)
        assert results.valid_rate == 0.5
        assert results.n_total == 2
        assert results.n_valid == 1

    def test_pattern_accuracy(self):
        """Pattern accuracy computed over valid predictions."""
        predictions = [
            {
                "shape": "settled_presence",
                "parsed": OutputParseResult(
                    eisv_tokens=["~stillness~"],
                    lumen_tokens=["quiet"],
                    pattern="SINGLE",
                    valid=True,
                ),
                "expected_pattern": "SINGLE",
            },
            {
                "shape": "rising_entropy",
                "parsed": OutputParseResult(
                    eisv_tokens=["~curiosity~", "~ripple~"],
                    lumen_tokens=["why", "busy"],
                    pattern="TRIPLE",  # wrong pattern
                    valid=True,
                ),
                "expected_pattern": "PAIR",
            },
        ]
        results = evaluate_predictions(predictions)
        assert results.pattern_accuracy == 0.5

    def test_empty_predictions(self):
        """Empty prediction list returns zero results."""
        results = evaluate_predictions([])
        assert results.mean_coherence == 0.0
        assert results.n_total == 0


class TestCheckGate1:
    """Test go/no-go Gate 1 criteria."""

    def test_passes(self):
        """Gate 1 passes when coherence > 0.933."""
        results = EvalResults(
            mean_coherence=0.95,
            valid_rate=0.98,
            pattern_accuracy=0.85,
            n_total=100,
            n_valid=98,
            per_shape_coherence={"settled_presence": 0.95},
            diversity=0.7,
        )
        passed, reasons = check_gate1(results)
        assert passed is True

    def test_fails_low_coherence(self):
        """Gate 1 fails when coherence <= 0.933."""
        results = EvalResults(
            mean_coherence=0.90,
            valid_rate=0.98,
            pattern_accuracy=0.85,
            n_total=100,
            n_valid=98,
            per_shape_coherence={},
            diversity=0.7,
        )
        passed, reasons = check_gate1(results)
        assert passed is False
        assert any("coherence" in r.lower() for r in reasons)

    def test_fails_low_valid_rate(self):
        """Gate 1 fails when valid output rate < 0.90."""
        results = EvalResults(
            mean_coherence=0.95,
            valid_rate=0.80,
            pattern_accuracy=0.85,
            n_total=100,
            n_valid=80,
            per_shape_coherence={},
            diversity=0.7,
        )
        passed, reasons = check_gate1(results)
        assert passed is False
        assert any("valid" in r.lower() for r in reasons)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_training/test_teacher_eval.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# eisv_lumen/training/teacher_eval.py
"""Teacher model evaluation for Layer 3.

Computes coherence, pattern accuracy, diversity, and valid output rate.
Implements Gate 1 go/no-go check.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from eisv_lumen.eval.metrics import expression_trajectory_coherence
from eisv_lumen.training.teacher_train import OutputParseResult, validate_output


GATE1_COHERENCE_THRESHOLD = 0.933
GATE1_VALID_RATE_THRESHOLD = 0.90


@dataclass
class EvalResults:
    """Evaluation results for teacher model."""
    mean_coherence: float
    valid_rate: float
    pattern_accuracy: float
    n_total: int
    n_valid: int
    per_shape_coherence: Dict[str, float]
    diversity: float


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
) -> EvalResults:
    """Evaluate a list of model predictions.

    Each prediction dict should have:
    - 'shape' (str): trajectory shape
    - 'parsed' (OutputParseResult): parsed model output
    - 'expected_pattern' (str, optional): ground truth pattern

    Parameters
    ----------
    predictions:
        List of prediction dicts.

    Returns
    -------
    EvalResults with aggregate metrics.
    """
    if not predictions:
        return EvalResults(
            mean_coherence=0.0,
            valid_rate=0.0,
            pattern_accuracy=0.0,
            n_total=0,
            n_valid=0,
            per_shape_coherence={},
            diversity=0.0,
        )

    n_total = len(predictions)
    n_valid = 0
    coherence_scores: List[float] = []
    per_shape_scores: Dict[str, List[float]] = defaultdict(list)
    pattern_correct = 0
    pattern_total = 0
    all_tokens: List[str] = []

    for pred in predictions:
        shape = pred["shape"]
        parsed = pred["parsed"]

        if not parsed.valid:
            continue

        if not validate_output(parsed):
            continue

        n_valid += 1

        # Coherence
        score = expression_trajectory_coherence(shape, parsed.eisv_tokens)
        coherence_scores.append(score)
        per_shape_scores[shape].append(score)

        # Pattern accuracy
        expected = pred.get("expected_pattern")
        if expected:
            pattern_total += 1
            if parsed.pattern == expected:
                pattern_correct += 1

        # Diversity tracking
        all_tokens.extend(parsed.eisv_tokens)

    mean_coherence = (
        sum(coherence_scores) / len(coherence_scores)
        if coherence_scores else 0.0
    )

    valid_rate = n_valid / n_total if n_total > 0 else 0.0

    pattern_accuracy = (
        pattern_correct / pattern_total
        if pattern_total > 0 else 0.0
    )

    per_shape_coherence = {
        shape: sum(scores) / len(scores)
        for shape, scores in per_shape_scores.items()
    }

    # Token diversity: unique / total
    diversity = (
        len(set(all_tokens)) / len(all_tokens)
        if all_tokens else 0.0
    )

    return EvalResults(
        mean_coherence=mean_coherence,
        valid_rate=valid_rate,
        pattern_accuracy=pattern_accuracy,
        n_total=n_total,
        n_valid=n_valid,
        per_shape_coherence=per_shape_coherence,
        diversity=diversity,
    )


def check_gate1(results: EvalResults) -> Tuple[bool, List[str]]:
    """Check Gate 1 go/no-go criteria.

    Criteria:
    - Mean coherence > 0.933 (beat feedback-enhanced rules)
    - Valid output rate >= 0.90

    Parameters
    ----------
    results:
        Evaluation results from evaluate_predictions().

    Returns
    -------
    Tuple of (passed: bool, reasons: List[str]).
    Reasons explain failures; empty if passed.
    """
    reasons: List[str] = []

    if results.mean_coherence <= GATE1_COHERENCE_THRESHOLD:
        reasons.append(
            f"Coherence {results.mean_coherence:.3f} <= "
            f"{GATE1_COHERENCE_THRESHOLD} threshold"
        )

    if results.valid_rate < GATE1_VALID_RATE_THRESHOLD:
        reasons.append(
            f"Valid output rate {results.valid_rate:.3f} < "
            f"{GATE1_VALID_RATE_THRESHOLD} threshold"
        )

    return len(reasons) == 0, reasons
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_training/test_teacher_eval.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add eisv_lumen/training/teacher_eval.py tests/test_training/test_teacher_eval.py
git commit -m "feat: add teacher evaluation with coherence metrics and Gate 1 check"
```

---

### Task 7: Training CLI Entry Point

**Files:**
- Create: `eisv_lumen/training/cli.py`
- Modify: `eisv_lumen/training/__init__.py`

This task creates the CLI that ties everything together. It requires a GPU + model download to actually run, so it's tested manually:

**Step 1: Write the CLI module**

```python
# eisv_lumen/training/cli.py
"""CLI entry point for Layer 3 teacher model training.

Usage:
    python -m eisv_lumen.training.cli train --config configs/teacher_lora.yaml
    python -m eisv_lumen.training.cli eval --model outputs/teacher_lora --test-data data/test.json
    python -m eisv_lumen.training.cli gate1 --results outputs/eval_results.json

Requires: GPU with >= 8GB VRAM, or Apple Silicon Mac with >= 8GB unified memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_prepare(args: argparse.Namespace) -> None:
    """Prepare training data and save to disk."""
    from eisv_lumen.training.teacher_train import prepare_training_data

    print(f"Preparing training data (min_per_shape={args.min_per_shape})...")
    train, val, test = prepare_training_data(
        real_records=[],  # TODO: load from anima.db when available
        min_per_shape=args.min_per_shape,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            # Strip messages (not JSON serializable easily) for disk format
            serializable = [
                {k: v for k, v in d.items() if k != "messages"}
                for d in data
            ]
            json.dump(serializable, f, indent=2)
        print(f"  {name}: {len(data)} examples -> {path}")

    print("Done.")


def cmd_train(args: argparse.Namespace) -> None:
    """Run LoRA fine-tuning (requires GPU/Apple Silicon)."""
    print("Training requires GPU. Use:")
    print(f"  python -m eisv_lumen.training.cli train --config {args.config}")
    print()
    print("Full training implementation will be added when ready to run.")
    print("Current focus: data preparation, evaluation, and gate checks.")


def cmd_gate1(args: argparse.Namespace) -> None:
    """Check Gate 1 criteria on evaluation results."""
    from eisv_lumen.training.teacher_eval import EvalResults, check_gate1

    with open(args.results) as f:
        data = json.load(f)

    results = EvalResults(**data)
    passed, reasons = check_gate1(results)

    print(f"\n{'='*50}")
    print(f"  GATE 1: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'='*50}")
    print(f"  Coherence: {results.mean_coherence:.3f} (threshold: >0.933)")
    print(f"  Valid rate: {results.valid_rate:.3f} (threshold: >=0.90)")
    print(f"  Pattern accuracy: {results.pattern_accuracy:.3f}")
    print(f"  Diversity: {results.diversity:.3f}")
    print(f"  N total: {results.n_total}, N valid: {results.n_valid}")

    if not passed:
        print(f"\n  Reasons for failure:")
        for r in reasons:
            print(f"    - {r}")

    print(f"{'='*50}\n")
    sys.exit(0 if passed else 1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EISV-Lumen Layer 3 Training CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # prepare
    prep = sub.add_parser("prepare", help="Prepare training data")
    prep.add_argument("--min-per-shape", type=int, default=50)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--output-dir", default="data/training")

    # train
    train = sub.add_parser("train", help="Run LoRA fine-tuning")
    train.add_argument("--config", default="eisv_lumen/training/configs/teacher_lora.yaml")

    # gate1
    gate = sub.add_parser("gate1", help="Check Gate 1 go/no-go")
    gate.add_argument("--results", required=True, help="Path to eval_results.json")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "gate1":
        cmd_gate1(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

```python
# eisv_lumen/training/__init__.py
"""Layer 3 training infrastructure for EISV-Lumen deep voice model."""
```

**Step 2: Run all training tests**

Run: `python3 -m pytest tests/test_training/ -v`
Expected: All tests PASS (Tasks 1-6: ~35 tests total)

**Step 3: Run CLI smoke test**

Run: `python3 -m eisv_lumen.training.cli prepare --min-per-shape 3 --output-dir /tmp/eisv_test_data`
Expected: Creates train.json, val.json, test.json with ~27+ examples

**Step 4: Run full test suite**

Run: `python3 -m pytest tests/ -q`
Expected: All 263+ existing tests still pass, plus ~35 new training tests

**Step 5: Commit**

```bash
git add eisv_lumen/training/__init__.py eisv_lumen/training/cli.py
git commit -m "feat: add Layer 3 training CLI with prepare, train, and gate1 commands"
```

---

### Task 8: Integration Test — Full Pipeline

**Files:**
- Create: `tests/test_training/test_pipeline.py`

**Step 1: Write the integration test**

```python
# tests/test_training/test_pipeline.py
"""Integration test: full training data pipeline.

Tests the complete flow from trajectory generation through
dataset building, chat formatting, and evaluation.
"""

import pytest

from eisv_lumen.training.teacher_train import prepare_training_data, parse_model_output
from eisv_lumen.training.teacher_eval import evaluate_predictions, check_gate1
from eisv_lumen.training.data_prep import build_training_example
from eisv_lumen.training.chat_format import format_chat_messages
from eisv_lumen.synthetic.trajectory_generator import generate_trajectory
from eisv_lumen.extract.derivatives import compute_trajectory_window
from eisv_lumen.shapes.shape_classes import classify_trajectory, TrajectoryShape


class TestFullPipeline:
    """End-to-end pipeline test."""

    def test_trajectory_to_training_example(self):
        """Generate trajectory -> classify -> build example -> format."""
        for shape in TrajectoryShape:
            states = generate_trajectory(shape.value, n_points=20, seed=42)
            window = compute_trajectory_window(states)
            classified = classify_trajectory(window)
            assert classified.value == shape.value

            example = build_training_example(shape.value, window, seed=42)
            messages = format_chat_messages(example)
            assert len(messages) == 3
            assert shape.value in messages[1]["content"]

    def test_prepare_and_evaluate(self):
        """Prepare data, simulate perfect predictions, evaluate."""
        train, val, test = prepare_training_data(
            real_records=[], min_per_shape=5, seed=42,
        )

        # Simulate perfect model: parse the ground truth output
        predictions = []
        for item in test:
            # The output_text is in the assistant message
            output_text = item["messages"][2]["content"]
            parsed = parse_model_output(output_text)
            predictions.append({
                "shape": item["shape"],
                "parsed": parsed,
                "expected_pattern": item["pattern"],
            })

        results = evaluate_predictions(predictions)
        # Rule-based labels should have high coherence (they use affinity weighting)
        assert results.mean_coherence > 0.4  # conservative threshold
        assert results.valid_rate > 0.9
        assert results.n_valid > 0

    def test_gate1_on_rule_labels(self):
        """Rule-based labels alone may not pass Gate 1 (no feedback)."""
        train, val, test = prepare_training_data(
            real_records=[], min_per_shape=20, seed=42,
        )
        predictions = []
        for item in test:
            output_text = item["messages"][2]["content"]
            parsed = parse_model_output(output_text)
            predictions.append({
                "shape": item["shape"],
                "parsed": parsed,
            })

        results = evaluate_predictions(predictions)
        # Without feedback, coherence is ~0.503 — Gate 1 requires >0.933
        # This test documents the baseline expectation
        passed, reasons = check_gate1(results)
        # We expect this to FAIL (rule labels without feedback < 0.933)
        # The teacher model needs to learn to beat this
        assert results.mean_coherence > 0.3  # sanity check: above random (0.265)
```

**Step 2: Run test**

Run: `python3 -m pytest tests/test_training/test_pipeline.py -v`
Expected: All 3 tests PASS

**Step 3: Commit**

```bash
git add tests/test_training/test_pipeline.py
git commit -m "test: add end-to-end training pipeline integration test"
```

---

### Task 9: Add PyYAML Dependency + Final Verification

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add PyYAML to dependencies**

Add `"pyyaml>=6.0"` to the dependencies list in `pyproject.toml` (needed for config loading).

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: 263+ existing tests + ~38 new training tests all PASS

**Step 3: Run CLI smoke test**

Run: `python3 -m eisv_lumen.training.cli prepare --min-per-shape 5 --output-dir /tmp/eisv_verify`
Expected: Creates 3 JSON files, prints counts

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyyaml dependency for training config"
```

**Step 5: Push**

```bash
git push origin main
```

---

## Summary

**9 tasks** building the Phase 1 teacher model infrastructure:

| Task | What | Tests | Cumulative |
|------|------|-------|-----------|
| 1 | Training data formatter | 9 | 9 |
| 2 | Dataset builder + splits | 8 | 17 |
| 3 | Chat template formatter | 6 | 23 |
| 4 | LoRA training config | 5 | 28 |
| 5 | Teacher training pipeline | 10 | 38 |
| 6 | Teacher evaluation + Gate 1 | 7 | 45 |
| 7 | CLI entry point | 0 (manual) | 45 |
| 8 | Integration test | 3 | 48 |
| 9 | Dependency + verify | 0 | 48 |

**After this plan:** The training infrastructure is complete. To actually train:
1. Download Llama-3.2-1B-Instruct (requires HF access token)
2. Run `python -m eisv_lumen.training.cli prepare`
3. Add the actual LoRA training loop to `teacher_train.py` (GPU-dependent code)
4. Train, evaluate, check Gate 1

**Next plan (after Gate 1 passes):** Phase 2 — distilled student model.
