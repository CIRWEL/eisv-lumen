# Layer 3: Fine-Tuned Deep Voice — Design Document

**Date:** 2026-02-12
**Status:** Approved
**Approach:** Staged Hybrid (Teacher → Distill → Deploy)

## Overview

Layer 3 replaces the rule-based expression generator (Layer 2) with a learned model that maps EISV trajectories to hierarchical expression tokens. The system trains a high-quality teacher model on Mac, distills it into a micro-transformer for Pi deployment, and falls back to rules when the model is uncertain.

**Goals:**
- Richer expressivity beyond fixed affinity matrices
- Two publishable HuggingFace model artifacts
- End-to-end learned system replacing hand-crafted rules
- Built right over months with go/no-go gates

## Three-Phase Architecture

```
Phase 1: Teacher (Mac)          Phase 2: Student (Mac)         Phase 3: Deploy (Pi)
Llama-3.2-1B + LoRA    →    8M param micro-transformer   →    ONNX on Pi Zero 2W
Train on 30-50K examples      Distill from teacher              Confidence-gated fallback
                              ↓                                  ↓
                         Gate 1: >0.933 coherence          Gate 2: ≥90% teacher quality
                                                           Gate 3: 48h stability
```

## Phase 1: Teacher Model

**Base model:** `meta-llama/Llama-3.2-1B-Instruct`
- 1.24B params, instruction-tuned
- Fits on Mac with 8GB+ unified memory

**Input format:**
```
<|system|>You are Lumen's trajectory-to-expression mapper.
Given an EISV trajectory, output the appropriate expression tokens.

<|user|>
TRAJECTORY:
shape: rising_entropy
window: 20 states over 40s
eisv_mean: E=0.71, I=0.48, S=0.62, V=0.09
derivatives: dE=+0.02, dI=-0.03, dS=+0.08, dV=+0.01
second_derivatives: d2E=-0.01, d2I=+0.01, d2S=-0.02, d2V=0.00

<|assistant|>
EISV_TOKENS: ~curiosity~ ~ripple~ ~questioning~
LUMEN_TOKENS: why busy what
PATTERN: TRIPLE
```

**Hierarchical output:** EISV tokens (reasoning) → Lumen tokens (expression) → pattern class.

**Training data:**
- 21,449 real trajectory windows with Layer 2 rule-based labels
- Augmented with synthetic trajectories for underrepresented shapes
- Target: 30-50K total examples after augmentation
- Split: 80/10/10 train/val/test

**LoRA config:**
- Rank: 16, alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj
- Learning rate: 2e-4, warmup: 100 steps
- Epochs: 3-5, early stopping on validation loss

**Gate 1:** Coherence > 0.933 on held-out test set.

## Phase 2: Distilled Student (TrajectoryTransformer)

**Architecture:**
```
Input: [20 × 4] EISV time-series (numerical, no text serialization)
  ↓
Temporal Encoder (4 layers, 128d, 4 heads)
  - Learned positional encoding for time steps
  - Cross-attention between EISV dimensions
  - Output: trajectory embedding (128d)
  ↓
Shape Head (auxiliary classification)
  - Linear(128, 9) → shape logits
  ↓
EISV Token Decoder (2 layers, 128d)
  - Autoregressive, max 3 tokens
  - Vocabulary: 15 EISV tokens + [PAD] + [EOS]
  - Cross-attention to trajectory embedding
  ↓
Lumen Token Decoder (shared weights, different output head)
  - Linear(128, 15) → Lumen token logits
  - Vocabulary: 15 Lumen primitives + [PAD] + [EOS]
```

**Model size:** ~8M parameters → ~32MB FP32, ~8MB INT8 quantized

**Distillation training:**
- Soft label distillation: KL divergence, temperature τ=4
- Feature matching: student embedding ≈ teacher last hidden state
- Multi-task loss: `L = 0.7·L_distill + 0.1·L_shape + 0.2·L_hard_labels`

**Inference on Pi:**
- INT8 quantized via ONNX Runtime for ARM
- Expected: ~30MB RAM, 200-500ms per expression

**Gate 2:** Coherence ≥ 0.84 (90% of teacher) AND ≤100MB RAM.

## Phase 3: Pi Deployment

**Integration into anima-mcp:**

```
anima-mcp server loop (~60s)
  → TrajectoryAwareness.classify()
  → TrajectoryModel.infer(eisv_window)
  → if confidence ≥ 0.6: use model output
    else: fall back to Layer 2 rules
  → coherence scoring + feedback logging
```

**New module:** `src/anima_mcp/eisv/model.py`
- Loads ONNX from `~/.anima/models/trajectory_student.onnx`
- `TrajectoryModel.infer(eisv_window) -> dict`
- Graceful degradation: model failure → Layer 2 fallback

**Modified:** `awareness.py`
- Optional `TrajectoryModel` dependency
- Source tracking: "model" vs "rules" in trajectory_events

**Model delivery:** ONNX file in `~/.anima/models/`, versioned.

**Observability extensions:**
- `get_eisv_trajectory_state` gains: model_version, model_confidence, model_used_pct
- Display: `[M]` or `[R]` indicator on trajectory line
- trajectory_events: source field

**Gate 3:** primitive_feedback ≥ Layer 2 rates, inference < 2s, RSS < 200MB, 48h stable.

## Evaluation

**Metrics:**

| Metric | Description |
|--------|-------------|
| Coherence | Overlap between predicted and shape-affine tokens |
| Pattern accuracy | Correct structural pattern prediction |
| Shape prediction | Auxiliary shape classification accuracy (student) |
| Diversity | Shannon entropy of token distributions |
| Bridge fidelity | EISV→Lumen translation matches model's direct Lumen output |
| Temporal consistency | Expression stability across consecutive same-shape windows |
| Pi latency | Inference time on Pi Zero 2W |

**Evaluation table (targets):**

| Metric | Random | Rules | Rules+FB | Teacher | Student |
|--------|--------|-------|----------|---------|---------|
| Coherence | 0.265 | 0.503 | 0.933 | >0.933 | >0.84 |
| Pattern accuracy | ~20% | — | — | TBD | TBD |
| Diversity | TBD | TBD | TBD | TBD | TBD |
| Pi latency | — | <10ms | <10ms | N/A | <2000ms |

## Publishable Artifacts

1. **`hikewa/eisv-trajectory-teacher`** — Llama-3.2-1B + LoRA adapter (HuggingFace)
2. **`hikewa/eisv-trajectory-student`** — TrajectoryTransformer 8M + ONNX (HuggingFace)
3. **`hikewa/unitares-eisv-trajectories`** — Training dataset (blocked on HF auth)

**Paper contributions:**
1. Thermodynamic trajectory conditioning for embodied expression
2. Knowledge distillation from rule-based to neural system
3. Edge deployment of micro-transformer on Pi Zero 2W
4. Hierarchical expression generation with intermediate reasoning tokens

## File Structure

```
eisv_lumen/training/
├── data_prep.py           # Format trajectories for teacher training
├── teacher_train.py       # LoRA fine-tuning on Llama-3.2-1B
├── teacher_eval.py        # Evaluate teacher model
├── student_model.py       # TrajectoryTransformer architecture
├── distill.py             # Knowledge distillation pipeline
├── student_eval.py        # Evaluate student model
├── export_onnx.py         # Export student to ONNX
└── configs/
    ├── teacher_lora.yaml  # LoRA hyperparameters
    └── student.yaml       # Student architecture config

anima-mcp (Phase 3):
├── src/anima_mcp/eisv/model.py  # ONNX inference wrapper
└── tests/test_eisv_model.py     # Model integration tests
```
