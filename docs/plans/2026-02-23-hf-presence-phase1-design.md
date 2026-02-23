# HF Presence Phase 1: Dataset-First Foundation

**Date**: 2026-02-23
**Goal**: Establish hikewa's Hugging Face presence by completing the EISV-Lumen story — models, interactive demo, and blog post.

## Context

- Account `hikewa` is authenticated but has no published repos or Spaces
- Dataset `hikewa/unitares-eisv-trajectories` already exists (21,449 real trajectories)
- Teacher model (LoRA v6, 0.952 coherence) and student models are trained but not published
- An 884-line blog post draft exists at `docs/huggingface-blog-post-draft.md`

## Deliverables

### 1. Teacher Model: `hikewa/eisv-lumen-teacher`

Upload the V6 LoRA adapter as a PEFT model.

**Files to upload** (from `outputs/teacher_lora_v6/final_adapter/`):
- `adapter_model.safetensors` (126 MB)
- `adapter_config.json`
- Tokenizer files (tokenizer.json, vocab.json, merges.txt)
- Model card README.md

**Model card contents**:
- EISV overview: 4 dimensions (Energy, Information Integrity, Entropy, Void)
- Task: Map EISV trajectory shapes to primitive expression tokens
- Training: LoRA r=16, alpha=32 on Qwen/Qwen3-4B, 1260 steps, 2880 examples (50/50 real+synthetic)
- 9 trajectory shapes, 15 expression tokens, 5 structural patterns
- Results: 0.952 mean coherence on 500 real trajectories
- Per-shape coherence breakdown
- Usage example with peft + transformers
- Links: dataset, Space, blog post

### 2. Student Model: `hikewa/eisv-lumen-student`

Upload the "small" student — best coherence (0.986) and Pi-deployable.

**Files to upload** (from `outputs/student_small/`):
- sklearn pickle files (pattern_clf.pkl, token1_clf.pkl, token2_clf.pkl, scaler.pkl, encoders)
- Exported JSON files (zero-dependency, ~1.5 MB total)
- `student_inference.py` (standalone inference module)
- Model card README.md

**Model card contents**:
- Distillation approach: teacher-generated labels -> RandomForest ensemble
- 3 classifiers: pattern (5 classes), token1 (15 classes), token2 (15+none classes)
- 12 numeric input features (EISV means, derivatives, second derivatives)
- Results: 0.986 coherence, 0.688 token1 agreement, 0.652 pattern agreement
- Pi deployment: JSON export runs on stdlib Python only
- Comparison table: teacher (0.952) vs student-small (0.986) vs baseline (0.495)
- Usage examples for both sklearn and JSON inference paths

### 3. Interactive Space: `hikewa/eisv-lumen-explorer`

Gradio app with three tabs.

**Tab 1: Trajectory Explorer**
- Load sample trajectories from the published dataset
- Matplotlib plot: EISV dimensions over time (4 colored lines)
- Show classified shape label, derivative summary
- Shape distribution chart

**Tab 2: Expression Generator**
- Select a trajectory shape from dropdown (or pick random)
- Optionally adjust EISV values with sliders
- Run Layer 2 rule-based generator
- Show: selected pattern, token(s), affinity weights that drove the selection
- Regenerate button to see stochastic variation

**Tab 3: Model Comparison**
- Same trajectory input
- Side-by-side output: rule-based vs student model
- Coherence score for each
- (Teacher inference optional — requires GPU, can show cached results instead)

**Tech stack**: Gradio, matplotlib, numpy. Rule-based generator + student JSON inference run in-process. No GPU required. Free HF Space tier.

### 4. Blog Post

Publish the existing draft (`docs/huggingface-blog-post-draft.md`) to HF blog.

**Pre-publish tasks**:
- Review and update any outdated metrics (ensure V6 numbers are used)
- Add links to the newly published model repos and Space
- Verify citations and references
- Submit via HF blog submission process

## Cross-Linking Strategy

All four items link to each other:
- Dataset card -> models, Space, blog
- Teacher model card -> dataset, student, Space, blog
- Student model card -> dataset, teacher, Space, blog
- Space -> dataset, both models, blog
- Blog -> everything

## What's NOT in Phase 1

- Governance MCP framework (separate project, later phase)
- Lumen's art gallery / drawing system
- Live system integration (Pi real-time data)
- Opening source on governance-mcp-v1 or anima-mcp repos

## Success Criteria

- All 4 items published and cross-linked
- Space loads and is interactive on free tier
- Model cards have usage examples that work
- Blog post accepted/published on HF
