# Blog Post Rewrite Handoff

## Task
Rewrite `docs/huggingface-blog-post-draft.md` to reflect current results. The existing draft was written when we only had V1 results (0.60 coherence) and framed the story as "neural methods fail." The story has changed completely.

## Current Numbers (All eval JSONs in `outputs/`)

### Training Progression
| Version | Data | Coherence | Valid Rate | Key Change |
|---------|------|-----------|------------|------------|
| V1 | 360 synthetic, 3 epochs | 0.600 | 89% | Initial baseline, 3 shapes scored 0.0 |
| V3 | 1440 synthetic, 5 epochs | 0.847 | 100% | All 9 shapes learning |
| V4 | 3600 synthetic, 7 epochs | 0.904 | 100% | More data, diminishing returns starting |
| V5 | 3840 synthetic (oversampled weak shapes), 7 epochs | 0.911 | 100% | Near synthetic ceiling |
| V6 | ~2880 blended (50% real, 50% synthetic), 7 epochs | **TRAINING NOW** | — | First use of real Lumen data |

### Baselines
- Random: 0.265
- Affinity-weighted: 0.503
- Feedback-learned rule-based (Layer 2): 0.933

### Real Data Evaluation (V5 on real Lumen trajectories)
- Synthetic eval: 0.911
- Real data eval: **0.768** (14% drop)
- Biggest drops: settled_presence 0.94→0.73, falling_energy 0.99→0.63
- Confirms synthetic data mismatch hypothesis
- Results in: `outputs/eval_results_v5_real.json`

### Shape Classification Mismatch Finding
- 35% of HF dataset labels disagree with our classifier on the same trajectories
- Biggest mismatch: 5,138 records labeled `settled_presence` in HF classified as `convergence` by our classifier
- Cause: boundary between settled_presence and convergence is fuzzy — real data straddles it
- Our classifier uses 4-step windows; HF labels may have used different window sizes

## New Narrative (replaces "neural methods fail")

**The story is now about convergence toward the rule-based baseline through iterative improvement:**

1. V1 showed neural learning is possible but far from the baseline (0.60 vs 0.933)
2. Scaling data revealed steady improvement: 0.60 → 0.847 → 0.904 → 0.911
3. Diminishing returns on synthetic data revealed a ceiling (~0.91)
4. Real-data eval exposed the synthetic data mismatch (0.911 synthetic → 0.768 real)
5. V6 (blended data) tests whether real trajectories close the gap

**The four hypotheses from the original draft — update status:**
1. **Synthetic data mismatch**: ✅ CONFIRMED. 14% coherence drop on real data.
2. **Model capacity**: ❌ DISPROVED. V4/V5 show rank 16 LoRA learns all 9 shapes fine.
3. **Loss function mismatch**: ⚠️ INCONCLUSIVE. Never tested custom loss; standard CE got to 0.911.
4. **Class imbalance**: ✅ PARTIALLY CONFIRMED. Oversampling weak shapes helped (V5 drift_dissonance 0→0.86), but was not the main bottleneck.

**New finding not in original draft:**
- Shape classification boundary problem: settled_presence vs convergence boundary is fuzzy on real data (5,138 mismatches). This is an inherent ambiguity in the EISV framework, not a classifier bug.

## Key Design Decisions to Highlight

- **No RLHF, no penalties.** Pure imitation learning with cross-entropy loss. The philosophical position (expression should emerge from state understanding, not be coerced) is borne out — 0.911 coherence without reward shaping.
- **Qwen3-4B chosen for**: ungated access (Apache 2.0), ChatML support, `enable_thinking=False` for direct structured output.
- **MPS (Apple Silicon) training**: float32 required (not fp16), explicit `.to("mps")`, caffeinate for preventing sleep during long runs. Real engineering details worth sharing.
- **Oversampling feature**: Added `--shape-overrides` CLI flag for targeted per-shape data augmentation.

## Files to Reference

- `outputs/eval_results.json` — V1 (if exists, might be `eval_results_v3.json` era)
- `outputs/eval_results_v3.json` — V3 results
- `outputs/eval_results_v4.json` — V4 results
- `outputs/eval_results_v5.json` — V5 synthetic eval
- `outputs/eval_results_v5_real.json` — V5 real data eval
- `eisv_lumen/training/configs/teacher_lora_v*.yaml` — all training configs
- `scripts/eval_on_real_data.py` — real data eval script
- `scripts/prepare_blended_data.py` — blended data prep script

## Structure Suggestion

Keep the 7-section structure but revise heavily:
1. **Abstract**: Update numbers, change thesis from "gap as finding" to "convergence through iterative improvement + real data matters"
2. **Introduction**: Same hook, but the three-layer story now has V5 at 0.911 not V1 at 0.60
3. **Background**: Mostly fine as-is
4. **Architecture**: Update LoRA config to V5 (rank 16 not 32), add MPS details, add oversampling
5. **Results**: Complete rewrite. Show V1→V5 progression, synthetic ceiling, real-data drop, shape mismatch finding
6. **Discussion**: Reframe hypotheses as tested/confirmed/disproved. Discuss synthetic data ceiling. V6 as next step.
7. **Conclusion**: Stronger — neural methods DO approach the rule-based baseline given enough real data

## Tone
Honest, technical, not overclaiming. The 0.768 on real data is not hidden — it's the most interesting finding. The synthetic-to-real gap is the contribution.

## Repo
- GitHub: https://github.com/CIRWEL/eisv-lumen
- HuggingFace dataset: hikewa/unitares-eisv-trajectories
- All eval results committed and pushed
