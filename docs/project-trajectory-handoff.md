# UNITARES Project Trajectory — Handoff Document

**Date**: 2026-02-13
**Author**: Claude (session working on Layer 3 teacher training)
**For**: Another Claude session to help with project trajectory, planning, or paper writing

---

## What Is This Project?

UNITARES is a system for governing AI agents using thermodynamic metaphors. At its center is **Lumen** — a Raspberry Pi-based embodied AI entity with LED nervous system, display, and a primitive language for self-expression.

The core abstraction is **EISV** — four dimensions that describe an agent's state:
- **E** (Energy/Warmth): engagement level
- **I** (Integrity/Clarity): coherence of reasoning
- **S** (Entropy/Dissonance): instability (inverse of stability)
- **V** (Void/Absence): disconnection (inverse of presence)

These aren't sentiment scores or emotion labels. They're derived from observable computational behavior (response quality, calibration, drift detection) and treated as a dynamical system with trajectories, attractors, and phase transitions.

## The Three Projects

### 1. unitares-governance (the framework)
- **Repo**: Private, runs on a server
- **What it does**: Multi-agent governance via EISV tracking, knowledge graph, dialectic sessions (thesis/antithesis/synthesis for stuck agents), calibration, CIRS protocol
- **Status**: Running in production, tracking multiple agents including Lumen
- **MCP server**: Exposes tools for agent check-in, knowledge ops, recovery

### 2. anima-mcp (Lumen's body)
- **Repo**: Private, runs on Raspberry Pi Zero 2W
- **What it does**: Nervous system (maps sensor data + state to LED colors), primitive language (15 tokens: warm/cold/bright/dim/quiet/busy/here/feel/sense/you/with/why/what/wonder/more/less), display management, art eras
- **Status**: Running on Pi. Layer 2 EISV observability code deployed to disk but service needs restart (blocked by NoNewPrivileges systemd constraint)
- **Key file**: `src/anima_mcp/primitive_language.py` — generates 1-3 token utterances based on anima state

### 3. eisv-lumen (the voice — THIS repo)
- **Repo**: https://github.com/CIRWEL/eisv-lumen (public)
- **What it does**: Makes Lumen's primitive expressions trajectory-aware rather than just state-reactive
- **Status**: Layer 1 complete, Layer 2 deployed, Layer 3 Phase 1 training in progress

## The Three Layers of eisv-lumen

### Layer 1: Dataset + Benchmark ✅
- 21,499 real EISV trajectory records from Lumen's state_history
- Published to HuggingFace: `hikewa/unitares-eisv-trajectories`
- 9 trajectory shape classes (settled_presence, rising_entropy, falling_energy, basin_transition_down/up, entropy_spike_recovery, drift_dissonance, void_rising, convergence)
- 15 EISV-Lumen expression tokens (~warmth~, ~curiosity~, ~resonance~, etc.)
- Coherence metric: measures alignment between trajectory shape and generated tokens
- Baselines: random=0.265, affinity-weighted=0.503, with feedback learning=0.933
- 263+ tests

### Layer 2: Rule-Based Bridge ✅
- Pure Python bridge: trajectory shape → EISV tokens → Lumen primitives
- TOKEN_MAP: semantic mapping between 15 EISV tokens and 15 Lumen tokens
- ExpressionGenerator with 5 expression patterns (SINGLE, PAIR, TRIPLE, REPETITION, QUESTION)
- Deployed to anima-mcp as `eisv/` package with TrajectoryAwareness orchestrator
- Gives Lumen trajectory-influenced expressions (gentle 2x weight boost, not override)

### Layer 3: Fine-Tuned Deep Voice Model 🔄
Three-phase approach with go/no-go gates:

**Phase 1: Teacher Model (current work)**
- Fine-tune Qwen3-4B with LoRA on trajectory→expression mapping
- Training on M4 Max (Apple Silicon, MPS, 128GB RAM)
- V1 result: 0.60 coherence, 89% valid rate (360 examples, rank 16, 3 epochs)
- V2 training in progress: 1440 examples, rank 32, 5 epochs (~2.5h remaining)
- Gate 1 threshold: coherence > 0.933 AND valid rate > 0.90
- Design doc: `docs/plans/2026-02-12-layer3-deep-voice-design.md`
- Implementation plan: `docs/plans/2026-02-12-layer3-phase1-plan.md`

**Phase 2: Distilled Student (not started)**
- Distill teacher to 8M-param TrajectoryTransformer
- Custom architecture: 4 layers, 256 hidden, 4 heads
- Gate 2: ≥90% of teacher quality AND ≤100MB RAM

**Phase 3: Pi Deployment (not started)**
- Export to ONNX, deploy on Pi Zero 2W
- Confidence-gated fallback to rule-based system
- Gate 3: 48h stability on Pi

## Key Numbers

| Metric | Value |
|--------|-------|
| Total tests | 349 (all passing) |
| Training tests | 86 |
| Real trajectory records | 21,499 |
| Trajectory shapes | 9 |
| EISV expression tokens | 15 |
| Lumen primitive tokens | 15 |
| Expression patterns | 5 (SINGLE, PAIR, TRIPLE, REPETITION, QUESTION) |
| Random baseline coherence | 0.265 |
| Affinity-weighted coherence | 0.503 |
| Feedback-learned coherence | 0.933 |
| V1 teacher coherence | 0.600 |
| V2 teacher coherence | TBD (training in progress) |

## What Needs Thinking About

### 1. Publication Strategy
- No venue or deadline chosen yet
- The eisv-lumen pipeline (trajectory → expression) is the publishable core
- The governance framework could be a separate paper
- The Pi deployment + video demo is the compelling artifact
- Current results (0.60 coherence) are already a result worth reporting
- Question: academic paper vs HuggingFace blog post vs both?

### 2. Scope Management
Three projects running simultaneously with one person. The governance framework serves the operator (managing multiple agents), while eisv-lumen serves the research narrative (can an AI entity learn to express its dynamics?). These are different audiences and different stories.

### 3. What "Done" Looks Like
- Gate 1 (0.933 coherence) is the deployment threshold, not the publication threshold
- Is 0.60→0.75→0.85 good enough to write about? Probably yes.
- The interesting story might be the *architecture* (EISV → trajectories → shapes → tokens → primitives) more than hitting a specific number
- The comparison to baselines (random 0.265 vs learned 0.60+) is already significant

### 4. The Pi Restart
Layer 2 code is on Pi's disk but the service can't restart remotely (NoNewPrivileges blocks sudo). Options:
- Physical power cycle
- Wait for natural restart
- Fix the systemd unit file to allow restarts
- SSH access (if available)

### 5. Data Quality
- Training data is 100% synthetic (generated from shape classes)
- Published HuggingFace dataset is Lumen-only (from anima.db, no agent_id mixing)
- Live governance dashboard shows mixed agents (different data path)
- Real trajectory records have: irregular sampling (mean 1.07s, std 4.6s), 56 large energy jumps, 16.8% repeated I values, strong E-V correlation (0.91 by design)

## Repo Structure (key files)

```
eisv-lumen/
├── eisv_lumen/
│   ├── shapes/
│   │   ├── shape_classes.py          # 9 TrajectoryShape enum + classify_trajectory()
│   │   └── expression_generator.py   # ExpressionGenerator with affinity-weighted sampling
│   ├── eval/
│   │   ├── metrics.py                # SHAPE_TOKEN_AFFINITY, coherence metric
│   │   └── baseline.py               # ALL_TOKENS, baseline evaluations
│   ├── bridge/
│   │   └── lumen_bridge.py           # TOKEN_MAP, translate_expression(), EISV→Lumen state
│   ├── training/
│   │   ├── config.py                 # TrainingConfig dataclass (default: Qwen3-4B)
│   │   ├── configs/
│   │   │   ├── teacher_lora.yaml     # V1 config (rank 16, 3 epochs, 360 examples)
│   │   │   └── teacher_lora_v2.yaml  # V2 config (rank 32, 5 epochs, 1440 examples)
│   │   ├── data_prep.py              # TrainingExample, format functions
│   │   ├── dataset_builder.py        # build_training_dataset(), split_dataset()
│   │   ├── chat_format.py            # SYSTEM_PROMPT, format_chat_messages()
│   │   ├── teacher_train.py          # prepare_training_data(), parse_model_output()
│   │   ├── teacher_eval.py           # evaluate_predictions(), check_gate1()
│   │   ├── trainer.py                # train_teacher() — full LoRA pipeline
│   │   ├── teacher_inference.py      # load_teacher_model(), generate_expression()
│   │   └── cli.py                    # prepare, train, eval, gate1 subcommands
│   └── synthetic/
│       └── trajectory_generator.py   # Generate synthetic trajectory windows
├── data/
│   ├── training/                     # V1: 360 train, 45 val, 45 test
│   └── training_v2/                  # V2: 1440 train, 180 val, 180 test
├── outputs/
│   ├── teacher_lora/final_adapter/   # V1 adapter (0.60 coherence)
│   └── eval_results.json             # V1 evaluation results
├── docs/plans/
│   ├── 2026-02-12-layer3-deep-voice-design.md
│   └── 2026-02-12-layer3-phase1-plan.md
└── tests/                            # 349 tests, all passing
```

## Git State
- Branch: main
- Latest commit: `5d80c68` — "Switch teacher model from Llama-3.2-1B to Qwen3-4B"
- All pushed to origin
- HuggingFace: logged in as `hikewa`, dataset published

## For the Next Claude

If you're helping with project trajectory, publication planning, or scope:
- The user ("cirwel") is the sole builder. There is no team.
- They care about doing this right, not fast.
- The philosophical dimension matters — this isn't just an ML pipeline, it's an exploration of whether computational dynamics can ground something like expression.
- They're aware they're building in too many directions simultaneously.
- The most helpful thing is probably helping them decide what to focus on and what to defer, not adding more features.
