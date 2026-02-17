# EISV-Lumen

[![Tests](https://img.shields.io/badge/tests-263%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-orange)](LICENSE)
[![Dataset](https://img.shields.io/badge/HuggingFace-unitares--eisv--trajectories-yellow)](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories)

**Dynamics-emergent voice and governance benchmark for embodied AI.**

EISV-Lumen is a rule-based, fully interpretable system that generates primitive expressions from thermodynamic governance trajectories. Rather than training a neural network to produce agent utterances, it classifies continuous EISV (Energy, Information Integrity, Entropy, Void) dynamics into 9 trajectory shape classes and uses affinity-weighted token sampling to produce contextually coherent expressions. Evaluated on 21,449 real trajectory records from [Lumen](https://github.com/CIRWEL/eisv-lumen) -- an embodied AI agent running on a Raspberry Pi within the [UNITARES](https://github.com/CIRWEL) governance framework -- the system achieves 0.503 coherence (beating random baseline by 23.8 percentage points) without any learned parameters, and 0.933 coherence with an online feedback loop.

---

## Quick Start

```bash
# Clone
git clone https://github.com/CIRWEL/eisv-lumen.git
cd eisv-lumen

# Install
pip install -e ".[dev]"

# Run tests
pytest

# Run full evaluation (requires anima.db)
python3 -m eisv_lumen.scripts.full_evaluation /path/to/anima.db
```

The evaluation script produces a JSON report with shape distribution, baseline comparisons, expression generator coherence, feedback-loop improvement, and a go/no-go gate decision.

---

## Architecture

EISV-Lumen is structured as a three-layer system. This release covers Layer 1 and the core of Layer 2.

```
Layer 3 (future)   Fine-tuned Deep Voice (LoRA on Qwen3-4B)
                         |
Layer 2 (this)     Dynamics-Emergent Primitive Voice (rule-based, interpretable)
                         |
Layer 1 (this)     Dataset + Benchmark + Evaluation Framework
                         |
                   Real Lumen Data (anima.db: 214,503 state snapshots)
```

**Layer 1** extracts EISV time-series from Lumen's SQLite database, computes finite-difference derivatives, assembles sliding-window trajectory records, and classifies each window into one of 9 dynamical shape classes. The result is a benchmark dataset published on HuggingFace.

**Layer 2** is the primary research contribution: a dynamics-emergent expression generator that maps trajectory shapes to primitive token expressions through shape-driven pattern selection and affinity-weighted sampling -- no gradient descent, no learned embeddings, fully inspectable.

**Layer 3** (planned) will fine-tune a language model on the trajectory-expression pairs produced by Layers 1-2.

---

## EISV Dimensions

The EISV framework maps Lumen's continuous anima state to four governance dimensions:

| Dimension | Symbol | Range | Lumen Mapping | Description |
|-----------|--------|-------|---------------|-------------|
| Energy | E | [0, 1] | warmth | Productive capacity; couples toward I, reduced by entropy cross-coupling |
| Information Integrity | I | [0, 1] | clarity | Signal fidelity; boosted by coherence, reduced by entropy |
| Entropy | S | [0, 1] | 1 - stability | Semantic uncertainty; decays naturally, rises with complexity and drift |
| Void | V | [0, 0.3] | (1 - presence) * 0.3 | Absence of engagement (scaled inverse presence) |

All dimensions are continuous and computed at every state snapshot. First and second derivatives are computed via finite differences over sliding windows (assembler default: window_size=10, stride=5; dataset publisher uses window_size=20, stride=10).

---

## Trajectory Shape Classes

Each trajectory window is classified into one of 9 dynamical shapes using a priority-ordered rule-based classifier. Rules are applied in sequence; the first match determines the class.

| Shape | Description | Real Data % | Distinguishing Feature |
|-------|-------------|-------------|----------------------|
| settled_presence | Stable state, low variance | 47.19% | All derivatives near zero, system at attractor |
| convergence | Approaching equilibrium | 41.45% | Small derivatives and second derivatives, nonzero dynamics |
| entropy_spike_recovery | Entropy spike then recovery | 4.91% | Entropy range >= 0.2 with interior maximum |
| basin_transition_up | Sharp energy increase | 1.71% | Energy range >= 0.2, positive mean dE |
| rising_entropy | Entropy increasing | 1.49% | Mean dS > 0.05 |
| basin_transition_down | Sharp energy decrease | 1.47% | Energy range >= 0.2, negative mean dE |
| falling_energy | Energy declining | 1.45% | Mean dE < -0.05 |
| void_rising | Void state increasing | 0.34% | Mean dV > 0.05 |
| drift_dissonance | Sustained integrity fluctuation | 0% | Ethical drift > 0.3 (synthetic only) |

8 of 9 shapes are observed in real Lumen data. The `drift_dissonance` shape has not yet been observed organically and is represented only through synthetic augmentation.

---

## Expression Generator

The dynamics-emergent expression generator is the primary research contribution. It produces primitive expressions from trajectory shapes using three mechanisms:

### 1. Shape-Driven Pattern Selection

Each trajectory shape has a probability distribution over 5 structural patterns:

| Pattern | Example | Description |
|---------|---------|-------------|
| SINGLE | `~stillness~` | One token |
| PAIR | `~stillness~ ~holding~` | Two distinct tokens |
| TRIPLE | `~stillness~ ~holding~ ~resonance~` | Three distinct tokens |
| REPETITION | `~stillness~ ~stillness~` | One token repeated |
| QUESTION | `~warmth~ ~questioning~` | Ends with inquiry token |

For example, `settled_presence` favors SINGLE (0.4) and PAIR (0.3) patterns, while `rising_entropy` favors TRIPLE (0.3) and QUESTION (0.3) patterns.

### 2. Affinity-Weighted Token Sampling

Each shape has an affinity set of semantically coherent tokens. Affine tokens receive sampling weight 3.0; all others receive 1.0. This produces contextually appropriate expressions without hard constraints:

| Shape | Affine Tokens |
|-------|--------------|
| settled_presence | ~stillness~, ~holding~, ~resonance~, ~deep_listening~ |
| rising_entropy | ~ripple~, ~emergence~, ~questioning~, ~curiosity~ |
| convergence | ~stillness~, ~resonance~, ~return~, ~deep_listening~ |
| entropy_spike_recovery | ~ripple~, ~return~, ~holding~, ~reflection~ |
| basin_transition_up | ~emergence~, ~reaching~, ~warmth~, ~return~ |
| basin_transition_down | ~releasing~, ~threshold~, ~boundary~ |
| falling_energy | ~releasing~, ~stillness~, ~boundary~, ~reflection~ |
| void_rising | ~reaching~, ~curiosity~, ~questioning~, ~threshold~ |
| drift_dissonance | ~boundary~, ~questioning~, ~reflection~ |

### 3. Feedback-Driven Weight Updates

An online learning loop adjusts token weights based on coherence scores:

```
reward = (score - 0.5) * 2.0      # maps [0, 1] -> [-1, 1]
weight += 0.08 * reward            # learning rate = 0.08
weight = clamp(weight, 0.1, 10.0)  # bounded update
```

This allows the system to adapt over time while remaining fully interpretable -- every weight is a readable float, every decision rule can be inspected.

### Primitive Token Vocabulary

The system uses 15 primitive tokens:

```
~warmth~  ~curiosity~  ~resonance~  ~stillness~  ~boundary~
~reaching~  ~reflection~  ~ripple~  ~deep_listening~  ~emergence~
~questioning~  ~holding~  ~releasing~  ~threshold~  ~return~
```

---

## Evaluation Results

Full evaluation on real Lumen data (21,449 trajectory records from 214,503 state snapshots, 921 primitive expressions):

### Coherence Scores

| Condition | Mean Coherence | Description |
|-----------|---------------|-------------|
| Random baseline | 0.265 | Uniform random token selection (lower bound) |
| Prompt-conditioned | ~0.70 | Simulated LLM with 70/30 affine/random mix |
| Shape-matched oracle | 1.000 | Always picks affine tokens (upper bound) |
| **Expression generator** | **0.503** | Rule-based, no feedback |
| **With feedback loop** | **0.933** | Online weight updates, near-oracle |

### Key Numbers

- **263 tests**, all passing
- **21,499 total trajectory records** (21,449 real + 50 synthetic)
- **8 of 9** trajectory shapes observed in real data
- **+23.8pp** over random baseline (without feedback)
- **+43.0pp** improvement from feedback loop (0.503 -> 0.933)
- **Go/no-go gate: GO** -- all three criteria passed:
  - Beats random by > 5pp
  - At least 3 distinct shapes observed
  - Feedback improves over no-feedback

---

## Dataset

The trajectory dataset is published on HuggingFace:

**[hikewa/unitares-eisv-trajectories](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories)**

### Schema

| Column | Type | Description |
|--------|------|-------------|
| shape | string | Trajectory shape class label |
| eisv_states | string (JSON) | Time-series of EISV state vectors |
| derivatives | string (JSON) | First derivatives of EISV dimensions |
| t_start | float | Start time of the trajectory window |
| t_end | float | End time of the trajectory window |
| provenance | string | Data source (`"lumen_real"` or `"synthetic"`) |
| tokens | string (JSON) | Expression token lists aligned to the trajectory |
| n_expressions | int | Number of aligned primitive expressions |

### Loading

```python
from datasets import load_dataset

ds = load_dataset("hikewa/unitares-eisv-trajectories")
```

---

## Bridge to Lumen

The `bridge/` module connects EISV-Lumen's trajectory-derived expressions to [Lumen's](https://github.com/CIRWEL/eisv-lumen) live primitive language system. Lumen uses 16 primitive tokens across 5 categories (STATE, PRESENCE, RELATIONAL, INQUIRY, CHANGE). The bridge provides:

1. **Token translation** -- maps each EISV-Lumen token to Lumen primitives (e.g., `~warmth~` -> `warm, feel`)
2. **State conversion** -- converts EISV vectors to Lumen anima states (warmth, clarity, stability, presence)
3. **Trigger hints** -- maps trajectory shapes to generation triggers with suggested token counts

```python
from eisv_lumen.bridge.lumen_bridge import generate_lumen_expression

result = generate_lumen_expression(
    shape="rising_entropy",
    eisv_state={"E": 0.7, "I": 0.5, "S": 0.8, "V": 0.1},
)
# result["eisv_tokens"]  -> ["~ripple~", "~emergence~", "~curiosity~"]
# result["lumen_tokens"] -> ["busy", "more", "why"]
# result["lumen_state"]  -> {"warmth": 0.7, "clarity": 0.5, ...}
```

---

## Project Structure

```
eisv-lumen/
├── pyproject.toml                        # Package config, dependencies
├── eisv_lumen/
│   ├── __init__.py
│   ├── extract/                          # Data extraction layer
│   │   ├── lumen_states.py               #   State history + EISV mapping
│   │   ├── lumen_expressions.py          #   Primitive expression history
│   │   ├── derivatives.py                #   Finite-difference EISV derivatives
│   │   ├── governance_data.py            #   Governance trajectory extraction
│   │   └── assembler.py                  #   Dataset assembly pipeline
│   ├── shapes/                           # Trajectory classification
│   │   ├── shape_classes.py              #   9 shape classes + rule-based classifier
│   │   └── expression_generator.py       #   Dynamics-emergent voice (primary contribution)
│   ├── eval/                             # Evaluation framework
│   │   ├── metrics.py                    #   Coherence, diversity, accuracy metrics
│   │   └── baseline.py                   #   3 baseline conditions (random, matched, prompted)
│   ├── synthetic/                        # Data augmentation
│   │   └── trajectory_generator.py       #   Synthetic trajectories for all 9 shapes
│   ├── bridge/                           # Integration layer
│   │   └── lumen_bridge.py               #   EISV-Lumen <-> Lumen primitive bridge
│   ├── publish/                          # Publishing
│   │   └── hf_dataset.py                 #   HuggingFace format + dataset card
│   └── scripts/                          # CLI tools
│       ├── full_evaluation.py            #   Full evaluation + go/no-go gate
│       └── publish_dataset.py            #   Dataset publisher
├── tests/                                # 263 tests
└── docs/plans/                           # Design + implementation plans
```

---

## Running the Scripts

### Full Evaluation

```bash
# With default anima.db path
python3 -m eisv_lumen.scripts.full_evaluation

# With custom path
python3 -m eisv_lumen.scripts.full_evaluation /path/to/anima.db
```

Outputs a JSON report to stdout with:
- Data summary (state counts, expression counts, trajectory windows)
- Shape distribution with percentages
- Baseline coherence scores (random, shape-matched, prompt-conditioned)
- Expression generator coherence (with and without feedback)
- Go/no-go gate decision

### Dataset Publication

```bash
# Dry run (validate without publishing)
python3 -m eisv_lumen.scripts.publish_dataset --dry-run

# Publish to HuggingFace
python3 -m eisv_lumen.scripts.publish_dataset --repo-id hikewa/unitares-eisv-trajectories

# Custom settings
python3 -m eisv_lumen.scripts.publish_dataset \
  --db-path /path/to/anima.db \
  --min-per-shape 50 \
  --repo-id hikewa/unitares-eisv-trajectories
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=eisv_lumen

# Run only unit tests (no anima.db required)
pytest -m "not integration"

# Run integration tests (requires real anima.db)
pytest -m integration
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Citation

```bibtex
@misc{eisv_lumen_2026,
  title   = {EISV-Lumen: Dynamics-Emergent Voice and Governance Benchmark},
  author  = {hikewa},
  year    = {2026},
  url     = {https://github.com/CIRWEL/eisv-lumen},
  note    = {Rule-based trajectory-to-expression system achieving 0.503 coherence
             (0.933 with feedback) on 21,449 real Lumen trajectories.
             Part of the UNITARES governance framework.}
}
```

---

*EISV-Lumen is part of the [UNITARES](https://github.com/CIRWEL) framework for thermodynamic AI governance.*
