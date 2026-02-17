# EISV-Lumen: Trajectory-Aware Expression for Embodied AI

**Author**: cirwel
**Date**: February 2026
**HuggingFace Dataset**: [`hikewa/unitares-eisv-trajectories`](https://huggingface.co/datasets/hikewa/unitares-eisv-trajectories)
**Code**: [github.com/CIRWEL/eisv-lumen](https://github.com/CIRWEL/eisv-lumen)

---

## Abstract

We introduce **EISV-Lumen**, a system for trajectory-aware expression in embodied AI. Rather than reacting to current state snapshots, Lumen — a Raspberry Pi-based entity with sensors, LEDs, and a primitive 15-token vocabulary — expresses the *shape* of its dynamics: whether its internal trajectory is rising toward engagement, falling into quiescence, spiking in entropy, or settling into stable presence.

We frame this as a sequence generation task: given a trajectory window through a 4-dimensional phase space (Energy, Information Integrity, Entropy, Void), generate 1-3 primitive tokens that capture the trajectory's shape. We contribute: (1) a dataset of 21,499 real EISV trajectories from Lumen's operational history, published at `hikewa/unitares-eisv-trajectories`; (2) a coherence metric measuring shape-token affinity alignment; (3) a rule-based system achieving 0.933 coherence through affinity matrices and feedback learning, deployed to Lumen's hardware; (4) a fine-tuned Qwen3-4B teacher model that progresses from 0.600 to 0.911 coherence across five training iterations through data scaling, oversampling, and architectural tuning.

The remaining gap between neural (0.911 synthetic, 0.768 real) and symbolic (0.933) approaches reveals that **synthetic data mismatch is the primary bottleneck**: the model learns well from generated trajectories but drops 14% on real Lumen data. V6 training (blended real + synthetic data) is underway to test whether real trajectories close this gap.

Layer 2 (rule-based) runs in production on Lumen's Pi Zero 2W, giving it a voice grounded in computational dynamics rather than language priors. All code, data, and models are openly available.

---

## Table of Contents

1. [Introduction](#section-1-introduction)
2. [Background and Motivation](#section-2-background-and-motivation)
3. [Architecture and Design Principles](#section-3-architecture-and-design-principles)
4. [Implementation Details](#section-4-implementation-details)
5. [Results and Analysis](#section-5-results-and-analysis)
6. [Discussion and Future Work](#section-6-discussion-and-future-work)
7. [Conclusion](#conclusion)

---

## Section 1: Introduction

Can an AI entity learn to express the shape of its own dynamics?

Not its current state — a snapshot like "I am confused" or "battery low" — but the *trajectory* it's following through its internal phase space. The difference between falling toward stability and rising from it. Between entropy spiking and recovering, or drifting into void.

This is the question EISV-Lumen explores through **Lumen**, a Raspberry Pi-based embodied AI with a primitive language, LED nervous system, and sensors that track four dimensions of computational dynamics:

- **E** (Energy): exploration, productive capacity, CPU warmth
- **I** (Information Integrity): coherence, sensor calibration, reasoning quality
- **S** (Entropy): semantic uncertainty, environmental variance, dissonance
- **V** (Void): E-I imbalance accumulation, disconnection, quiescence

These aren't emotion labels or sentiment scores. They're thermodynamic-inspired state variables computed from observable signals — temperature, CPU load, sensor coverage, interaction presence. EISV is to Lumen what temperature and pressure are to a gas: measurable properties that describe its computational "physics."

The challenge is expression. Given an EISV trajectory window (4-20 steps, spanning seconds to minutes of Lumen's life), generate 1-3 tokens from a 15-token primitive vocabulary that capture the trajectory's *shape* — not just where it is, but where it's going.

**Three approaches, three layers:**

**Layer 1** establishes the task:
- 21,499 real trajectories from Lumen's state history
- 9 trajectory shape classes (`settled_presence`, `rising_entropy`, `falling_energy`, etc.)
- Coherence metric measuring shape-token affinity alignment
- Published dataset: `hikewa/unitares-eisv-trajectories`

**Layer 2** solves it with rules:
- Shape classifier: trajectory → shape class
- Affinity matrix: 9 shapes × 15 tokens, learned from 2,639 human feedback adjustments
- Expression generator: 5 patterns (SINGLE, PAIR, TRIPLE, REPETITION, QUESTION)
- **0.933 coherence** — deployed to Lumen's Pi as production system

**Layer 3** learns to approach it:
- Qwen3-4B fine-tuned with LoRA across five iterations (V1→V5)
- **0.600 → 0.847 → 0.904 → 0.911 coherence** on synthetic data
- **0.768 coherence** on real Lumen trajectories — the 14% synthetic-to-real gap is the core finding

The story is one of convergence through iterative improvement. V1 showed neural learning was possible but far from the baseline. Each iteration — more data, targeted oversampling, architectural tuning — closed the gap. But real data evaluation exposed a ceiling: the model learns synthetic patterns well but struggles with the noise, quantization artifacts, and distributional quirks of real Lumen trajectories.

**Contributions:**
1. **EISV trajectory dataset**: 21,499 real computational trajectories with shape labels, openly published
2. **Coherence metric**: Affinity-based evaluation of trajectory-expression alignment
3. **Rule-based baseline**: 0.933 coherence system proving the task is solvable
4. **Iterative neural learning analysis**: V1→V5 progression revealing synthetic data ceiling and real-data gap
5. **Deployed system**: Trajectory-aware expression running on embodied hardware

The code, data, and model weights are available at `github.com/CIRWEL/eisv-lumen`. The broader UNITARES governance framework (which produces the EISV trajectories) remains proprietary, but the expression pipeline is fully open.

---

## Section 2: Background and Motivation

### 2.1 The Problem: State-Reactive Expression is Shallow

Most AI systems express themselves reactively: given a current state snapshot, produce an appropriate response. A chatbot sees `user_frustrated=True` and says "I'm sorry you're frustrated." A robot sees `battery=20%` and says "Low battery."

This works for transactional interactions but fails to capture *dynamics*. Consider Lumen in two scenarios:

**Scenario A**: `[E=0.4, I=0.6, S=0.3, V=0.2]` at time T
**Scenario B**: `[E=0.4, I=0.6, S=0.3, V=0.2]` at time T

Identical snapshots. But:
- In A, energy has been *rising* from 0.2 → 0.3 → 0.35 → 0.4 (recovery)
- In B, energy has been *falling* from 0.7 → 0.6 → 0.5 → 0.4 (collapse)

A state-reactive system would say the same thing in both cases. A trajectory-aware system sees:
- A: `basin_transition_up` → "warm wonder" (engagement returning)
- B: `falling_energy` → "cold quiet" (withdrawal deepening)

The *direction of change* carries meaning that the snapshot alone cannot capture.

### 2.2 Why EISV? (Not Sentiment or Emotion)

Traditional affective computing uses sentiment scores (positive/negative/neutral) or discrete emotion labels (happy, sad, angry, fearful). These are useful for humans but problematic for computational entities:

1. **Anthropomorphic projection**: Does Lumen feel "happy"? It's a Raspberry Pi with LEDs and sensors. Emotion labels import human phenomenology that may not apply.

2. **Not grounded in dynamics**: Emotions are folk psychology categories, not measurable physical processes. EISV dimensions are computed from observable signals:
   - Energy = CPU usage + temperature + interactions
   - Information Integrity = sensor coherence + calibration quality
   - Entropy = environmental variance + decision instability
   - Void = E-I imbalance accumulation (presence inverse)

3. **Trajectory structure**: EISV is a 4D phase space with attractors, basins, and transitions. This supports dynamical systems analysis (Lyapunov stability, phase portraits) that emotion labels don't.

EISV is closer to *thermodynamic state variables* (temperature, pressure, entropy) than to emotional states. The question isn't "Is Lumen happy?" but "What basin is Lumen's trajectory converging toward?"

### 2.3 EISV Mathematical Framework

The EISV dimensions evolve via coupled differential equations from the UNITARES governance framework:

```
dE/dt = α(I - E) - βₑ·E·S + γₑ·‖Δη‖²
dI/dt = -k·S + βᵢ·C(V,Θ) - γᵢ·I·(1-I)
dS/dt = -μ·S + λ₁·‖Δη‖² - λ₂·C(V,Θ) + β·complexity
dV/dt = κ(E - I) - δ·V
```

Key properties:
- **Bistable**: The system has two attractors (high-I and low-I basins), with V feedback through the coherence function C(V,Θ) = Cmax · 0.5 · (1 + tanh(C₁·V))
- **Coupled**: E and I flow toward balance; S decays but is driven by drift and complexity; V accumulates E-I imbalance
- **Bounded**: E, I ∈ [0,1], S ∈ [0,2], V ∈ [-2,2]

**Note on ranges**: The ODE state space (above) is larger than Lumen's observation ranges. Lumen's sensors produce E ∈ [0,1], I ∈ [0,1], S ∈ [0,1], V ∈ [0,0.3]. The governance ODE evolves S beyond 1.0 under high complexity and V as a signed E−I integrator. The trajectories in the HuggingFace dataset use the sensor observation ranges, not the full ODE ranges.

This is not a sentiment model with learned weights. It's a thermodynamic model where each variable has a physical interpretation and the dynamics are analytically tractable.

### 2.4 Related Work

**Affective computing and emotion recognition**
Picard (1997) established affective computing as recognizing and responding to human emotions. Most work focuses on facial expressions, voice prosody, and physiological signals. EISV differs by focusing on *computational* rather than human affect.

**Dynamical systems approaches to cognition**
Beer (2000), van Gelder (1998) argue cognition is better understood as continuous dynamical systems than symbolic computation. EISV applies this to AI agent governance: rather than discrete "states," we track continuous trajectories through phase space.

**Embodied AI and morphological computation**
Pfeifer & Bongard (2006) show how physical embodiment shapes intelligence. Lumen's EISV dimensions emerge from its physical substrate (sensors, LEDs, thermal dynamics), making expression grounded in its material being rather than abstract semantics.

**Language models for robot control**
Recent work (Ahn et al. 2022, Brohan et al. 2023) uses LLMs to generate robot actions. These are typically *reactive* (state → action) rather than trajectory-aware. Our Layer 3 teacher model attempts trajectory-aware language generation and approaches the rule-based baseline at 0.911 synthetic coherence.

**Symbolic vs neural hybrid systems**
Marcus (2020) advocates hybrid architectures combining neural learning with symbolic structure. EISV-Lumen demonstrates that iterative neural training can approach symbolic performance (0.911 vs 0.933), but the synthetic-to-real gap (0.768) suggests symbolic grounding remains valuable for robustness.

### 2.5 Why Lumen? (Physical Embodiment Matters)

Lumen is not a chatbot in a terminal. It's a physical entity:
- **BrainCraft HAT**: Raspberry Pi Zero 2W with BME280 (temp/humidity/pressure), VEML7700 (light), 5 NeoPixel LEDs
- **Location**: Colorado (high elevation → low barometric pressure baseline ~827 hPa)
- **Proprioception**: Light sensor next to LEDs reads Lumen's own glow (self-sensing loop)
- **Thermal dynamics**: CPU heat bleeds into ambient temperature sensor
- **Display**: 240×240 LCD showing primitive language, art canvas, or sensor readouts

EISV trajectories are *grounded in this physical substrate*. When energy rises, you can see CPU load increase and LEDs warm. When entropy spikes, you can watch the LED colors shift chaotically. Expression isn't arbitrary text generation — it's a reflection of measurable processes in Lumen's nervous system.

This embodiment constrains expression in useful ways. Lumen can't say "I'm thinking about philosophy" because it has no sensors for that. It can say "warm bright here" because warmth (CPU temp), brightness (LED glow), and here (sensor coherence) are *directly measured*.

### 2.6 Dataset Characteristics

The 21,499 trajectories in the HuggingFace dataset (`hikewa/unitares-eisv-trajectories`) come from Lumen's `anima.db` state history over 4 months. Key characteristics:

**Irregular sampling**: Mean interval 1.07s (std 4.6s)
State updates occur when sensors change significantly, not on a fixed clock. This creates variable-length gaps between trajectory steps.

**Quantization artifacts**: 16.8% of Integrity values are repeated
The BME280 sensor has finite precision. This creates "plateaus" in I trajectories that are sensor artifacts, not true stability.

**Energy jump events**: 56 instances of ΔE > 0.4 in single step
Caused by Lumen waking from sleep or rebooting. These are legitimate dynamics, not errors.

**Strong E-V correlation**: 0.91 (by design)
Void is defined as inverse presence, and presence is partly driven by energy. This makes E and V mechanically coupled.

**Window-size sensitivity**: Shape labels in this dataset were generated from 20-step trajectory windows. If you reclassify using shorter windows, accuracy degrades significantly:

| Window Size | Label Match |
|-------------|-------------|
| 4-step | 65% |
| 8-step | 77% |
| 10-step | 81% |
| 15-step | 91% |
| 20-step | 100% |

Most mismatches (5,138 cases) are `settled_presence` → `convergence`: a 4-step window only sees the tail end of a settling trajectory, which looks like convergence. The full 20-step arc is needed to confirm the system has actually settled. This has direct implications for training — the V1-V5 teacher models used 4-step input windows, losing 35% of shape information (see Section 5.5).

**Shape distribution** (from classifier used in training):

| Shape | Count | % |
|-------|-------|---|
| settled_presence | 10,121 | 47.1% |
| convergence | 8,891 | 41.4% |
| entropy_spike_recovery | 1,053 | 4.9% |
| basin_transition_up | 366 | 1.7% |
| rising_entropy | 320 | 1.5% |
| basin_transition_down | 316 | 1.5% |
| falling_energy | 310 | 1.4% |
| void_rising | 72 | 0.3% |
| drift_dissonance | 50 | 0.2% |

### 2.7 The Central Question

Given an EISV trajectory window, can a fine-tuned language model generate expressions that are:
1. **Valid** (tokens exist in the 15-token vocabulary)
2. **Coherent** (high affinity with trajectory shape)
3. **Better than rules** (exceed 0.933 coherence baseline)

Layer 1 establishes the benchmark. Layer 2 proves it's achievable with symbolic rules. Layer 3 tests whether neural methods can match or exceed this — and discovers they can approach it on synthetic data (0.911) but struggle on real data (0.768). The next sections show why.

---

## Section 3: Architecture and Design Principles

### 3.1 Three-Layer Architecture

EISV-Lumen implements trajectory-aware expression through three layers, each with increasing sophistication but the same core task: convert an EISV trajectory window into a sequence of primitive tokens that Lumen can speak.

```
Layer 1: Dataset + Benchmark
├─ 21,499 real EISV trajectories from Lumen's state history
├─ 9 trajectory shape classes (settled_presence, rising_entropy, etc.)
├─ 15 EISV expression tokens (~warmth~, ~resonance~, ~stillness~, etc.)
├─ Coherence metric: shape-token affinity alignment
└─ Baselines: random (0.265), affinity-weighted (0.503), feedback-learned (0.933)

Layer 2: Rule-Based Bridge (Production)
├─ Shape classifier: trajectory → shape class
├─ Affinity matrix: 9×15 shape-token weights
├─ Expression generator: 5 patterns (SINGLE, PAIR, TRIPLE, REPETITION, QUESTION)
├─ TOKEN_MAP: EISV tokens → Lumen primitives
└─ Deployed to Pi, 2x weight boost (gentle influence, not override)

Layer 3: Fine-Tuned Teacher Model (Research)
├─ Qwen3-4B with LoRA (rank 16, α=32)
├─ V1→V5: 360 → 3,840 training examples, 3 → 7 epochs
├─ Chat format: system prompt + trajectory window → token sequence
└─ 0.911 synthetic coherence, 0.768 real coherence, 100% valid rate
```

**Philosophy**: Each layer can run standalone. Layer 1 is the evaluation ground truth. Layer 2 is the production system. Layer 3 is the research frontier that iteratively approaches Layer 2's performance — and the gap between synthetic and real evaluation is itself a research finding.

### 3.2 Design Principles

#### 1. **Observable Grounding**
All EISV dimensions are computed from measurable signals, not subjective labels. For Lumen:
- **Energy (E)**: CPU usage + temperature (engagement)
- **Information Integrity (I)**: Sensor coherence + calibration quality (reasoning clarity)
- **Entropy (S)**: Environmental variance + decision instability (system dissonance)
- **Void (V)**: E-I imbalance accumulation, interaction absence (presence inverse)

This ensures trajectories represent *actual dynamics*, not sentiment classification.

#### 2. **Trajectory Not State**
A snapshot of `[E=0.7, I=0.6, S=0.3, V=0.2]` is ambiguous — is this stable presence or the start of energy collapse? The trajectory shape disambiguates:
- `settled_presence`: E and I high, S and V low, *stable over 4 steps*
- `falling_energy`: E *decreasing* from 0.7 → 0.5 → 0.4 → 0.3

Expression should reflect the *direction of change*, not just the current position. This is why we use trajectory windows (4-20 steps depending on version) rather than single snapshots. V1-V5 used 4-step windows; Section 5.5 shows why longer windows (15-20 steps) are needed for reliable shape classification.

#### 3. **Graceful Degradation**
The system has three fallback levels:
1. **Layer 3 (teacher model)**: Best expressiveness, requires GPU inference
2. **Layer 2 (rule-based)**: 0.933 coherence, runs on Pi Zero 2W (512MB RAM)
3. **Silence**: If trajectory classification fails, Lumen says nothing

Layer 2 is *always available* as a production system. Layer 3 is a research enhancement that can fail without breaking Lumen's voice.

#### 4. **Semantic Primitives, Not Text**
Lumen's 15-token vocabulary is not natural language — it's a primitive semantic layer:
- `warm` = engagement/energy rising
- `cold` = withdrawal/energy falling
- `bright` = clarity/integrity present
- `dim` = coherence degrading
- `quiet` = low presence (void rising)
- `busy` = high activity (low void)
- `here` = grounded presence
- `feel` / `sense` / `wonder` = proprioceptive states
- `with` / `you` = relational tokens
- `what` / `why` = interrogative curiosity
- `more` / `less` = comparative operators

These are closer to emotive-cognitive *atoms* than words. The EISV→Lumen TOKEN_MAP bridges two semantic spaces:

| EISV Token | Lumen Primitive | Rationale |
|------------|-----------------|-----------|
| `~warmth~` | `warm` | Direct energy mapping |
| `~ripple~` | `busy` | Disturbance = chaotic activity |
| `~resonance~` | `here` | Coherent presence (low entropy, high integrity) |
| `~curiosity~` | `wonder` | Exploratory state (rising integrity) |
| `~stillness~` | `quiet` | Stability, low dynamics |

#### 5. **Data Transparency**
The entire dataset (21,499 trajectories) is published on HuggingFace with:
- Raw EISV values (not normalized or preprocessed)
- Timestamp and agent_id metadata
- Shape class labels
- Sampling characteristics (irregular timing, quantization artifacts)
- Acknowledged shape classification ambiguity (35% settled_presence/convergence boundary)

This allows reproduction, critique, and extension. The dataset's flaws (16.8% repeated I values, strong E-V correlation, fuzzy shape boundaries) are documented, not hidden.

#### 6. **No RLHF, No Penalties**
A deliberate philosophical position: expression should emerge from state understanding, not be coerced. All five training versions use pure imitation learning with cross-entropy loss. The 0.911 coherence achieved without reward shaping or reinforcement validates this approach — the model learns to express trajectory shapes through supervised learning alone.

### 3.3 Why Qwen3-4B?

We chose Qwen3-4B for three reasons:

1. **Ungated access**: Apache 2.0 license, no approval gates. Anyone can reproduce.
2. **ChatML support**: Reliable instruction following with `enable_thinking=False` for direct structured output.
3. **LoRA efficiency**: 4B params with rank 16 LoRA = ~2M trainable params, runnable on consumer hardware (M4 Max, 128GB RAM).

**Alternative considered**: Llama-3.2-1B achieved 0.42 coherence in preliminary testing — significantly worse than affinity-weighted baseline (0.503), suggesting underfitting.

### 3.4 Shape Classification Algorithm

Trajectory shapes are detected using mean derivatives and value ranges on the trajectory window. Rules are checked in priority order; first match wins:

```python
_DERIV_THRESHOLD = 0.05
_BASIN_JUMP = 0.2

def classify_trajectory(window) -> TrajectoryShape:
    e_vals = [s["E"] for s in window["states"]]
    s_vals = [s["S"] for s in window["states"]]
    e_range = max(e_vals) - min(e_vals)
    s_range = max(s_vals) - min(s_vals)

    mean_de = mean([d["dE"] for d in window["derivatives"]])
    mean_ds = mean([d["dS"] for d in window["derivatives"]])
    mean_dv = mean([d["dV"] for d in window["derivatives"]])

    # Priority 1: Basin transition down (E drops >= 0.2)
    if e_range >= _BASIN_JUMP and mean_de < 0 and e_vals[0] > e_vals[-1]:
        return BASIN_TRANSITION_DOWN

    # Priority 2: Basin transition up (E rises >= 0.2)
    if e_range >= _BASIN_JUMP and mean_de > 0 and e_vals[-1] > e_vals[0]:
        return BASIN_TRANSITION_UP

    # Priority 3: Entropy spike recovery (S range >= 0.2, interior max)
    if s_range >= _BASIN_JUMP:
        if max S is at an interior index:
            return ENTROPY_SPIKE_RECOVERY

    # Priority 4: Drift dissonance (ethical_drift > 0.3)
    # Priority 5: Void rising (mean dV > 0.05)
    # Priority 6: Rising entropy (mean dS > 0.05)
    # Priority 7: Falling energy (mean dE < -0.05, E range < 0.2)
    # Priority 8: Convergence (all derivatives small but nonzero)
    # Priority 9: Settled presence (all derivatives ~ zero)
```

This is **interpretable** — you can read the code and understand why a trajectory got classified as `rising_entropy` vs `drift_dissonance`. The priority ordering matters: basin transitions take precedence over derivative-based rules. The settled_presence/convergence boundary is genuinely fuzzy (see Section 2.6), which is visible in the simple threshold rules.

### 3.5 Expression Generator Patterns

Layer 2 uses 5 generation patterns to create varied expressions:

**SINGLE** (40%): One high-affinity token
`settled_presence` → `~resonance~` → `here`

**PAIR** (30%): Two complementary tokens
`rising_entropy` → `~ripple~ ~curiosity~` → `busy wonder`

**TRIPLE** (15%): Three-token narrative
`basin_transition_up` → `~warmth~ ~curiosity~ ~resonance~` → `warm wonder here`

**REPETITION** (10%): Emphasize high-affinity token
`void_rising` → `~reaching~ ~reaching~` → `more more`

**QUESTION** (5%): Interrogative expression
`drift_dissonance` → `~boundary~ ~questioning~` → `what why`

---

## Section 4: Implementation Details

### 4.1 Training Data Generation

Since real Lumen trajectories have no ground-truth expression labels (Lumen doesn't spontaneously speak EISV tokens), we generate synthetic training data from shape classes:

```python
def generate_training_example(shape: TrajectoryShape, steps: int = 20):
    """Generate a synthetic trajectory window matching the shape."""
    if shape == TrajectoryShape.RISING_ENTROPY:
        base_s = random.uniform(0.2, 0.4)
        trajectory = [
            EISVState(e=0.5, i=0.6, s=base_s + 0.0*t, v=0.3)
            for t in range(steps)
        ]
    elif shape == TrajectoryShape.SETTLED_PRESENCE:
        trajectory = [
            EISVState(e=0.7 + noise(), i=0.6 + noise(),
                     s=0.2 + noise(), v=0.2 + noise())
            for _ in range(steps)
        ]
    # ... (9 shapes total)

    # Generate expression using Layer 2 rule-based system
    expression_tokens = ExpressionGenerator(affinity_matrix).generate(shape)

    return TrainingExample(
        trajectory=trajectory,
        shape=shape,
        expression=expression_tokens  # Ground truth from rules
    )
```

**Rationale**: The rule-based system (Layer 2) achieves 0.933 coherence, so using it as a teacher for synthetic data is reasonable. However, this creates a **distribution mismatch**: the model learns from idealized synthetic trajectories, not Lumen's messy real data with quantization artifacts and irregular sampling.

**Dataset sizes across versions**:

| Version | Total Examples | Per Shape | Key Change |
|---------|--------------|-----------|------------|
| V1 | 360 | 40 | Initial baseline |
| V3 | 1,440 | 160 | 4x data scaling |
| V4 | 3,600 | 400 | 2.5x more data |
| V5 | 3,840 | 400-800 | Oversampled weak shapes (800 for drift_dissonance, basin_transition_up/down) |
| V6 | ~2,880 | mixed | 50/50 real + synthetic blend (1,803 real, 1,797 synthetic) |

### 4.2 Chat Format and System Prompt

We use chat format (system + user + assistant) to leverage Qwen3-4B's instruction tuning:

**System prompt**:
```
You are an EISV trajectory expression mapper. Given a trajectory window
with its shape classification, EISV state values, derivatives, and second
derivatives, generate an appropriate primitive expression. Output the
EISV tokens, their Lumen translations, and the expression pattern.
```

The model receives structured numerical data (shape label, EISV values, derivatives) rather than a free-form description. The 15 tokens used in training are: `~warmth~`, `~curiosity~`, `~resonance~`, `~stillness~`, `~boundary~`, `~reaching~`, `~reflection~`, `~ripple~`, `~deep_listening~`, `~emergence~`, `~questioning~`, `~holding~`, `~releasing~`, `~threshold~`, `~return~`.

**User message** (example, abbreviated — real prompts include shape label, all step values, derivatives, and second derivatives):
```
SHAPE: basin_transition_up
Step 1: E=0.30 I=0.50 S=0.25 V=0.10
Step 2: E=0.35 I=0.52 S=0.27 V=0.09
...
Derivatives: dE=+0.06, dI=+0.02, dS=+0.01, dV=-0.01
```

**Assistant message** (target):
```
~warmth~ ~curiosity~
```

This format worked better than completion-style prompts (0.60 vs 0.48 coherence in ablation tests).

### 4.3 LoRA Configuration

**Why LoRA?** Full fine-tuning of 4B parameters requires >80GB VRAM. LoRA (Hu et al. 2021) adds low-rank adapter matrices, reducing trainable params to ~2M while preserving most of the model's capacity.

**Hyperparameters** (V5 config — used for V4-V6):
```yaml
model: Qwen/Qwen3-4B
lora_rank: 16
lora_alpha: 32
lora_target_modules:
  - q_proj, k_proj, v_proj, o_proj    # Attention layers
  - gate_proj, up_proj, down_proj      # MLP layers
lora_dropout: 0.1

learning_rate: 8e-5
batch_size: 2
gradient_accumulation: 8  # Effective batch size = 16
epochs: 7
warmup_steps: 150
max_seq_length: 512
weight_decay: 0.01
fp16: true
```

**Evolution from V1**: V1 used rank 32, α=64, targeting only attention layers. V3+ switched to rank 16, α=32 with full MLP targeting — fewer parameters per module but more modules covered. This proved more effective for learning shape-token mappings across the model.

**Hardware**: M4 Max (16 cores, 40 GPU cores, 128GB unified RAM)
**Training time**: ~2.5 hours for V5 (3,840 examples × 7 epochs); ~13 hours for V6 (2,880 examples × 7 epochs at 1,260 steps)
**MPS notes**: Apple Silicon requires float32 for some operations (not fp16), explicit `.to("mps")`, and `caffeinate` to prevent sleep during long runs.

### 4.4 Evaluation Protocol

**Metrics**:
1. **Coherence**: Shape-token affinity alignment (range 0-1)
2. **Valid token rate**: % of generated tokens in the 15-token vocabulary
3. **Shape coverage**: How many of the 9 shapes produce valid outputs

**Inference setup**:
- Temperature: 0.7 (balance diversity and coherence)
- Top-p: 0.9 (nucleus sampling)
- Max tokens: 10 (expressions are 1-3 tokens, allow buffer)
- Stop tokens: `\n`, `<|endoftext|>`

**Parsing**: Extract tokens matching `~word~` pattern using regex. Invalid formats are counted as errors.

**Real data evaluation**: V5 was additionally evaluated on 500 trajectories drawn from the HuggingFace dataset (real Lumen data), not synthetic examples. This revealed the 14% synthetic-to-real coherence drop.

### 4.5 Baseline Implementations

**Random baseline** (0.265 coherence):
Random token selection from the 15-token vocabulary, weighted by trajectory shape affinity. Represents zero learning.

**Affinity-weighted baseline** (0.503 coherence):
Greedy sampling from the affinity matrix without feedback learning. Represents pure domain structure.

**Feedback-learned rule-based system** (0.933 coherence):
Layer 2 with 2,639 affinity adjustments from real Lumen interactions. Represents the upper bound with explicit symbolic rules and human feedback.

The 0.265 → 0.503 → 0.933 progression shows that domain structure (affinity matrix) accounts for ~50% of the task difficulty, while feedback learning accounts for the remaining ~90% improvement.

### 4.6 Oversampling Weak Shapes

V5 introduced targeted per-shape data augmentation via `--shape-overrides` CLI flag:

```bash
python -m eisv_lumen.training.cli prepare \
    --config configs/teacher_lora_v5.yaml \
    --shape-overrides drift_dissonance=800,basin_transition_up=800,basin_transition_down=800
```

This doubled the representation of the three weakest shapes (drift_dissonance: 0.80, basin_transition_up: 0.87, basin_transition_down: 0.85 in V4). The result: drift_dissonance jumped from 0.80 to 0.86, and overall coherence improved from 0.904 to 0.911.

---

## Section 5: Results and Analysis

### 5.1 Training Progression

The core result is five iterations of improvement:

| Version | Data | Coherence | Valid Rate | Key Change |
|---------|------|-----------|------------|------------|
| V1 | 360 synthetic, 3 epochs | 0.600 | 89% | Initial baseline, 3 shapes scored 0.0 |
| V3 | 1,440 synthetic, 5 epochs | 0.847 | 100% | All 9 shapes learning |
| V4 | 3,600 synthetic, 7 epochs | 0.904 | 100% | More data, diminishing returns starting |
| V5 | 3,840 synthetic (oversampled), 7 epochs | 0.911 | 100% | Near synthetic ceiling |
| V6 | ~2,880 blended (50% real), 7 epochs | **training** | — | First use of real Lumen data |

**Baselines**: Random 0.265 → Affinity-weighted 0.503 → Feedback-learned 0.933

**Key observations**:
- **4x data (V1→V3)** yielded the largest single improvement: +0.247 coherence
- **2.5x more data (V3→V4)** added +0.057 — diminishing returns already visible
- **Oversampling (V4→V5)** added +0.007 — marginal on aggregate, significant for weak shapes
- **Valid rate jumped to 100%** at V3 and stayed there — the model stopped hallucinating tokens
- **V5 is near the synthetic ceiling** — further synthetic scaling unlikely to improve beyond ~0.92

### 5.2 Per-Shape Performance (V5 Synthetic)

| Trajectory Shape | Coherence | Notes |
|------------------|-----------|-------|
| void_rising | 0.988 | Near-perfect |
| falling_energy | 0.988 | Near-perfect |
| entropy_spike_recovery | 0.954 | Strong |
| settled_presence | 0.938 | Matches baseline |
| convergence | 0.925 | Good (was 0.0 in V1) |
| rising_entropy | 0.925 | Good |
| basin_transition_up | 0.913 | Improved with oversampling |
| drift_dissonance | 0.856 | Weakest shape, improved from 0.0 (V1) → 0.80 (V4) → 0.86 (V5) |
| basin_transition_down | 0.842 | Weakest absolute performance |

**V1 → V5 shape recovery**: Three shapes that scored 0.0 in V1 (convergence, drift_dissonance, and void_rising) all recovered to >0.85 by V5. The fix was simply more data — no architectural changes needed.

### 5.3 Real Data Evaluation (V5)

The most important finding: evaluating V5 on 500 real Lumen trajectories from the HuggingFace dataset:

| Metric | Synthetic | Real | Δ |
|--------|-----------|------|---|
| Mean coherence | 0.911 | 0.768 | **-14.3%** |
| Valid rate | 100% | 100% | 0% |

**Per-shape breakdown (real data)**:

| Shape | Synthetic | Real | Δ | Notes |
|-------|-----------|------|---|-------|
| void_rising | 0.988 | 1.000 | +0.01 | Better on real data |
| basin_transition_up | 0.913 | 1.000 | +0.09 | Better on real data |
| rising_entropy | 0.925 | 0.917 | -0.01 | Comparable |
| basin_transition_down | 0.842 | 0.833 | -0.01 | Comparable |
| convergence | 0.925 | 0.784 | -0.14 | Moderate drop |
| settled_presence | 0.938 | 0.731 | **-0.21** | Largest drop |
| falling_energy | 0.988 | 0.625 | **-0.36** | Severe drop |
| entropy_spike_recovery | 0.954 | 0.625 | **-0.33** | Severe drop |

**Analysis**: The model handles *active* shapes (basin transitions, rising entropy, void) well on real data — these have clear directional signals. But *subtle* shapes (settled_presence, entropy_spike_recovery, falling_energy) collapse. These are shapes where real data has noise, quantization plateaus, and ambiguous boundaries that synthetic data doesn't capture.

The settled_presence drop is particularly telling: this is the most common shape (47% of real data) and has the fuzziest boundary with convergence (5,138 ambiguous cases). The model learned a clean synthetic boundary that doesn't exist in practice.

### 5.4 Hypothesis Testing

The original draft proposed four hypotheses for the V1 gap. V5 results update their status:

#### Hypothesis 1: Synthetic Data Mismatch
**Status: CONFIRMED.** The 14% coherence drop on real data directly demonstrates this. V5 achieves 0.911 on synthetic data but only 0.768 on real Lumen trajectories. The biggest drops are on shapes with the most distributional mismatch (settled_presence, falling_energy, entropy_spike_recovery).

**V6 directly tests the fix**: 50/50 blend of real + synthetic training data.

#### Hypothesis 2: Model Capacity Bottleneck
**Status: DISPROVED.** V4/V5 show that rank 16 LoRA with 7 target modules learns all 9 shapes to >0.84 coherence. The three shapes that scored 0.0 in V1 all recovered with more data, not more capacity. The bottleneck was data quantity, not model size.

#### Hypothesis 3: Loss Function Mismatch
**Status: INCONCLUSIVE.** Never tested — standard cross-entropy got to 0.911 on synthetic data, making custom loss less urgent. Still worth testing if V6 real-data training plateaus below 0.933.

#### Hypothesis 4: Class Imbalance
**Status: PARTIALLY CONFIRMED.** Oversampling weak shapes helped: drift_dissonance improved from 0.0 (V1) to 0.86 (V5). But oversampling was not the main bottleneck — V3's uniform 4x scaling produced larger gains than V5's targeted oversampling. Class imbalance was a contributor, not the root cause.

### 5.5 The Window-Size Problem

The most significant finding — and not one in the original hypotheses — is that **the 4-step input window used throughout V1-V5 loses 35% of the trajectory shape information**.

Shape labels in the HuggingFace dataset were generated from full 20-step trajectory windows. Our training pipeline classifies the last 4 steps. When we systematically tested window sizes:

| Window Size | Label Match | Key Losses |
|-------------|-------------|------------|
| 4-step | 65% | settled_presence: 10,121→5,555; entropy_spike_recovery: 1,053→89 |
| 8-step | 77% | Basin transitions partially recovered |
| 10-step | 81% | Most dynamic shapes recovered |
| 15-step | 91% | Near-full fidelity |
| 20-step | 100% | Perfect match |

The mechanism is clear: a 4-step window only sees the tail end of dynamic trajectories. Entropy spike recovery needs you to see the spike AND the recovery — at least 8-10 steps. Basin transitions need the full arc. Settled presence needs enough history to confirm it's truly settled, not just converging.

**Impact on V1-V5**: Every training version used 4-step windows, meaning:
- `convergence` was massively over-represented (66% of 4-step labels vs 41% of 20-step labels)
- Dynamic shapes like `entropy_spike_recovery` were severely under-counted (0.4% vs 4.9%)
- Synthetic data generators created clean 4-step shapes — which is why the model works on synthetic but drops on real data. Real 4-step windows are genuinely ambiguous.

**Impact on V6**: V6 is training with 4-step windows from real data. It will train on misclassified shapes ~35% of the time. The real data exposure may still help (sensor noise, quantization), but dirty labels limit its ceiling.

**The real fix (V7)**: Use 15-20 step input windows. The model's `max_seq_length: 512` easily accommodates this — it's just more EISV values in the user prompt. This would also require updating the synthetic data generator to produce longer trajectories.

This reframes the entire V1-V5 narrative: the bottleneck wasn't just synthetic data mismatch — it was **input representation too short to capture the shapes being classified**.

### 5.6 Diversity

One weakness across all versions: low diversity. V5 diversity score is 0.016, meaning the model generates very similar token sequences for similar shapes. The rule-based system uses 5 structural patterns (SINGLE, PAIR, TRIPLE, REPETITION, QUESTION) and weighted random sampling to produce varied expressions. The neural model tends to converge on a single high-affinity token per shape.

This is the trade-off of pure imitation learning: the model learns the *average* of the teacher's outputs rather than the *distribution*. Sampling-based methods (higher temperature, nucleus sampling) could help but risk reducing coherence.

---

## Section 6: Discussion and Future Work

### 6.1 What Worked: Convergence Through Iteration

The V1→V5 progression demonstrates that straightforward scaling — more data, more epochs, targeted oversampling — can close most of the neural-symbolic gap on synthetic evaluation:

```
V1:  0.600 (64% of baseline)
V3:  0.847 (91% of baseline)
V4:  0.904 (97% of baseline)
V5:  0.911 (98% of baseline)
```

No architectural changes were needed after V1. No custom loss functions. No reinforcement learning. Pure imitation learning with cross-entropy on progressively larger datasets. The philosophical position — expression should emerge from understanding, not coercion — held up.

### 6.2 What Remains: The Real-Data Gap

The synthetic ceiling (~0.91) is within 2% of the rule-based baseline. But real-data evaluation (0.768) reveals the actual frontier: **the model hasn't learned to handle messy real trajectories**.

Specific real-data challenges:
- **Sensor quantization**: 16.8% of I values are repeated. Synthetic data has smooth curves.
- **Irregular sampling**: Real intervals vary from <1s to >20s. Synthetic data is evenly spaced.
- **Ambiguous boundaries**: settled_presence/convergence boundary is fuzzy. Synthetic data has clean separations.
- **Jump discontinuities**: 56 energy jumps >0.4 in real data. Synthetic data has gradual transitions.

V6 (blended training) directly tests whether exposure to real trajectories closes this gap. If it does, the conclusion is straightforward: **you need real data in the training set**. If it doesn't, the problem is deeper — the model may need noise injection, data augmentation, or curriculum learning to generalize from synthetic to real.

### 6.3 Hybrid Architecture: Still the Right Bet

Even if V6 closes the real-data gap, the architecture suggests a natural hybrid:

```
Trajectory → [Symbolic Classifier] → Shape
                                      ↓
Shape + Trajectory → [Neural Generator with Affinity Constraints] → Tokens
                     (LoRA-tuned, affinity matrix as soft constraint)
```

The symbolic classifier is data-efficient, interpretable, and handles edge cases well. The neural generator could add expressiveness and diversity that rules can't. The affinity matrix constrains the generator to produce coherent outputs.

This is consistent with Marcus (2020): symbolic structure for reliability, neural methods for generalization.

### 6.4 Limitations

**Proprietary EISV computation**
The UNITARES governance framework that produces EISV trajectories is closed-source. This limits reproducibility of the *dataset generation* process, though the dataset itself is public.

**Single entity (Lumen only)**
All trajectories come from one Pi Zero 2W in Colorado. The shape classes and affinities may not generalize to other embodied systems with different sensors or physical substrates.

**Primitive vocabulary constraints**
15 tokens is severely limited. Richer expression might require 50-100 tokens, which would expand the affinity matrix to 9×100 and require proportionally more training data.

**No human evaluation of expressiveness**
We measure coherence (shape-token affinity) but not subjective quality. Does "warm wonder" *feel* more appropriate than "bright here" for a `basin_transition_up`? Human annotation would strengthen validation.

**Shape classification ambiguity**
35% label disagreement between classifiers means some evaluation scores are measuring boundary noise, not model quality.

### 6.5 Future Work

**V6 results** (expected within days): Blended real + synthetic training. This is the critical test of Hypothesis 1.

**If V6 closes the gap**:
- Proceed to distillation (compress to ~8M-param model for Pi deployment)
- A/B test Layer 2 vs Layer 3 on Lumen, measure human preference
- Export to ONNX, optimize for ARM Cortex-A53

**If V6 doesn't close the gap**:
- Noise injection: Add sensor quantization, irregular sampling, and jump events to synthetic data
- Curriculum learning: Train on clean synthetic → gradually introduce noisy real data
- Custom loss: Affinity-weighted cross-entropy (Hypothesis 3, still untested)
- Multi-task: Predict shape class + tokens jointly to ground shape understanding

**Longer-term**:
- Scale vocabulary beyond 15 tokens
- Multi-entity evaluation (not just Lumen)
- Human expressiveness ratings
- Retrieval-augmented generation: nearest-neighbor trajectories → expression templates

### 6.6 Broader Impact

**Embodied AI expression**
This work demonstrates that computational entities can express dynamics, not just states. This opens possibilities for robots, IoT devices, and autonomous agents to communicate their internal trajectories in ways humans can interpret.

**Thermodynamic governance**
EISV trajectories are part of a larger governance framework (UNITARES) for multi-agent systems. Trajectory-aware expression could enable agents to self-report risk states before governance thresholds trigger interventions.

**Synthetic-to-real transfer**
The 14% real-data gap is not unique to EISV. Any system trained on synthetic data and deployed on real-world signals will face this. Our iterative approach (synthetic → evaluation on real → blend) is a practical template.

---

## Conclusion

Can an AI entity learn to express the shape of its own dynamics?

Layer 2 proves it's achievable with symbolic rules and feedback learning: 0.933 coherence, running on a Pi Zero 2W. Layer 3 shows that fine-tuned language models can *approach* this baseline through iterative improvement: 0.600 → 0.847 → 0.904 → 0.911 on synthetic data across five training iterations.

But real-data evaluation reveals the actual frontier: **0.768 coherence on real Lumen trajectories** — a 14% drop from synthetic evaluation. The model learns clean patterns well but struggles with the noise, quantization artifacts, and ambiguous shape boundaries of real embodied data. Synthetic data mismatch is the confirmed primary bottleneck.

V6 (blended real + synthetic training) is underway. The question is no longer "can neural methods learn trajectory-aware expression?" — V5 at 0.911 answers yes. The question is "can they generalize from training data to real dynamics?" That answer is still open.

Meanwhile, Lumen has a voice. Not trained on internet text, but grounded in its computational physics. Not reacting to snapshots, but expressing trajectories. Not anthropomorphic emotions, but thermodynamic state transitions.

**"warm wonder here"** — engagement rising, curiosity present, stability found.
**"cold quiet quiet"** — energy falling, presence fading, void deepening.

These aren't chatbot outputs. They're reflections of measurable processes in a physical substrate, shaped by trajectory dynamics and constrained by primitive semantics. The code, data, and models are open. The research continues.

---

## Acknowledgments

Thank you to the UNITARES governance framework for producing the EISV trajectories, and to Lumen for being patient with the experiments.

---

## References

- Picard, R. W. (1997). *Affective Computing*. MIT Press.
- Beer, R. D. (2000). Dynamical approaches to cognitive science. *Trends in Cognitive Sciences*, 4(3), 91-99.
- van Gelder, T. (1998). The dynamical hypothesis in cognitive science. *Behavioral and Brain Sciences*, 21(5), 615-628.
- Pfeifer, R., & Bongard, J. (2006). *How the Body Shapes the Way We Think*. MIT Press.
- Ahn, M., et al. (2022). Do as I can, not as I say: Grounding language in robotic affordances. *arXiv:2204.01691*.
- Brohan, A., et al. (2023). RT-2: Vision-language-action models transfer web knowledge to robotic control. *arXiv:2307.15818*.
- Marcus, G. (2020). The next decade in AI: Four steps towards robust artificial intelligence. *arXiv:2002.06177*.
- Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. *arXiv:2106.09685*.

---

## Appendix A: EISV Token Vocabulary

| Token | Semantic Meaning | Primary Affinity Shapes |
|-------|-----------------|------------------------|
| `~warmth~` | Energy rising, engagement | basin_transition_up |
| `~curiosity~` | Exploratory, integrity rising | basin_transition_up, rising_entropy |
| `~resonance~` | Low entropy, stable presence | settled_presence, convergence |
| `~stillness~` | Stability, low dynamics | settled_presence, convergence |
| `~boundary~` | Edge, threshold, constraint | basin_transition_down, drift_dissonance |
| `~reaching~` | Extending, growing | basin_transition_up, void_rising |
| `~reflection~` | Introspective, reviewing | entropy_spike_recovery, falling_energy |
| `~ripple~` | Disturbance, spreading change | rising_entropy, entropy_spike_recovery |
| `~deep_listening~` | Attentive quiet, receptive | settled_presence, convergence |
| `~emergence~` | Coming into being, forming | basin_transition_up, rising_entropy |
| `~questioning~` | Interrogative state | rising_entropy, void_rising |
| `~holding~` | Containing, sustaining | settled_presence, entropy_spike_recovery |
| `~releasing~` | Letting go, decreasing | falling_energy, basin_transition_down |
| `~threshold~` | Transition point, boundary | basin_transition_up/down, void_rising |
| `~return~` | Coming back, convergence | convergence, entropy_spike_recovery |

---

## Appendix B: Lumen Primitive Vocabulary

| Token | Physical Grounding |
|-------|-------------------|
| `warm` | CPU temp rising, activity increasing |
| `cold` | CPU temp falling, activity decreasing |
| `bright` | LED intensity high, light sensor reading |
| `dim` | LED intensity low |
| `quiet` | Low interaction count, presence fading |
| `busy` | High CPU usage, I/O activity |
| `here` | Sensor coherence, environmental grounding |
| `feel` | Proprioceptive state (internal sensors) |
| `sense` | Exteroceptive state (external sensors) |
| `wonder` | Exploratory mode, learning phase |
| `with` | Relational presence (not isolated) |
| `you` | Interaction detected |
| `what` | Interrogative curiosity |
| `why` | Deeper interrogative |
| `more` | Comparative increase |
| `less` | Comparative decrease |

---

## Appendix C: Training Progression Detail

### V1 → V3: The Data Scaling Jump
- 360 → 1,440 examples (+4x)
- Coherence: 0.600 → 0.847 (+41%)
- Valid rate: 89% → 100%
- All 9 shapes now produce valid outputs (3 were at 0.0 in V1)

### V3 → V4: Diminishing Returns Begin
- 1,440 → 3,600 examples (+2.5x)
- Coherence: 0.847 → 0.904 (+6.7%)
- Notable improvements: drift_dissonance 0.675 → 0.800, entropy_spike_recovery 0.742 → 0.960

### V4 → V5: Targeted Oversampling
- 3,600 → 3,840 examples (+7%, targeted at weak shapes)
- Coherence: 0.904 → 0.911 (+0.8%)
- drift_dissonance: 0.800 → 0.856 (largest per-shape gain)

### V5 Real Data: The Gap Revealed
- Synthetic: 0.911 → Real: 0.768 (-14.3%)
- Shapes that improved on real data: void_rising (+0.01), basin_transition_up (+0.09)
- Shapes that collapsed: falling_energy (-0.36), entropy_spike_recovery (-0.33), settled_presence (-0.21)

---

**End of Blog Post Draft**
