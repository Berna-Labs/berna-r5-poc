# Berna R5: Experiment Plan

## Objective
Validate all 5 theorems on a 50M-parameter Proof of Concept (PoC)
before scaling to 1.5B. No claims without evidence.

## Timeline
- Phase 1: Setup (1 week)
- Phase 2: PoC training (2 weeks)
- Phase 3: Theorem validation (1 week)
- Phase 4: Baselines comparison (1 week)
- Phase 5: Analysis (1 week)
- **Total: 6 weeks**

## Hardware
- 1x RTX 5090 (32 GB VRAM)
- Fallback: CPU for analysis only

## Datasets
- WikiText-103 (~100M tokens) — primary
- The Stack (Python only, ~500M tokens) — code
- OpenWebMath (~200M tokens) — math
- Total: ~800M tokens for PoC

---

## Experiment 1: Saturation Dynamics (Theory 1)

### Hypothesis
The saturation score S(c, t) converges to S* ~ 0.85 within
5000 steps, and remains stable thereafter.

### Setup
- 50M model, 10 initial cells
- Log K_6D(c, t), omega(c, t), S(c, t) every 100 steps
- Run for 50,000 steps (~50M tokens)

### Measurements
- S(t) curve for each cell
- Fit: S(t) = S* (1 - exp(-t/tau))
- Estimate S* and tau
- Isotropy: max |omega_i * K_i - omega_j * K_j| over last 1000 steps

### Success Criteria
- |S* - 0.85| < 0.05
- tau < 5000 steps
- Isotropy < 0.05
- No NaN/Inf (Zero-Error check)

### Failure Mode
If S* drifts > 0.10 from 0.85, retune or reject the theorem.

---

## Experiment 2: Splitting Behavior (Theory 1 + 4)

### Hypothesis
Cells split when S >= S* for 3 consecutive steps, and each
split decreases global loss.

### Setup
- Same 50M model
- Enable growth with S* = 0.85
- Run for 50,000 steps

### Measurements
- Number of splits over time: N(t)
- Loss before/after each split
- Daughter divergence: div(d1, d2)
- Capacity check: sum footprint <= Cap

### Success Criteria
- Loss decreases at every split (avg)
- div >= 0.3 for all splits
- N(t) converges (no runaway growth)
- No capacity violation

### Failure Mode
If loss increases at any split, halt and inspect.

---

## Experiment 3: Incorporation Events (Theory 2)

### Hypothesis
Every incorporation preserves K_6D(B) with delta >= -1e-6.

### Setup
- Base (50M) + 3 specialists (50M each: code, math, med)
- Generate 1000 requests spanning all 3 domains
- Run incorporation protocol

### Measurements
- delta_K for each dimension, per request
- Routing accuracy: correct specialist selected?
- Gap queue size over time
- Specialist K_6D unchanged?

### Success Criteria
- 100% min delta_K >= -1e-6
- Routing accuracy >= 85%
- Gap queue bounded
- Specialists unchanged

### Failure Mode
If any delta_K < -1e-5, halt and inspect verification gate.

---

## Experiment 4: Zero-Forgetting (Theory 3)

### Hypothesis
After training on tasks T1, T2, T3 sequentially, F_j < 1e-6
for all j.

### Setup
- Task T1: WikiText (general text)
- Task T2: The Stack (code)
- Task T3: OpenWebMath (math)
- Train sequentially on each for ~16,000 steps
- Evaluate on all tasks after each phase

### Measurements
- P_j(t) for all j, all t
- F_j(t) = max P_j - current P_j
- Compare with naive fine-tuning baseline

### Success Criteria
- F_j < 1e-6 for all j
- No task regresses
- Performance on new task increases

### Failure Mode
If F_j > 1e-5, the cell isolation or transaction has failed.

---

## Experiment 5: Baseline Comparison

### Baselines
1. **Vanilla Transformer 50M** — no cells, no growth
2. **MoE 50M** — 8 experts, fixed routing
3. **EWC 50M** — Elastic Weight Consolidation
4. **PackNet 50M** — pruning-based

### Comparison Metrics
| Metric | How measured |
|---|---|
| Loss | validation loss |
| Perplexity | exp(loss) |
| Forgetting | F_j (Theory 3) |
| Throughput | tokens/sec |
| Memory | peak VRAM |
| Latency | time per token |

### Success Criteria
- Berna R5 loss competitive with best baseline
- Berna R5 forgetting strictly better (< 1e-6)
- Acceptable throughput (within 30% of vanilla)

### Failure Mode
If Berna R5 loses by > 50% in loss, revisit architecture.

---

## Experiment 6: Plexus Dynamics (Theory 5)

### Hypothesis
Plexus weights converge to W* = A* / lambda_e within 1000
steps, with rate rho = |1 - eta_e * lambda_e|.

### Setup
- 20 cells, random initial edges
- Simulate co-activations over 10,000 steps
- Log all edge weights

### Measurements
- W_e(t) curves
- Fit to W* + (W_0 - W*) * rho^t
- Small-world coefficient: sigma = C/C_random
- Scale-free exponent: gamma

### Success Criteria
- Convergence within 1000 steps
- Measured rho matches predicted
- sigma > 3 (small-world)
- gamma in [2, 3] (scale-free)

### Failure Mode
If W diverges, check eta_e < 2 / lambda_e.

---

## Experiment 7: Ablations

### Ablations to Run
1. **No growth** — cells never split (N constant)
2. **No incorporation** — base learns alone
3. **Fixed omega** — no adaptive weights
4. **100 vs 50 chromosomes**
5. **6 dims vs 3 dims** (drop T, H, D)

### Purpose
Show that each component contributes to performance.
If an ablated model performs equally, the component is not needed.

### Success Criteria
- Full model beats each ablation by >= 5% in loss
- Removing critical components (growth, incorporation) shows large degradation

---

## Metrics Summary

| Experiment | Primary Metric | Threshold |
|---|---|---|
| Saturation | S* convergence | \|S*-0.85\|<0.05 |
| Splitting | loss decrease | > 0 at each split |
| Incorporation | min delta_K | >= -1e-6 |
| Forgetting | F_j | < 1e-6 |
| Baseline | loss | within 1.3x of best |
| Plexus | rho match | \|rho_meas - rho_pred\| < 0.01 |
| Ablations | delta loss | > 5% |

---

## Reproducibility

- Seed: fixed at 42
- Data hash: SHA256 of tokenized files
- Model hash: SHA256 of config
- Every experiment logged to registry (SQL)
- Command line: `python experiments/run.py --exp N --seed 42`

---

## Risk Management

| Risk | Likelihood | Mitigation |
|---|---|---|
| OOM on 5090 | Medium | Reduce batch, use gradient checkpoint |
| Slow convergence | Medium | Tune LR, warmup |
| Theorem not validated | Low | Report honestly, revise |
| Data issues | Low | Deduplicate, filter |
| Power loss | High (Iraq) | Atomic checkpoint |

---

## Deliverables

After Phase 5:
1. `results/exp1_saturation.json`
2. `results/exp2_growth.json`
3. `results/exp3_incorporation.json`
4. `results/exp4_forgetting.json`
5. `results/exp5_baselines.json`
6. `results/exp6_plexus.json`
7. `results/exp7_ablations.json`
8. Figures: 7 plots (one per experiment)
9. Summary: `results/summary.md`

---

## What We Do NOT Claim

- SOTA on any benchmark
- Superiority over all existing methods
- Scalability beyond 1.5B (untested)
- Theoretical completeness (see zero_error_creed.md)

We claim: these 7 experiments validate or falsify the 5 theorems
at 50M scale.

---

## Summary

7 experiments, 6 weeks, 1 GPU, 800M tokens.
Every experiment has explicit success criteria and failure modes.
No hand-waving. Results either validate or falsify.
