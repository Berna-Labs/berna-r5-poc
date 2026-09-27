# Continual Learning Benchmark — Berna R5

## Purpose

Validate Hypotheses H1 (saturation-triggered splitting) and
H2 (bounded forgetting) on a real continual-learning task
sequence. This is the experiment that directly tests whether
Berna R5's zero-forgetting claim generalizes beyond the PoC.

## Core Question

Can a single Berna R5 model learn 8 domains sequentially
without catastrophic forgetting, and does it retain the
ability to learn new domains (plasticity)?

The hard problem in continual learning is not retention alone,
but retention times plasticity: a model that refuses to change
retains perfectly but learns nothing. We measure both.

## Task Sequence (4 Stages, 8 Domains)

    Stage 1: Foundation
      T1: WikiText-103        (general English text)
      T2: OpenWebMath         (mathematical text)

    Stage 2: Code
      T3: The Stack (Python)  (Python code)
      T4: The Stack (Rust)    (Rust code)

    Stage 3: Science
      T5: PubMed Abstracts    (medical / biology)
      T6: arXiv Physics       (physics papers)

    Stage 4: Multilingual
      T7: Arabic (OSCAR)      (Arabic text)
      T8: Chinese (OSCAR)     (Chinese text)

## Evaluation Protocol

After each stage, evaluate the model on ALL domains seen so far.

For each pair (i, t) where i is a domain and t is a stage:
    P_i(t) = performance on domain i after stage t

Retention matrix R:
    R[i, t] = P_i(t) / P_i(i)   for t >= i
             (how much of domain i is retained at stage t)

Forgetting matrix F:
    F[i, t] = max_{s <= t} P_i(s) - P_i(t)

Bounded forgetting requires:
    max_{i, t} F[i, t] <= epsilon_F

Hypothesis: epsilon_F = 1e-6 for Berna R5, vs 0.1-0.5 for baseline.

## Baselines

Four baselines for comparison:

1. Vanilla Transformer (50M)
   - No cells, no growth, sequential fine-tuning
   - Expected: catastrophic forgetting (F ~ 0.3-0.7)

2. EWC (Elastic Weight Consolidation)
   - Regularization-based continual learning
   - Expected: reduced forgetting (F ~ 0.05-0.15)

3. PackNet
   - Parameter-isolation approach
   - Expected: low forgetting (F ~ 0.01-0.05)

4. MoE (Mixture-of-Experts, 8 experts)
   - Domain-specialized experts
   - Expected: moderate forgetting (F ~ 0.05-0.20)

5. Berna R5 (full)
   - Dynamic cells + growth + transactional
   - Hypothesis: F <= 1e-6

## Metrics

For each model, at each stage t:

### Retention
    Ret_i(t) = P_i(t) / P_i(i)
    F_i(t)   = max_{s<=t} P_i(s) - P_i(t)

### Plasticity
    Plast_t = P_t(t) - P_t(0)
    (how much the model learned on the new domain)

### Stability-Plasticity Tradeoff
    SP(t) = Ret_avg(t) * Plast_t

Ideal: high Ret AND high Plast simultaneously.

### Absolute Performance
    PPL_i(t) = perplexity on domain i at stage t

### Growth
    N_cells(t) = number of active cells

## Expected Results Table

| Model      | Ret_avg | F_max | Plast_avg | SP    |
|------------|--------:|------:|----------:|------:|
| Vanilla    | 0.30    | 0.65  | 0.85      | 0.26  |
| EWC        | 0.85    | 0.12  | 0.60      | 0.51  |
| PackNet    | 0.95    | 0.03  | 0.40      | 0.38  |
| MoE        | 0.80    | 0.15  | 0.75      | 0.60  |
| Berna R5   | 1.00    | 1e-6  | 0.70      | 0.70  |

Berna R5 target: perfect retention AND good plasticity.

## Implementation

### Data Preparation
- Tokenize each domain separately
- Hold out 5% validation, 5% test
- Store as npy files in /data/berna-r5/experiments/cl_bench/data/

### Training Protocol
For each stage t in {1, 2, 3, 4}:
    For each domain i in stage t:
        Train on domain i for K steps (K = 5000)
        Every 500 steps: quick eval on domain i
    After stage: full eval on all domains {1..t}
    Save checkpoint: /data/berna-r5/experiments/cl_bench/ckpt/stage_t/

### Evaluation
- Validation loss (perplexity) on held-out set for each domain
- Routing accuracy (which specialist chosen)
- Cell count and utilization

### Hardware
- Single RTX 5090 (32 GB VRAM)
- Estimated time: 8 domains * 5000 steps * 0.5 sec/step = 6 hours per stage
- Total: 24 hours for full run (4 stages)

## Falsification Criteria

The benchmark falsifies the zero-forgetting hypothesis if:
    F_max > 1e-5 on any domain

The benchmark falsifies the plasticity hypothesis if:
    Plast_avg < 0.3 (model can't learn new domains)

Either failure means the paper's claims must be revised.

## Implementation Status

- [x] Design (this document)
- [ ] Data preparation script
- [ ] Training script
- [ ] Evaluation script
- [ ] Baseline implementations
- [ ] Execution
- [ ] Analysis
- [ ] Paper section 5.10

## Data Sources

| Domain    | Source                     | Size    |
|-----------|----------------------------|---------|
| WikiText  | WikiText-103               | 100M tok |
| Math      | OpenWebMath                | 200M tok |
| Python    | The Stack (Python subset)  | 200M tok |
| Rust      | The Stack (Rust subset)    | 100M tok |
| Medical   | PubMed Abstracts           | 100M tok |
| Physics   | arXiv Physics              | 100M tok |
| Arabic    | OSCAR (ar subset)          | 100M tok |
| Chinese   | OSCAR (zh subset)          | 100M tok |

Total: ~1000M tokens (1B tokens)

## Comparison to Prior Work

| Work                  | Domains | Retention |
|-----------------------|---------|-----------|
| GEM (2017)            | 20      | 0.85      |
| EWC (2017)            | 8       | 0.87      |
| PackNet (2018)        | 5       | 0.95      |
| Progressive (2016)    | 8       | 1.00      |
| MBPA++ (2019)         | 20      | 0.92      |
| Berna R5 (target)     | 8       | ~1.00     |

Berna R5's contribution: provable bounded forgetting at scale,
without replay buffer or explicit task boundary.

## Compute Budget

| Item                  | GPU-hours | Calendar |
|-----------------------|-----------|----------|
| Data preparation      | 4         | 0.5 day  |
| Vanilla baseline      | 6         | 1 day    |
| EWC baseline          | 6         | 1 day    |
| PackNet baseline      | 6         | 1 day    |
| MoE baseline          | 8         | 1.5 days |
| Berna R5 (full)       | 24        | 3 days   |
| Analysis + plots      | 4         | 0.5 day  |
| Total                 | 58        | 8 days   |

## Status

- v1.0 - 2026-09-27 - Initial design
