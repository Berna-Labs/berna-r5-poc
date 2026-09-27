# Ablation Study Design — Berna R5

## Purpose

Isolate the causal contribution of each architectural component.
This is the single most important experiment for validating that
Berna R5 is not just a complex system, but a system where each
component matters causally.

## Hypothesis

For each component C in {Growth, Cells, Plexus, DNA, Transactional},
removing C degrades at least one of:
- downstream loss
- retention (F_j)
- plasticity (learning new tasks)
- routing accuracy

If a component can be removed without measurable degradation,
that component is not justified.

## Ablation Matrix

Six variants, trained on the same data, same seed, same hardware.

| Variant | Growth | Cells | Plexus | DNA | Transactional |
|---------|:------:|:-----:|:------:|:---:|:-------------:|
| V0: Vanilla        | no  | no  | no  | no  | no  |
| V1: +Cells         | yes | yes | no  | no  | no  |
| V2: +Plexus        | yes | yes | yes | no  | no  |
| V3: +DNA           | yes | yes | yes | yes | no  |
| V4: +Transactional | yes | yes | yes | yes | yes |
| V5: Full R5        | yes | yes | yes | yes | yes |

Note: V5 is identical to V4 if Transactional is a runtime
property rather than a training component. In that case, V5 is
dropped and we have 5 variants.

## Configuration

- Model: 50M parameters (PoC scale)
- Data: 800M tokens (WikiText-103 + The Stack + OpenWebMath)
- Hardware: 1x RTX 5090
- Seed: 42 (fixed across all variants)
- Optimizer: AdamW, lr=3e-4
- Batch: 4 (safe), grad_accum=32
- Sequence length: 512

## Metrics

For each variant, measure:

### Training Metrics
- Final training loss (perplexity)
- Convergence speed (steps to reach loss=5.0)
- Tokens per second (throughput)

### Retention Metrics
- F_j for each domain (per Theory 3)
- Max forgetting across domains
- Forgetting rate (per billion tokens)

### Plasticity Metrics
- Learning speed on new domain (loss decrease per 1000 steps)
- Final loss on new domain

### Routing Metrics
- Routing accuracy (correct specialist for domain)
- Routing latency (ms per request)

### Growth Metrics
- Number of cells at end (N*)
- Splits occurred
- Diversity of daughters (avg div(d1, d2))

### Causal Metrics
- Cell activation to output correlation
- Cell ablation impact (remove one cell, measure loss delta)

## Experimental Protocol

### Phase 1: Single-Task Training
Train each variant on WikiText-103 (100M tokens).
Measure: final loss, throughput, cell count.

### Phase 2: Sequential Multi-Task
Train on 4 tasks sequentially:
1. WikiText-103
2. The Stack (Python)
3. OpenWebMath
4. WikiText-103 again (re-test retention)

After each task, evaluate on ALL previous tasks.
Measure: F_j, retention matrix.

### Phase 3: Zero-Shot Transfer
After Phase 2, evaluate each variant on held-out test sets
for all 4 domains without additional training.
Measure: generalization.

### Phase 4: Causal Cell Test
For V5 (full R5):
- Freeze all cells except one
- Zero out that cell's weights
- Measure loss increase on the domain it activates on
- Expected: significant loss increase (>10%)

If loss doesn't change, the cell is metadata, not computation.

## Expected Outcomes

### Prediction 1: Growth helps retention
- V0 (no growth): high forgetting (F_j ~ 0.1-0.3)
- V4 (full): low forgetting (F_j ~ 0)
- Delta: 10x improvement

### Prediction 2: DNA helps convergence
- V2 (no DNA): slower convergence
- V3 (with DNA): faster convergence
- Delta: 10-20% fewer steps

### Prediction 3: Plexus helps routing
- V1 (no plexus): random routing
- V2 (with plexus): accurate routing
- Delta: 30%+ routing accuracy gain

### Prediction 4: Transactional preserves correctness
- V3 (no transactional): occasional K_6D decreases
- V4 (with transactional): monotonic K_6D
- Delta: zero vs. nonzero regression

### Prediction 5: Cells are causal
- Freezing one cell: loss increase on its domain
- Threshold: >5% loss increase on that domain
- Threshold: <1% loss change on other domains

## Falsification Criteria

The ablation study falsifies the architecture if:
- Any variant matches V4 within 5% on all metrics
- Removing a component has no measurable effect
- Cell ablation produces no domain-specific loss increase

If falsified, the corresponding component should be removed
from the paper.

## Compute Budget

| Phase   | GPU-hours | Calendar days |
|---------|-----------|---------------|
| Phase 1 | 6         | 1             |
| Phase 2 | 24        | 2             |
| Phase 3 | 2         | 0.5           |
| Phase 4 | 4         | 0.5           |
| Total   | 36        | 4             |

Note: This requires the GPU to be free. R4 currently
occupies the RTX 5090 for ~24 more days.

Options:
1. Wait for R4 to finish
2. Rent a separate GPU (Vast.ai, RunPod)
3. Run at smaller scale (10M params, 100M tokens) for quick validation

## Reproducibility

All variants will be run with:
- Fixed seed 42
- Same data hash
- Same hardware
- Same code version (git commit hash)

Commands:
    python experiments/ablation/run.py --variant V0 --phase 1
    python experiments/ablation/run.py --variant V4 --phase 2

## Deliverables

After completion:
1. results/ablation_matrix.csv - all metrics for all variants
2. results/retention_heatmap.png - visual retention matrix
3. results/cell_ablation.json - causal cell test results
4. Paper section 5.9 rewritten with real numbers
5. Decision: which components are kept in the paper

## Status

- [x] Design (this document)
- [ ] Implementation of run.py
- [ ] Data preparation
- [ ] Execution
- [ ] Analysis
- [ ] Paper integration

## Version

- v1.0 - 2026-09-27 - Initial design
