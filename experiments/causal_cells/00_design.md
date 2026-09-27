# Causal Cell Activation — Berna R5

## Purpose

Prove that knowledge cells are not metadata. They must causally
affect model output. This is the single most important test for
distinguishing Berna R5 from an elaborate metadata system.

A cell is "causal" if:
    - Removing it changes model output on its domain
    - Removing it does NOT change output on other domains
    - Its activation correlates with domain-specific input

If any of these fails, the cell is not a computational unit.

## Core Question

When a cell activates, does it change the computation, or
does it only change a label? If the latter, Berna R5 is MoE
with documentation, not a new architecture.

## Three Tests

### Test A: Ablation Causality
For a cell c in domain d:
1. Measure loss on domain d with c active: L_with(c, d)
2. Zero out c's weights: L_without(c, d)
3. Measure delta: delta(c, d) = L_without - L_with

Causality requires:
    delta(c, d) > threshold (e.g., 5%)
    delta(c, d') < 1% for d' != d

### Test B: Activation Correlation
For a cell c and input x:
1. Compute activation a(c, x)
2. Compute domain label of x: dom(x)
3. Measure: corr(a(c, x), 1[dom(x) = c.domain])

Causality requires:
    corr > 0.7 for c's own domain
    corr < 0.2 for other domains

### Test C: Gradient Causality
For input x in domain d, cell c in domain d:
1. Compute gradient of loss w.r.t. c's weights: grad_c
2. Compute gradient norm: ||grad_c||
3. Compare to gradient of unrelated cell: ||grad_other||

Causality requires:
    ||grad_c|| > 3 * ||grad_other||

If cell c receives no gradient on its own domain, it is not
participating in the computation.

## Experimental Setup

### Model
- 50M-parameter Berna R5 PoC
- Trained on 8-domain corpus (see CL benchmark)
- Cells split during training via saturation

### Test Data
- Held-out test sets for each domain
- 1000 examples per domain

### Cells to Test
- All cells active at end of training
- Expected: 20-50 cells
- Each cell is tested on all 8 domains

## Protocol

For each cell c:
    For each domain d:
        For each test example x in d:
            1. a = activation(c, x)
            2. L_with = loss(model_with_c, x)
            3. L_without = loss(model_without_c, x)
        delta(c, d) = mean(L_without - L_with)
        corr(c, d) = corr(activations, domain_labels)

## Expected Output Matrix

    Cell-to-Domain Delta Matrix
    (rows = cells, columns = domains)

         d1     d2     d3     d4     d5     d6     d7     d8
    c1   0.08   0.00   0.00   0.01   0.00   0.00   0.00   0.00
    c2   0.00   0.06   0.01   0.00   0.00   0.00   0.00   0.00
    c3   0.00   0.00   0.12   0.00   0.00   0.00   0.00   0.00
    c4   0.00   0.00   0.00   0.09   0.00   0.00   0.00   0.00
    ...

Berna R5 target: strong diagonal, weak off-diagonal.

## Metrics

### Sparsity
    S = (sum of diagonal) / (sum of all entries)
Target: S > 0.8

### Causality Strength
    C = mean over cells of max_d delta(c, d)
Target: C > 0.05

### Domain Selectivity
    DS = mean over cells of delta(c, c.domain) / max_{d != c.domain} delta(c, d)
Target: DS > 5.0

### Activation Correlation
    AC = mean over cells of corr(c, c.domain)
Target: AC > 0.7

## Falsification Criteria

The causal-cell hypothesis is falsified if:
- Diagonal delta < 0.01 (cells do nothing)
- Off-diagonal delta ~ diagonal delta (cells are not specific)
- Activation correlation < 0.3 (cells don't fire on their domain)

If falsified, Berna R5 is a metadata system, not a knowledge-
cell architecture. The paper must be rewritten to reflect this.

## Why This Test Matters

MoE has experts that are computationally real: each expert is
a distinct MLP block. Berna R5 must demonstrate the same, plus
something extra: the cells can grow, split, and be frozen.

Without this test, a reviewer can claim:
    "Cells are just labels on a single shared MLP."

This test rules out that claim.

## Implementation

### Required Code
1. activation_extractor.py - extracts cell activations for input
2. ablation_runner.py - zeros out one cell, measures loss delta
3. correlation_calculator.py - computes activation correlation
4. plot_delta_matrix.py - visualizes the delta matrix

### Integration with Existing Code
- src/cell.py already has cell.weights structure
- src/plexus.py already tracks cell activations
- Need: extract and freeze/unfreeze operations

### Hardware
- Same as CL benchmark: RTX 5090
- Time: ~4 hours for all cells and all domains

## Deliverables

1. results/delta_matrix.csv - cell-by-domain delta
2. results/correlation_matrix.csv - cell-by-domain correlation
3. results/plots/delta_heatmap.png - visual
4. results/plots/correlation_heatmap.png - visual
5. Paper section 5.11: Causal Validation of Cells

## Compute Budget

| Item                    | GPU-hours | Calendar |
|-------------------------|-----------|----------|
| Activation extractor    | 1         | 0.5 day  |
| Ablation runner         | 2         | 0.5 day  |
| Correlation calculator  | 1         | 0.5 day  |
| Plotting + analysis     | 1         | 0.5 day  |
| Total                   | 5         | 2 days   |

## Integration with Existing Theories

This experiment directly tests:
- Theory 1 (Saturation): do cells that saturate on a domain
  become domain-specific?
- Theory 2 (Incorporation): when a specialist answers, does
  its cell activate?
- Theory 5 (Plexus): do co-activated cells form clusters?

A positive result strengthens all five theories. A negative
result would require rethinking the cell concept.

## Status

- [x] Design (this document)
- [ ] activation_extractor.py
- [ ] ablation_runner.py
- [ ] correlation_calculator.py
- [ ] plot_delta_matrix.py
- [ ] Execution
- [ ] Analysis
- [ ] Paper section 5.11

## Version

- v1.0 - 2026-09-27 - Initial design
