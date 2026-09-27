# Theory 4: Balanced Growth in the Federation

## 1. Motivation

A growing federation risks two failure modes:
    (a) Explosion: cells split unboundedly, VRAM overflows
    (b) Stagnation: cells never split, base stays narrow

Berna R5 requires balanced growth: splits happen when needed,
stop when saturated, and never exceed hardware capacity.

This theory formalizes growth control with three constraints:
    C1: necessity      - split only when beneficial
    C2: capacity       - total cells fit in budget
    C3: diversity      - daughters differ meaningfully

## 2. Definitions

Definition 2.1 (Growth Event).
    g = (parent_cell, daughter_1, daughter_2, t)
Triggered when S(parent) >= S* for T_consecutive >= 3.

Definition 2.2 (Cell Count).
    N(t) = number of active cells in the federation.

Definition 2.3 (Capacity Budget).
    Cap = sum over nodes of (VRAM_gb * safe_ratio)
Safe ratio default 0.85.

Definition 2.4 (Daughter Divergence).
    div(d1, d2) = 1 - cos( embed(d1.domain), embed(d2.domain) )
High div = daughters are meaningfully different.

Definition 2.5 (Growth Value).
    V(g) = benefit(g) - cost(g)
where:
    benefit(g) = expected K_6D gain from split
    cost(g)    = added inference cost + storage

## 3. Three Growth Constraints

### C1: Necessity
Split only if V(g) > 0, i.e.,
    benefit(g) > cost(g)

Benefit estimated from:
    saturation S(parent) (higher = more benefit)
    domain breadth W(parent) (wider = more room to split)
    gap queue pressure in parent's domain

Cost estimated from:
    parent's FLOPs per token
    expected daughter FLOPs
    storage footprint

### C2: Capacity
Total cells must satisfy:
    sum_c footprint(c) <= Cap

If split would exceed Cap:
    - Option A: evict a dormant cell first
    - Option B: postpone split until capacity frees
    - Option C: shrink cells to int8 to fit

Default: Option A if dormant cell exists, else B.

### C3: Diversity
Daughters must differ:
    div(d1, d2) >= tau_div (default 0.3)

If not, the split is rejected. Reason: non-diverse daughters
are redundant and waste capacity.

## 4. Main Theorem

Theorem 4.1 (Balanced Growth Equilibrium).
Under constraints C1-C3 and adaptive thresholds, the federation
converges to a stable cell count N* where:
    (i)   No further splits are beneficial (V(g) <= 0)
    (ii)  All cells fit in capacity (C2 satisfied)
    (iii) All daughters are diverse (C3 satisfied)

Proof sketch.
Each split either increases K_6D(F) or is rejected.
K_6D(F) is bounded above by 1 in each dimension.
Therefore splits strictly increase a bounded quantity.
By monotone convergence, splits must eventually stop.
The stopping point N* satisfies V(g) <= 0 by definition. ∎

Corollary 4.2 (Bounded Cell Count).
    N(t) <= Cap / min_cell_footprint

Corollary 4.3 (Bounded Latency).
Inference latency grows at most linearly with N(t),
and N(t) is bounded, so latency is bounded.

## 5. Growth Controller Algorithm

At every K steps, for each saturated cell c:

    step 1: compute V(split(c))
    step 2: if V <= 0: skip
    step 3: check capacity: if full, evict dormant or postpone
    step 4: propose daughters d1, d2
    step 5: compute div(d1, d2)
    step 6: if div < tau_div: try different split direction
    step 7: if all checks pass: execute split
    step 8: log event to registry

Zero-Error: any failed check => reject split, no partial state.

## 6. Zero-Error Guarantees

Property 6.1 (No Orphan Cells).
Every active cell has a parent (except the original seed cells).

Property 6.2 (Capacity Invariant).
At every step: sum_c footprint(c) <= Cap.
Assert before any split.

Property 6.3 (Reversibility).
Any split can be undone by merging daughters back.
Merge is allowed only if both are unsaturated.

Property 6.4 (Audit Trail).
Every split recorded in registry.growth_events with:
parent_id, d1_id, d2_id, V, div, timestamp.

## 7. Integration with Other Theories

With Theory 1 (Saturation):
    S(parent) >= S* triggers split proposal.
    S(daughters) initialized to 0, then grow.

With Theory 2 (Incorporation):
    High gap queue pressure in domain d increases benefit(g)
    for splits in domain d.

With Theory 3 (Zero-Forgetting):
    Split does not delete parent. Parent becomes dormant.
    Parent's knowledge is preserved for verification.

## 8. Implementation Notes

Registry table:
    growth_events(
        event_id, parent_id, d1_id, d2_id,
        V_benefit, V_cost, div, timestamp
    )

Saturation thresholds:
    S* = 0.85 (from Theory 1)
    T_consecutive = 3

Capacity:
    Cap computed from hw_profile at boot
    updated on hardware change

Diversity:
    tau_div = 0.3 (initial)
    embed(domain) via base embedding layer

Eviction:
    if Cap full and dormant cell exists: evict LRU dormant
    if no dormant: postpone

## 9. Open Questions

1. Is tau_div = 0.3 universal across domains?
2. How to measure benefit(g) empirically?
3. Can splits be batched to reduce overhead?
4. What if a domain needs more than Cap/2 cells?
5. Does balanced growth prevent specialization collapse?

## 10. Summary

Balanced growth is controlled by three constraints:
necessity (V > 0), capacity (sum fits budget), and diversity
(daughters differ). The controller converges to a stable
cell count N* bounded by hardware. Growth is reversible,
auditable, and zero-error.
