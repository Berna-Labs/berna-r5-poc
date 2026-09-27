# The Zero-Error Creed

## Definition
The Zero-Error Invariant is a runtime guarantee, not a
mathematical claim. It states:

    Any violation of a declared invariant triggers
    immediate FAIL_CLOSED. There are no silent failures.

## What It Covers
1. NaN/Inf in any tensor
2. Out-of-range values
3. Non-normalized weights
4. K_6D outside [0, 1]
5. Division by zero
6. Disconnected plexus nodes
7. Power-loss recovery
8. Missing registry
9. Duplicate UUIDs
10. Invalid signatures

## What It Does NOT Cover
- Completeness of mathematical proofs
- Empirical validation at scale
- Comparison with baselines
- Optimality of design choices (100 chromosomes, 6 dims)

## The Honest Creed
We do not claim:
- "Our theory is complete."
- "Our model is the best."
- "Every design choice is optimal."

We DO claim:
- "No runtime error goes undetected."
- "Every limitation is documented."
- "Every claim is falsifiable."

## How to Falsify
Find a single input that produces a silent failure.
We welcome such attempts. Falsification is progress.

## Why This Matters
Science advances through falsifiable claims, not through
absolute proclamations. A creed is credible only when
it is bounded.
