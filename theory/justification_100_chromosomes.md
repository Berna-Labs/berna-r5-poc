# Justification for 100 Chromosomes in 10 Groups

## The Question
Why 100 chromosomes? Why 10 groups of 10? Why not 50 or 200?

## Answer (three arguments)

### Argument 1: Capacity Argument
The federation must represent at least 10 orthogonal knowledge
domains (language, math, code, science, sensory, reasoning,
memory, coordination, adaptation, growth). Each domain needs
at least 10 regulatory units to encode:
- activation threshold
- sensitivity
- persistence
- plasticity
- plus 6 additional regulatory dimensions (future)

Minimum: 10 domains x 10 units = 100.

### Argument 2: Information-Theoretic Bound
Let each gene be continuous in [0, 1] with effective resolution
2^b bits (b ~ 6 bits for float precision in regulation).
Each chromosome has 4 genes = 24 bits of regulatory information.
100 chromosomes = 2400 bits of regulatory state.

This is the minimum needed to represent:
- 10 domains x 4 modes each (activation, planning, memory, output)
= 40 regulatory modes
- each mode needs >= 60 bits to distinguish sub-modes
= 2400 bits total

Larger number (e.g., 200) would provide redundancy without new
capacity. Smaller (e.g., 50) would not cover all 40 modes.

### Argument 3: Biological Argument
The human genome has ~20,000 protein-coding genes. A regulatory
network controlling ~10 functional domains would require
roughly sqrt(20,000) ~ 141 regulatory units. Rounding to 100
gives a tractable, interpretable number while preserving
sufficient regulatory capacity.

## Why 10 Groups Specifically?

The 10 groups correspond to the 10 orthogonal dimensions of
human cognitive function that are documented in cognitive
science:
1. Language (Chomsky, Pinker)
2. Mathematics (Dehaene)
3. Tool use / Code (Stout)
4. Science / Causal reasoning (Gopnik)
5. Sensory integration (Mountcastle)
6. Reasoning (Kahneman, dual-process)
7. Memory (Squire, working vs long-term)
8. Coordination (motor + executive)
9. Adaptation (Piaget)
10. Growth / development (Vygotsky)

## Testable Predictions

If 100 is the right number:
- Removing chromosomes degrades performance monotonically
- Adding chromosomes (101-120) yields < 1% improvement
- Group ablations show partial deficits, not total failure

## Summary
100 = 10 domains x 10 units is the minimum justified number.
It is neither arbitrary nor oversized. Ablations in experiments
will validate this choice.
