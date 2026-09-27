# Berna R5 — Corrections Manifest v2.0

**DOI:** 10.5281/zenodo.22998311 (unchanged)
**Date:** 2026-09-27
**Version:** v2.0

This document records the corrections applied to the Berna R5 paper
following internal review. It documents what changed and why.

---

## 1. Title

**Before:** DNA-Kernel Plexus: A Bio-Inspired Architecture for Growing,
Self-Correcting Language Models with **Proven Zero-Forgetting**

**After:** DNA-Kernel Plexus: A Bio-Inspired Architecture for Growing,
Self-Correcting Language Models with **Bounded Forgetting**

**Reason:** "Proven" overstated the nature of the zero-forgetting result.
The result is a design invariant under explicit isolation assumptions,
not an unconditional mathematical theorem. "Bounded" is more accurate.

---

## 2. Classification of Results

**Before:** "We prove five theorems."

**After:** A mix of theorems, invariants, propositions, and hypotheses:

| ID | Type | Statement |
|---|---|---|
| **Theorem 1** | Mathematical | Plexus connectivity convergence (LTI analysis) |
| **Theorem 2** | Mathematical | Balanced growth equilibrium |
| **Invariant 1** | Design | Transactional incorporation consistency |
| **Invariant 2** | Design | Protected knowledge retention under isolation |
| **Proposition 1** | Mathematical (partial) | Saturation state dynamics convergence |
| **Hypothesis H1** | Empirical | Saturation-triggered split decreases loss |
| **Hypothesis H2** | Empirical | Zero-forgetting holds on continual tasks |

**Reason:** Distinguishing mathematical theorems from design invariants
from empirical hypotheses is more honest. Reviewers can evaluate each
category separately.

---

## 3. Terminology Changes

| Before | After | Reason |
|---|---|---|
| "6D manifold" | "6D knowledge state space" | State space is measurable; manifold implies structure not yet proven |
| "100 chromosomes information-theoretically justified" | "100 chromosomes as a structured design prior" | No formal derivation exists for exactly 100 |
| "Zero-Error Creed" | "Fail-Closed Integrity Invariant" | "Creed" is rhetoric; "Integrity Invariant" is technical |
| "S* = 0.85 theoretically derived" | "S* = 0.85 empirically tuned" | No theorem gives 0.85 |

---

## 4. Related Work Additions

Added sections on:
- **Model Editing:** ROME, MEMIT, MEND, SERAC
- **Dynamic Networks:** DEN (explicit), Net2Net, Progressive Nets
- **NCA:** Growing Neural Cellular Automata

---

## 5. Abstract Rewrite

Removed: "We prove five theorems... zero-forgetting"
Added: "We present... under explicit isolation assumptions..."

---

## 6. What Did NOT Change

The following remain as originally published:

- The 6D state space design (L, W, H, D, T, E)
- The DNA Kernel architecture (100 chromosomes, 10 groups)
- The Plexus co-activation dynamics
- The Transactional incorporation protocol
- The SQL registry + lineage system
- The Zero-Error / Fail-Closed runtime design
- All 155 tests
- All source code
- The 6D Theorem-1 convergence proof (T5, now Theorem 1)
- The 6D Theorem-2 balanced growth proof (T4, now Theorem 2)

---

## 7. What Remains to be Proven Empirically

These are the open challenges that make Berna R5 an ongoing
research project rather than a completed result:

1. **H1:** Saturation-triggered split decreases loss (not yet empirically validated at scale)
2. **H2:** Zero-forgetting on real continual tasks (only simulated so far)
3. **Scale:** Behavior at 1.5B parameters
4. **Comparison:** vs. EWC, PackNet, MoE on standard benchmarks
5. **Causal cells:** Cell activation must causally affect output, not just metadata

---

## 8. Version History

- **v1.0** — 2026-09-27 — Initial preprint on Zenodo (DOI: 10.5281/zenodo.22998311)
- **v2.0** — 2026-09-27 — Corrections to title, terminology, classification

---

## 9. Acknowledgments

We thank the reviewer whose critique prompted these corrections.
Their feedback strengthened the paper.
