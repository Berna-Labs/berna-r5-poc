# 1. Introduction

Large language models (LLMs) have achieved remarkable
performance across diverse tasks, yet their architectures
remain fundamentally static. Once training completes, an
LLM's parameters are frozen: it cannot grow new capacity,
cannot correct specific knowledge without retraining,
and cannot forget selectively. This rigidity creates three
well-documented failures:

**Failure 1 (Fixed Capacity).** A model deployed on a new
domain either overfits (small model) or wastes compute
(large model). No mechanism exists to grow capacity only
when needed.

**Failure 2 (Catastrophic Forgetting).** Fine-tuning on a
new task destroys performance on previous tasks. Existing
solutions (EWC, PackNet, LoRA) reduce forgetting but cannot
eliminate it.

**Failure 3 (No Localized Correction).** Correcting a single
fact requires retraining millions of parameters. There is no
surgical mechanism to update one piece of knowledge without
perturbing the rest.

### 1.1 The Gap

Three research communities have attacked these failures
separately.

**Mixture-of-Experts (MoE)** addresses capacity by activating
a subset of experts per input. But experts are fixed at
initialization: no new experts can be added without
retraining the router. MoE also does not address forgetting
or localized correction.

**Continual Learning** addresses forgetting through
regularization (EWC) or parameter isolation (PackNet,
Progressive Nets). But these methods apply to small models
and do not scale to billion-parameter LLMs. They also do
not provide growth.

**Retrieval-Augmented Generation (RAG)** addresses localized
knowledge by retrieving from an external store. But RAG
does not modify model weights: it cannot learn new
capabilities, only surface existing text.

No existing work unifies growth, federation, and
zero-forgetting under a single theoretical framework.

### 1.2 Our Contribution

We present Berna R5, an architecture that addresses all
three failures simultaneously. Our contributions are:

**Contribution 1 (Architecture).** A model in which
knowledge is stored in dynamic weighted cells. Cells are
born when the model encounters new domains, split when
saturated, and interconnect via a co-activation plexus.
A 100-chromosome DNA Kernel regulates cell behavior.

**Contribution 2 (Theory).** Five theorems with complete
proofs:

- **Theorem 1** (Saturation Equilibrium): the 6-dimensional
  knowledge vector K converges to a weighted-isotropic
  point with threshold S* = 0.85.
- **Theorem 2** (Reliable Incorporation): incorporating
  knowledge from a specialist into the base does not
  decrease any dimension of K beyond epsilon_neg = 1e-6.
- **Theorem 3** (Zero-Forgetting): under cell isolation
  and transactional incorporation, the forgetting rate is
  provably zero per billion tokens.
- **Theorem 4** (Balanced Growth): cell count converges to
  a stable N* bounded by hardware capacity.
- **Theorem 5** (Plexus Convergence): edge weights
  converge exponentially to W* = A* / lambda_e.

**Contribution 3 (Empirical Validation).** Seven
experiments on a 50M-parameter PoC trained on 800M tokens,
covering saturation dynamics, growth behavior,
incorporation consistency, zero-forgetting, baseline
comparison, plexus dynamics, and ablations.

**Contribution 4 (Open Source).** A complete implementation
including SQL registry, hardware-adaptive deployment, and
atomic checkpointing for power-loss recovery.

### 1.3 What We Do Not Claim

We make no claim of SOTA on any benchmark. Our model is
50M parameters, far smaller than production LLMs. We
compare against 50M baselines, not GPT-4-class systems.

We make no claim of theoretical completeness. All five
theorems rest on explicitly stated assumptions (see
`theory/proofs/assumptions.md`). When those assumptions
fail, the theorems do not apply. Counterexamples are
documented in each proof.

We make no claim that 100 chromosomes or 6 dimensions are
optimal. Both are justified heuristically and
information-theoretically (see
`theory/justification_100_chromosomes.md`), but ablation
studies are needed to confirm.

### 1.4 The Fail-Closed Integrity Invariant

A central design principle of Berna R5 is the Zero-Error
Invariant: any violation of a declared invariant triggers
immediate FAIL_CLOSED. There are no silent failures. This
is a runtime guarantee, not a mathematical claim.

The invariant covers ten classes: NaN/Inf in any tensor,
out-of-range values, non-normalized weights, K outside
[0,1], division by zero, disconnected plexus nodes,
power-loss recovery, missing registry, duplicate UUIDs,
and invalid signatures.

We do not claim that this invariant makes the model
infallible. We claim that any error is documented,
reproducible, and falsifiable.

### 1.5 Paper Structure

Section 2 reviews related work in MoE, continual learning,
NAS, and retrieval. Section 3 describes the architecture.
Section 4 presents the theoretical results. Section 5 presents
the seven experiments. Section 6 discusses findings.
Section 7 concludes. Section 8 documents limitations.

Full proofs are in Appendix A. Implementation details are
in Appendix B. Additional experiments are in Appendix C.
