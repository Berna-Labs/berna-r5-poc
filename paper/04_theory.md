# 4. Theory

We present results in three categories:

- **2 Mathematical Theorems** (T1, T2): full proofs.
- **2 Design Invariants** (I1, I2): enforced by system design.
- **1 Proposition + 2 Hypotheses** (P1, H1, H2): partial
  results and empirical claims.

This classification is conservative. Not every result is a
theorem in the strict mathematical sense.

## 4.1 Hypothesis 1: Saturation-Triggered Splitting Equilibrium (6D)

**Statement.** Let K(t) in [0,1]^6 be the 6D knowledge vector
of a cell c. Let omega(t) be the adaptive weight vector in
the simplex. Define S(c,t) = ||K(c,t)||_omega. Under
assumptions A1-A3 (L-smooth, mu-strongly convex, sub-Gaussian
noise), there exists S* in (0,1) and finite T* such that:

(i) If S < S*, continued training weakly decreases L.
(ii) If S >= S* for T_consecutive >= 3, splitting c weakly
     decreases global loss.
(iii) K* is weighted-isotropic: omega_i * K*_i = omega_j * K*_j
      for all i, j.

**Key Lemma.** Under A1-A3, each dimension i converges at
rate O(1/sqrt(t)). The weighted norm ||K||_omega has a unique
fixed point at the isotropy condition.

**Constant.** S* = 0.85 (empirical). T_consecutive = 3.

**Intuition.** A cell is "saturated" when its six knowledge
dimensions are balanced and no further gradient signal points
toward improvement. Splitting at this point doubles capacity
along the dominant dimension.

## 4.2 Invariant 1: Transactional Incorporation

**Statement.** Let B be a base model with K_6D(B, t). Let s_i
be a specialist returning answer a with confidence c_i >=
TAU_CONF. Under the verification gate (logical + consistency
+ 6D checks), incorporating (r, a) into B:

(i) does not decrease K_6D(B) by more than epsilon_neg = 1e-6
(ii) strictly increases K_6D(B) along the dominant dimension
     i*(r) if the answer fills a gap
(iii) leaves K_6D(s_i) unchanged

**Key Lemma.** The verification gate explicitly rejects any
update that would decrease K_6D(B) beyond epsilon_neg. Frozen
specialists cannot change.

**Constant.** epsilon_neg = 1e-6, TAU_CONF = 0.8.

**Intuition.** Incorporation is transactional. Either it
passes all four verification checks and is integrated, or it
is rejected and queued as a gap. There is no partial state.

## 4.3 Hypothesis 2: Bounded Forgetting

**Statement.** Let T = {T_1, ..., T_M} be a fixed task set.
Define F_j(t) = max_{s <= t} P_j(s) - P_j(t). Under:
(H1) cell isolation, (H2) transactional incorporation,
(H3) persistent registry log, (H4) correct thresholds,

then for all j and all t: F_j(t) <= epsilon_neg = 1e-6.

**Key Lemma.** Frozen cells have requires_grad = False and
therefore cannot be modified by future training. New cells
do not overwrite old cells. Transactional rejection preserves
state.

**Constant.** F_j <= 1e-6 per task.

**Intuition.** Knowledge does not decay because it is stored
in immutable cells. Learning new tasks creates new cells,
not modifying old ones. This is the strongest form of
continual learning.

## 4.4 Theorem 2: Balanced Growth Equilibrium

**Statement.** Let g = (parent, d1, d2) be a split. Define
V(g) = benefit(g) - cost(g). Under:
(C1) V(g) > 0 required,
(C2) sum footprint <= Cap,
(C3) div(d1, d2) >= tau_div,

the federation converges to a stable cell count N* where:
(i) V(g) <= 0 for all further proposed splits,
(ii) capacity is always satisfied,
(iii) diversity is always satisfied.

**Key Lemma.** N(t) is bounded by Cap/min_footprint. Benefit
decreases as cells cover more of the K_6D space. Therefore
there is a point beyond which splits are no longer beneficial.

**Constant.** tau_div = 0.3, safe_ratio = 0.85.

**Intuition.** A federation with unlimited growth would
explode. Balanced growth uses three filters: necessity
(benefit > cost), capacity (fits in VRAM), diversity
(daughters differ meaningfully).

## 4.5 Theorem 1: Plexus Connectivity Convergence Connectivity Convergence

**Statement.** For each edge e, weight evolves as:
W_e(t+1) = W_e(t) + eta_e * (A_e(t) - lambda_e * W_e(t)).
Under (H1) stationary A_e(t), (H2) eta_e < 2/lambda_max,
(H3) lambda_e > 0, W converges exponentially to
W*_e = A*_e / lambda_e.

**Key Lemma.** The update is a linear time-invariant (LTI)
system: W(t+1) = M * W(t) + b with M = (1 - eta_e * lambda_e) I.
The spectral radius rho(M) = |1 - eta_e * lambda_e| < 1 under
H2, guaranteeing convergence.

**Constant.** eta_e = 0.01, lambda_e = 0.001, rho = 0.99999.

**Intuition.** The plexus is not fixed. Edges strengthen with
use and decay without use. The LTI analysis proves
convergence to a unique fixed point, making routing stable.

## 4.6 What the Theorems Do Not Claim

The theorems do NOT claim:
- The model is SOTA on any benchmark.
- The architecture is optimal.
- The 6 dimensions are the only useful ones.
- The 100 chromosomes are the correct number.
- The constants (S* = 0.85, tau_div = 0.3, etc.) are universal.

They DO claim:
- Each mechanism (saturation, incorporation, isolation,
  growth, connectivity) is mathematically well-defined.
- Each converges under explicit assumptions.
- Each constant is given explicitly and can be tested.

## 4.7 Falsifiability

Every theorem is falsifiable:

- T1: a single run with S > 0.9 failing to split refutes it.
- T2: an incorporation decreasing K_6D refutes it.
- T3: any task with F_j > 1e-6 refutes it.
- T4: N(t) > N_max refutes it.
- T5: divergence of W under H1-H3 refutes it.

Section 5 reports experiments that attempt such falsification.

## 4.8 Summary Table

    Theorem              | Constant         | Falsifiable Test
    ---------------------|------------------|------------------
    T1 Saturation        | S* = 0.85        | S convergence
    T2 Incorporation     | eps = 1e-6       | delta_K >= -eps
    T3 Zero-Forgetting   | F_j <= 1e-6      | forgetting monitor
    T4 Balanced Growth   | N* <= Cap/fp     | N(t) bound
    T5 Connectivity      | rho < 1          | weight convergence


## 4.9 Classification Summary

| ID | Type | Statement | Status |
|---|---|---|---|
| **T1** | Theorem | Plexus connectivity convergence (LTI) | Proven |
| **T2** | Theorem | Balanced growth equilibrium | Proven |
| **I1** | Invariant | Transactional incorporation consistency | Design-enforced |
| **I2** | Invariant | Protected retention under isolation | Design-enforced |
| **P1** | Proposition | Saturation state dynamics | Partial proof |
| **H1** | Hypothesis | Saturation-triggered split reduces loss | Empirical |
| **H2** | Hypothesis | Zero-forgetting on continual tasks | Empirical |

**Note on terminology:** we reserve "Theorem" for results with
complete mathematical proofs. "Invariant" denotes a property
enforced by system design (verifiable by code inspection, not
by proof). "Hypothesis" denotes an empirically testable claim.

## 4.10 What is NOT Proven

To be explicit about what this paper does NOT establish:

- **H1 is not proven.** The causal chain from saturation to
  loss decrease is not derived from first principles.
- **H2 is not proven.** We simulate zero-forgetting under
  isolation, but have not tested on real continual-learning
  benchmarks at scale.
- **S* = 0.85 is not derived.** It is empirically tuned.
- **100 chromosomes is not optimal.** It is a structured
  design prior.
- **6D state space is a modeling choice.** Other
  dimensionalities not explored.
- **Scale behavior is unknown.** Results at 50M may not hold
  at 1.5B.
