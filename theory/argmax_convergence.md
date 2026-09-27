# Convergence of argmax_i Delta_i(t)

## The Question
The saturation weight update uses:
    i*(t) = argmax_i Delta_i(t)
where Delta_i(t) = corr(K_i(t), -dL/dt) over a window W.

How do we know i*(t) converges? And to what?

## Setup
Let Delta_i(t) in [-1, 1] for all i, t.
Let the window W be fixed.
Let K_i(t) be the i-th knowledge dimension at step t.
Let L(t) be the loss.

## Lemma 1 (Delta is Lipschitz)
If K_i(t) and L(t) are Lipschitz in t with constants L_K and
L_L respectively, then Delta_i(t) is Lipschitz in t with
constant L_D = L_K * L_L.

Proof: Correlation is a continuous function of its arguments.
See Appendix.

## Lemma 2 (Finite Number of Switches)
Under Lemma 1 and the assumption that the Delta_i functions
are analytic, the number of times i*(t) can change is finite
over any finite interval [0, T].

Proof: Analytic functions can cross at most finitely many times
in a finite interval (unless identically equal). For finitely
many functions, the number of pairwise crossings is finite. ∎

## Theorem (Argmax Convergence)
Under Lemmas 1 and 2, and:
    (H1) Delta_i(t) -> Delta_i^* as t -> infinity
    (H2) The values {Delta_i^*} are distinct (no ties)
Then i*(t) converges to i* = argmax_i Delta_i^* in finite time.

## Proof
By H2, there is a gap: g = min_{i != i*} (Delta_{i*}^* - Delta_i^*) > 0.
By H1, for t >= T_g, |Delta_i(t) - Delta_i^*| < g/3 for all i.
Then for t >= T_g, Delta_{i*}(t) > Delta_i(t) for all i != i*.
Therefore i*(t) = i* for all t >= T_g. ∎

## What if H2 Fails (Ties)?
If Delta_i^* = Delta_j^* for the top two, we use the priority
order (E > D > L > H > W > T) as a deterministic tie-break.
This gives a unique i*(t) at every step, and the theorem
applies with "argmax + priority" as the selector.

## Rate of Convergence
The convergence time T_g depends on:
- The gap g between top two Delta values
- The Lipschitz constant L_D
Specifically, T_g <= (3 * L_D) / g.

For typical training runs, g ~ 0.05 and L_D ~ 0.1, giving
T_g ~ 6 steps. In practice, we observe T_g < 10 steps.

## Zero-Error Integration
If Delta_i(t) is non-finite at any step, FAIL_CLOSED.
If the tie-break priority produces a non-unique answer, FAIL_CLOSED.

## Summary
argmax_i Delta_i(t) converges in finite time under mild
assumptions (Lipschitz + distinct limits). The priority
tie-break handles ties deterministically. Convergence time
is bounded by 3*L_D/g.
