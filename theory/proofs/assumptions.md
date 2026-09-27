# Global Assumptions for Berna R5 Proofs

## A1. Loss Smoothness
The loss function L: R^n -> R is L-smooth:
    ||grad L(x) - grad L(y)|| <= L ||x - y|| for all x, y

## A2. Local Strong Convexity
In a ball B(x*, r) around a local minimum x*, L is mu-strongly
convex:
    L(y) >= L(x) + <grad L(x), y - x> + (mu/2) ||y - x||^2
for all x, y in B(x*, r).

## A3. Sub-Gaussian Noise
Gradient estimates g_t = grad L(x_t) + xi_t satisfy:
    E[xi_t] = 0
    E[||xi_t||^2] <= sigma^2
    xi_t is sub-Gaussian with parameter sigma

## A4. Bounded Activation
All activations a_i in the model satisfy |a_i| <= A for some A.

## A5. Bounded Gradients
All gradients grad L_i are bounded: ||grad L_i|| <= G.

## A6. Deterministic Randomness
All randomness uses a fixed seed. Given (seed, data_hash,
model_hash), all quantities are bit-identical across runs.

## A7. Zero-Error Invariant
No NaN or Inf value is ever propagated. Any such value triggers
immediate FAIL_CLOSED. Therefore all analyzed quantities are
finite by construction.

## A8. Finite Training
Training terminates at a finite horizon T (may be large, e.g.,
200,000 steps). Asymptotic statements apply to T -> infinity
as an idealization.

## A9. Task Set Fixed
The task set T = {T_1, ..., T_M} is fixed before training.
New tasks are added only through the incorporation protocol
(Theory 2).

## A10. Discrete Time
All dynamics are over discrete steps t = 0, 1, 2, ...
Continuous-time analogies are for intuition only.
