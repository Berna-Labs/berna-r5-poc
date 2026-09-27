"""Theory 1: Knowledge Saturation in 6D Manifold."""

import math
from typing import Optional

from .config import (
    DIM_ORDER, DIM_PRIORITY, ETA_OMEGA, E_MIN_FOR_SPLIT,
    S_STAR, S_LOW, EPSILON_NEG, BernaFatalError,
)
from .zero_error import assert_dim_vector, assert_range, assert_simplex


# ============================================================
# DIMENSION SCORERS (each returns value in [0, 1])
# ============================================================
def score_length(correct_positions: list, seq_len: int) -> float:
    """L = max distance a cell correctly predicts, normalized."""
    if seq_len <= 0:
        raise BernaFatalError(f"seq_len must be positive, got {seq_len}")
    if not correct_positions:
        return 0.0
    return min(1.0, max(correct_positions) / seq_len)


def score_width(domain_counts: dict, total_domains: int,
                tau: float = 0.1) -> float:
    """W = fraction of domains where cell is active above tau."""
    if total_domains <= 0:
        raise BernaFatalError(f"total_domains must be positive")
    active = sum(1 for v in domain_counts.values() if v > tau)
    return min(1.0, active / total_domains)


def score_height(activation_entropy: float, n_cells: int) -> float:
    """H = 1 - normalized entropy. High H = focused representation."""
    if n_cells <= 1:
        return 0.0
    max_entropy = math.log(n_cells)
    if max_entropy <= 0:
        return 0.0
    H = 1.0 - min(1.0, activation_entropy / max_entropy)
    return max(0.0, min(1.0, H))


def score_depth(path_length: float, max_path: float) -> float:
    """D = effective computation path length / max."""
    if max_path <= 0:
        raise BernaFatalError(f"max_path must be positive")
    return max(0.0, min(1.0, path_length / max_path))


def score_time(age_steps: int, decay_lambda: float = 1e-4,
               freshness: float = 1.0) -> float:
    """T = exp(-lambda * age) * freshness."""
    if age_steps < 0:
        raise BernaFatalError(f"age_steps must be >= 0, got {age_steps}")
    T = math.exp(-decay_lambda * age_steps) * freshness
    return max(0.0, min(1.0, T))


def score_encompassment(facet_coverages: dict) -> float:
    """E = average of min(1, coverage) across all facets."""
    if not facet_coverages:
        return 0.0
    vals = [min(1.0, max(0.0, c)) for c in facet_coverages.values()]
    return sum(vals) / len(vals)


# ============================================================
# K VECTOR
# ============================================================
def build_K(L: float, W: float, H: float, D: float,
            T: float, E: float) -> dict:
    """Build and validate 6D knowledge vector."""
    K = {"L": L, "W": W, "H": H, "D": D, "T": T, "E": E}
    assert_dim_vector(K, DIM_ORDER)
    return K


# ============================================================
# SATURATION SCORE
# ============================================================
def compute_saturation(K: dict, omega: dict) -> float:
    """S = ||K||_omega, weighted 6D norm."""
    assert_dim_vector(K, DIM_ORDER)
    assert_simplex([omega[d] for d in DIM_ORDER])
    s_sq = sum(omega[d] * K[d] ** 2 for d in DIM_ORDER)
    S = math.sqrt(max(0.0, s_sq))
    assert_range(S, 0.0, 1.0, "S")
    return S


def dominant_dimension(K: dict, omega: dict) -> str:
    """i* = argmax omega_i * K_i, tie-break by DIM_PRIORITY."""
    scores = [(omega[d] * K[d], DIM_PRIORITY[d], d) for d in DIM_ORDER]
    scores.sort(key=lambda x: (-x[0], x[1]))
    return scores[0][2]


# ============================================================
# ADAPTIVE OMEGA UPDATE
# ============================================================
def update_omega(omega: dict, delta: dict,
                 eta: float = ETA_OMEGA) -> dict:
    """omega_i(t+1) = omega_i(t) * exp(eta * Delta_i) / Z(t)."""
    assert_simplex([omega[d] for d in DIM_ORDER])
    for d in DIM_ORDER:
        assert_range(delta[d], -1.0, 1.0, f"delta[{d}]")
    new_w = {d: omega[d] * math.exp(eta * delta[d]) for d in DIM_ORDER}
    Z = sum(new_w.values())
    if Z <= 0:
        raise BernaFatalError(f"Omega normalization Z={Z}")
    return {d: new_w[d] / Z for d in DIM_ORDER}


def uniform_omega() -> dict:
    """Uniform initialization: each dim = 1/6."""
    return {d: 1.0 / len(DIM_ORDER) for d in DIM_ORDER}


# ============================================================
# TRIGGERS (Theory 1 corollaries)
# ============================================================
def should_split(S_history: list, T_consecutive: int = 3) -> bool:
    if len(S_history) < T_consecutive:
        return False
    recent = S_history[-T_consecutive:]
    return all(s >= S_STAR for s in recent)


def should_go_dormant(S_history: list, utilization: float,
                      T_consecutive: int = 50) -> bool:
    if len(S_history) < T_consecutive:
        return False
    recent = S_history[-T_consecutive:]
    return all(s < S_LOW for s in recent) and utilization < 0.1


def can_split_with_E(K: dict) -> bool:
    """Corollary 5.3: preserve E during split."""
    return K["E"] >= E_MIN_FOR_SPLIT
