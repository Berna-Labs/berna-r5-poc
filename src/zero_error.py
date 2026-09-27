"""Zero-Error helpers. All violations raise BernaFatalError."""

import math
from typing import Any, Iterable

from .config import BernaFatalError


def assert_finite(x: Any, name: str = "value") -> None:
    """Assert scalar or array is finite."""
    if isinstance(x, (int, float)):
        if not math.isfinite(x):
            raise BernaFatalError(f"{name} is not finite: {x}")
    else:
        try:
            import numpy as np
            if not np.all(np.isfinite(x)):
                raise BernaFatalError(f"{name} contains non-finite values")
        except ImportError:
            pass


def assert_range(x: float, lo: float, hi: float, name: str = "value") -> None:
    """Assert scalar in [lo, hi]."""
    assert_finite(x, name)
    if not (lo <= x <= hi):
        raise BernaFatalError(f"{name}={x} out of range [{lo}, {hi}]")


def assert_simplex(w: Iterable[float], name: str = "weights") -> None:
    """Assert weights sum to 1 and all >= 0."""
    w = list(w)
    if len(w) == 0:
        raise BernaFatalError(f"{name} is empty")
    if any(x < 0 for x in w):
        raise BernaFatalError(f"{name} has negative entries")
    total = sum(w)
    if abs(total - 1.0) > 1e-6:
        raise BernaFatalError(f"{name} sum={total} != 1.0")


def assert_dim_vector(K: dict, dims: list) -> None:
    """Assert 6D knowledge vector: all dims in [0,1], finite."""
    for d in dims:
        if d not in K:
            raise BernaFatalError(f"Missing dimension: {d}")
        assert_range(K[d], 0.0, 1.0, f"K[{d}]")


def fail(msg: str) -> None:
    """Explicit fail-closed."""
    raise BernaFatalError(msg)
