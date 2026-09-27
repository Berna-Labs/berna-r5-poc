"""Chromosome: a regulatory unit in the DNA Kernel."""

import math
import uuid
from typing import Optional

from .config import BernaFatalError
from .zero_error import assert_range, assert_finite


def new_chromosome_id() -> str:
    return str(uuid.uuid4())


# 10 domain groups x 10 chromosomes each = 100
DOMAIN_GROUPS = [
    "language", "math", "code", "science", "sensory",
    "reasoning", "memory", "coordination", "adaptation", "growth",
]


class Chromosome:
    """A single chromosome: regulatory genes for a specific role."""

    def __init__(self, group: str, index: int):
        if group not in DOMAIN_GROUPS:
            raise BernaFatalError(f"Invalid domain group: {group}")
        if not (0 <= index < 10):
            raise BernaFatalError(f"Index must be in [0, 9], got {index}")

        self.chromosome_id = new_chromosome_id()
        self.group = group
        self.index = index
        self.full_name = f"{group}.{index:02d}"

        # Gene expression levels (learned)
        # Each chromosome has 4 genes controlling behavior
        self.genes = {
            "activation": 0.0,   # how readily this chromosome expresses
            "sensitivity": 0.5,  # threshold for triggering
            "persistence": 0.5,  # how long expression lasts
            "plasticity": 0.1,   # how much it can mutate
        }

        # Expression state
        self.expression = 0.0
        self.history = []

    # ----------------------------------------------------------
    # Gene expression
    # ----------------------------------------------------------
    def express(self, context: dict) -> float:
        """Compute expression level given context signals."""
        # context: dict of signal values in [0, 1]
        # e.g., {'loss_grad': 0.3, 'entropy': 0.6, ...}

        if not context:
            self.expression = 0.0
            return 0.0

        # Simple weighted expression
        signals = list(context.values())
        for s in signals:
            assert_range(s, 0.0, 1.0, "context signal")

        mean_signal = sum(signals) / len(signals)
        # Expression = sigmoid(activation + mean_signal - sensitivity)
        z = (self.genes["activation"]
             + mean_signal
             - self.genes["sensitivity"])
        expr = 1.0 / (1.0 + math.exp(-z))
        assert_finite(expr, "expression")

        self.expression = expr
        self.history.append(expr)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return expr

    # ----------------------------------------------------------
    # Mutation (Theory 2: reliable mutation)
    # ----------------------------------------------------------
    def mutate(self, delta: dict, rate: float = 0.01) -> None:
        """Apply a verified delta to gene values. Rate scales change."""
        assert_range(rate, 0.0, 1.0, "rate")
        for key in delta:
            if key not in self.genes:
                raise BernaFatalError(f"Unknown gene: {key}")
            assert_range(delta[key], -1.0, 1.0, f"delta[{key}]")
            new_val = self.genes[key] + rate * delta[key]
            # Clamp to [0, 1]
            self.genes[key] = max(0.0, min(1.0, new_val))

    # ----------------------------------------------------------
    # Saturation (per-chromosome)
    # ----------------------------------------------------------
    def saturation(self) -> float:
        """How saturated is this chromosome's regulatory role?"""
        if len(self.history) < 10:
            return 0.0
        recent = self.history[-10:]
        variance = max(recent) - min(recent)
        # High persistence + low variance = saturated
        return min(1.0, self.genes["persistence"] * (1.0 - variance))

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------
    def info(self) -> dict:
        return {
            "id": self.chromosome_id[:8],
            "name": self.full_name,
            "genes": {k: round(v, 4) for k, v in self.genes.items()},
            "expression": round(self.expression, 4),
            "saturation": round(self.saturation(), 4),
        }
