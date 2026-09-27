"""DNA Kernel: 100-chromosome regulatory layer."""

import hashlib
import json
from typing import Optional

from .chromosome import Chromosome, DOMAIN_GROUPS
from .config import BernaFatalError
from .zero_error import assert_range, assert_finite


class DNAKernel:
    """Regulatory layer of 100 chromosomes.

    Groups:
      language, math, code, science, sensory,
      reasoning, memory, coordination, adaptation, growth
    Each group has 10 chromosomes (indices 0-9).
    """

    def __init__(self):
        self.chromosomes = {}  # full_name -> Chromosome
        for group in DOMAIN_GROUPS:
            for i in range(10):
                c = Chromosome(group, i)
                self.chromosomes[c.full_name] = c
        if len(self.chromosomes) != 100:
            raise BernaFatalError(
                f"Expected 100 chromosomes, got {len(self.chromosomes)}"
            )

    # ----------------------------------------------------------
    # Expression
    # ----------------------------------------------------------
    def express_group(self, group: str, context: dict) -> float:
        """Mean expression of all chromosomes in a group."""
        if group not in DOMAIN_GROUPS:
            raise BernaFatalError(f"Invalid group: {group}")
        vals = []
        for i in range(10):
            c = self.chromosomes[f"{group}.{i:02d}"]
            vals.append(c.express(context))
        return sum(vals) / len(vals)

    def express_all(self, context: dict) -> dict:
        """Express all 10 groups given a shared context."""
        return {g: self.express_group(g, context) for g in DOMAIN_GROUPS}

    def dominant_group(self, context: dict) -> str:
        """Which group has highest expression for this context?"""
        exprs = self.express_all(context)
        return max(exprs, key=lambda g: exprs[g])

    # ----------------------------------------------------------
    # Mutation (Theory 2: reliable mutation)
    # ----------------------------------------------------------
    def mutate_chromosome(self, full_name: str, delta: dict,
                          rate: float = 0.01) -> None:
        """Verified mutation of a single chromosome's genes."""
        if full_name not in self.chromosomes:
            raise BernaFatalError(f"Unknown chromosome: {full_name}")
        self.chromosomes[full_name].mutate(delta, rate=rate)

    # ----------------------------------------------------------
    # Saturation (aggregate)
    # ----------------------------------------------------------
    def group_saturation(self, group: str) -> float:
        """Mean saturation across 10 chromosomes in a group."""
        vals = [self.chromosomes[f"{group}.{i:02d}"].saturation()
                for i in range(10)]
        return sum(vals) / len(vals)

    def total_saturation(self) -> float:
        """Mean saturation across all 100 chromosomes."""
        vals = [c.saturation() for c in self.chromosomes.values()]
        return sum(vals) / len(vals)

    # ----------------------------------------------------------
    # Replication (for cell split)
    # ----------------------------------------------------------
    def replicate(self, mutation_rate: float = 0.005) -> "DNAKernel":
        """Produce a new kernel with slight gene perturbation."""
        new_kernel = DNAKernel()
        for name, c in self.chromosomes.items():
            new_c = new_kernel.chromosomes[name]
            # Copy genes with small noise
            for key, val in c.genes.items():
                # deterministic small shift based on name hash
                seed = int(hashlib.sha256(
                    f"{name}:{key}".encode()
                ).hexdigest()[:8], 16)
                noise = ((seed % 1000) / 1000.0 - 0.5) * 2 * mutation_rate
                new_val = max(0.0, min(1.0, val + noise))
                new_c.genes[key] = new_val
        return new_kernel

    # ----------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------
    def state_hash(self) -> str:
        """Deterministic hash of gene state (for registry audit)."""
        payload = {
            name: {
                k: round(v, 6) for k, v in c.genes.items()
            }
            for name, c in sorted(self.chromosomes.items())
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> dict:
        return {
            name: dict(c.genes) for name, c in self.chromosomes.items()
        }

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------
    def stats(self) -> dict:
        expr_means = {}
        sat_means = {}
        for g in DOMAIN_GROUPS:
            expr_means[g] = round(
                sum(self.chromosomes[f"{g}.{i:02d}"].expression
                    for i in range(10)) / 10, 4
            )
            sat_means[g] = round(self.group_saturation(g), 4)
        return {
            "total_chromosomes": len(self.chromosomes),
            "total_saturation": round(self.total_saturation(), 4),
            "group_saturation": sat_means,
            "state_hash": self.state_hash()[:16],
        }
