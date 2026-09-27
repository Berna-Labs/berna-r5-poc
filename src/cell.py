"""Knowledge Cell: born, split, die, interconnect."""

import math
import uuid
from typing import Optional

from .config import (
    S_STAR, S_LOW, E_MIN_FOR_SPLIT,
    T_CONSECUTIVE_SPLIT, T_CONSECUTIVE_DORMANCY,
    T_CONSECUTIVE_DEATH, BernaFatalError,
)
from .zero_error import assert_range, assert_dim_vector
from .saturation import (
    build_K, compute_saturation, dominant_dimension,
    uniform_omega, update_omega, should_split,
    should_go_dormant, can_split_with_E,
)


def new_cell_id() -> str:
    return str(uuid.uuid4())


class KnowledgeCell:
    """A single knowledge cell in the Berna R5 federation."""

    def __init__(self, domain: str, owner_id: str,
                 owner_type: str, birth_step: int,
                 parent_cell: Optional[str] = None):
        if owner_type not in ("base", "specialist"):
            raise BernaFatalError(f"Invalid owner_type: {owner_type}")
        self.cell_id = new_cell_id()
        self.domain = domain
        self.owner_id = owner_id
        self.owner_type = owner_type
        self.birth_step = birth_step
        self.parent_cell = parent_cell

        # State
        self.state = "active"
        self.omega = uniform_omega()
        self.K = build_K(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # T starts fresh
        self.S = 0.0
        self.utilization = 0.0

        # History for trigger decisions
        self.S_history = []
        self.age = 0  # steps since last update

        # Parameters placeholder (weights will be attached later)
        self.weights = None  # nn.Parameter dict

    # ----------------------------------------------------------
    # Update
    # ----------------------------------------------------------
    def update(self, K_new: dict, utilization: float,
               delta_omega: Optional[dict] = None) -> None:
        """Update 6D knowledge and recompute saturation."""
        assert_dim_vector(K_new, ["L", "W", "H", "D", "T", "E"])
        assert_range(utilization, 0.0, 1.0, "utilization")

        self.K = K_new
        self.utilization = utilization
        self.age = 0

        if delta_omega is not None:
            self.omega = update_omega(self.omega, delta_omega)

        self.S = compute_saturation(self.K, self.omega)
        self.S_history.append(self.S)
        if len(self.S_history) > 1000:
            self.S_history = self.S_history[-1000:]

    def tick(self) -> None:
        """Advance age by one step (called every training step)."""
        self.age += 1

    # ----------------------------------------------------------
    # Trigger decisions (Theory 1 corollaries)
    # ----------------------------------------------------------
    def check_split(self) -> bool:
        if self.state != "active":
            return False
        if not should_split(self.S_history, T_CONSECUTIVE_SPLIT):
            return False
        if not can_split_with_E(self.K):
            return False
        return True

    def check_dormancy(self) -> bool:
        if self.state != "active":
            return False
        return should_go_dormant(
            self.S_history, self.utilization, T_CONSECUTIVE_DORMANCY
        )

    def check_death(self) -> bool:
        if self.state != "active":
            return False
        if len(self.S_history) < T_CONSECUTIVE_DEATH:
            return False
        recent = self.S_history[-T_CONSECUTIVE_DEATH:]
        return all(s < S_LOW for s in recent) and self.utilization < 0.01

    # ----------------------------------------------------------
    # Split (Theory 4)
    # ----------------------------------------------------------
    def split(self) -> tuple:
        """Split into two daughters. Parent becomes dormant."""
        if not self.check_split():
            raise BernaFatalError(
                f"Cannot split cell {self.cell_id}: conditions not met"
            )
        i_star = dominant_dimension(self.K, self.omega)

        # Daughter domains: parent domain + dimension suffix
        d1 = KnowledgeCell(
            domain=f"{self.domain}:{i_star}_a",
            owner_id=self.owner_id,
            owner_type=self.owner_type,
            birth_step=self.birth_step,
            parent_cell=self.cell_id,
        )
        d2 = KnowledgeCell(
            domain=f"{self.domain}:{i_star}_b",
            owner_id=self.owner_id,
            owner_type=self.owner_type,
            birth_step=self.birth_step,
            parent_cell=self.cell_id,
        )
        # Inherit K partially along split dimension
        d1.K = dict(self.K); d1.K["T"] = 1.0
        d2.K = dict(self.K); d2.K["T"] = 1.0
        d1.omega = dict(self.omega)
        d2.omega = dict(self.omega)
        d1.S = compute_saturation(d1.K, d1.omega)
        d2.S = compute_saturation(d2.K, d2.omega)

        self.state = "dormant"
        return d1, d2

    # ----------------------------------------------------------
    # State transitions
    # ----------------------------------------------------------
    def go_dormant(self) -> None:
        if self.state != "active":
            raise BernaFatalError(f"Cannot dormancy from state {self.state}")
        self.state = "dormant"

    def freeze(self) -> None:
        if self.state == "dead":
            raise BernaFatalError("Cannot freeze dead cell")
        self.state = "frozen"

    def die(self) -> None:
        if self.state == "dead":
            raise BernaFatalError("Already dead")
        self.state = "dead"

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------
    def info(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "domain": self.domain,
            "state": self.state,
            "S": round(self.S, 4),
            "K": {k: round(v, 3) for k, v in self.K.items()},
            "omega": {k: round(v, 4) for k, v in self.omega.items()},
            "utilization": round(self.utilization, 3),
            "age": self.age,
            "parent_cell": self.parent_cell,
        }
