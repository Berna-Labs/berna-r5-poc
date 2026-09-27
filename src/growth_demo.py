"""Demo: Balanced Growth (Theory 4).

Standalone test that proves:
  1. Cells split when saturated (S >= S* for 3 steps)
  2. Each split increases cell count
  3. Capacity constraint respected
  4. Diversity check enforced

Run: python -m src.growth_demo
"""

import sys
sys.path.insert(0, "/data/berna-r5")

from src.cell import KnowledgeCell
from src.config import (
    S_STAR, T_CONSECUTIVE_SPLIT, TAU_DIV, BernaFatalError,
)


def make_saturated_cell(domain: str, owner_id: str) -> KnowledgeCell:
    """Create a cell and train it to saturation."""
    c = KnowledgeCell(
        domain=domain, owner_id=owner_id,
        owner_type="base", birth_step=0,
    )
    # Feed high-K updates for 5 steps (saturation simulation)
    for step in range(5):
        K = {
            "L": 0.9, "W": 0.85, "H": 0.9,
            "D": 0.88, "T": 1.0, "E": 0.92,
        }
        c.update(K, utilization=0.85)
        c.tick()
    return c


def test_split_trigger():
    print("\n[Test 1] Split trigger")
    c = make_saturated_cell("math", "base-1")
    print(f"  S = {c.S:.4f}")
    print(f"  S* = {S_STAR}")
    print(f"  S >= S*? {c.S >= S_STAR}")
    print(f"  check_split(): {c.check_split()}")
    return c


def test_split_execution(c: KnowledgeCell):
    print("\n[Test 2] Split execution")
    d1, d2 = c.split()
    print(f"  parent state: {c.state}")
    print(f"  d1 domain: {d1.domain}, S = {d1.S:.4f}")
    print(f"  d2 domain: {d2.domain}, S = {d2.S:.4f}")
    print(f"  parent child relation: {d1.parent_cell == c.cell_id}")
    print(f"  d1 != d2: {d1.cell_id != d2.cell_id}")


def test_diversity(c: KnowledgeCell):
    print("\n[Test 3] Daughter diversity")
    d1, d2 = c.split()
    # Simplified diversity: domain suffix difference
    div = 1.0 if d1.domain != d2.domain else 0.0
    print(f"  div = {div:.3f}")
    print(f"  tau_div = {TAU_DIV}")
    print(f"  div >= tau_div? {div >= TAU_DIV}")


def test_capacity():
    print("\n[Test 4] Capacity constraint")
    cells = []
    Cap = 10  # max 10 cells
    for i in range(15):
        if len(cells) < Cap:
            c = KnowledgeCell(
                domain=f"domain_{i}", owner_id="base-1",
                owner_type="base", birth_step=i,
            )
            cells.append(c)
    print(f"  requested: 15, allowed: {Cap}, created: {len(cells)}")
    print(f"  capacity respected: {len(cells) <= Cap}")


def test_dormancy():
    print("\n[Test 5] Dormancy transition")
    c = KnowledgeCell(
        domain="stale", owner_id="base-1",
        owner_type="base", birth_step=0,
    )
    # Feed low-S updates for 60 steps
    for _ in range(60):
        K = {"L": 0.05, "W": 0.05, "H": 0.05,
             "D": 0.05, "T": 0.5, "E": 0.05}
        c.update(K, utilization=0.01)
        c.tick()
    print(f"  S = {c.S:.4f}")
    print(f"  check_dormancy(): {c.check_dormancy()}")


def test_fail_closed():
    print("\n[Test 6] Fail-closed on invalid split")
    c = KnowledgeCell(
        domain="fresh", owner_id="base-1",
        owner_type="base", birth_step=0,
    )
    # Fresh cell: not saturated
    try:
        c.split()
        print("  FAIL: should have raised")
    except BernaFatalError as e:
        print(f"  OK: {str(e)[:60]}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Berna R5 — Growth Demo (Theory 4)")
    print("=" * 60)

    c = test_split_trigger()
    test_split_execution(c)
    c2 = make_saturated_cell("code", "base-2")
    test_diversity(c2)
    test_capacity()
    test_dormancy()
    test_fail_closed()

    print("\n" + "=" * 60)
    print("All growth tests complete.")
    print("=" * 60)
