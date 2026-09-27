"""Tests for KnowledgeCell (Theory 1 + 4)."""

import pytest

from src.cell import KnowledgeCell
from src.config import BernaFatalError


class TestCellCreation:
    def test_basic_creation(self):
        c = KnowledgeCell(
            domain="math", owner_id="base-1",
            owner_type="base", birth_step=0,
        )
        assert c.domain == "math"
        assert c.state == "active"
        assert c.parent_cell is None
        assert c.cell_id  # UUID string

    def test_invalid_owner_type_raises(self):
        with pytest.raises(BernaFatalError):
            KnowledgeCell(
                domain="math", owner_id="x",
                owner_type="invalid", birth_step=0,
            )

    def test_unique_ids(self):
        c1 = KnowledgeCell("a", "b", "base", 0)
        c2 = KnowledgeCell("a", "b", "base", 0)
        assert c1.cell_id != c2.cell_id


class TestCellUpdate:
    def test_update_K(self, sample_K):
        c = KnowledgeCell("math", "base-1", "base", 0)
        c.update(sample_K, utilization=0.5)
        assert c.S > 0.0
        assert c.utilization == 0.5

    def test_invalid_K_raises(self):
        c = KnowledgeCell("math", "base-1", "base", 0)
        bad_K = {"L": 1.5, "W": 0.3, "H": 0.7,
                 "D": 0.4, "T": 0.9, "E": 0.6}
        with pytest.raises(BernaFatalError):
            c.update(bad_K, utilization=0.5)

    def test_invalid_utilization_raises(self, sample_K):
        c = KnowledgeCell("math", "base-1", "base", 0)
        with pytest.raises(BernaFatalError):
            c.update(sample_K, utilization=1.5)

    def test_tick_increments_age(self):
        c = KnowledgeCell("math", "base-1", "base", 0)
        c.tick()
        c.tick()
        assert c.age == 2

    def test_update_resets_age(self, sample_K):
        c = KnowledgeCell("math", "base-1", "base", 0)
        c.tick(); c.tick()
        c.update(sample_K, utilization=0.5)
        assert c.age == 0


class TestCellSaturation:
    def test_saturated_cell_can_split(self, saturated_K):
        c = KnowledgeCell("math", "base-1", "base", 0)
        for _ in range(5):
            c.update(saturated_K, utilization=0.85)
            c.tick()
        assert c.check_split() is True

    def test_fresh_cell_cannot_split(self):
        c = KnowledgeCell("math", "base-1", "base", 0)
        assert c.check_split() is False

    def test_dormancy_trigger(self):
        c = KnowledgeCell("stale", "base-1", "base", 0)
        # All dims low including T: S = sqrt(6 * 0.05^2 / 6) = 0.05 < 0.15
        low_K = {"L": 0.05, "W": 0.05, "H": 0.05,
                 "D": 0.05, "T": 0.05, "E": 0.05}
        for _ in range(60):
            c.update(low_K, utilization=0.01)
            c.tick()
        assert c.check_dormancy() is True


class TestCellSplit:
    def test_split_produces_two_daughters(self, saturated_K):
        c = KnowledgeCell("math", "base-1", "base", 0)
        for _ in range(5):
            c.update(saturated_K, utilization=0.85)
            c.tick()
        d1, d2 = c.split()
        assert d1.cell_id != d2.cell_id
        assert c.state == "dormant"
        assert d1.parent_cell == c.cell_id
        assert d2.parent_cell == c.cell_id

    def test_split_fails_when_unsaturated(self):
        c = KnowledgeCell("math", "base-1", "base", 0)
        with pytest.raises(BernaFatalError):
            c.split()

    def test_daughters_have_different_domains(self, saturated_K):
        c = KnowledgeCell("math", "base-1", "base", 0)
        for _ in range(5):
            c.update(saturated_K, utilization=0.85)
            c.tick()
        d1, d2 = c.split()
        assert d1.domain != d2.domain


class TestStateTransitions:
    def test_freeze(self):
        c = KnowledgeCell("a", "b", "base", 0)
        c.freeze()
        assert c.state == "frozen"

    def test_die(self):
        c = KnowledgeCell("a", "b", "base", 0)
        c.die()
        assert c.state == "dead"

    def test_die_twice_raises(self):
        c = KnowledgeCell("a", "b", "base", 0)
        c.die()
        with pytest.raises(BernaFatalError):
            c.die()

    def test_freeze_dead_raises(self):
        c = KnowledgeCell("a", "b", "base", 0)
        c.die()
        with pytest.raises(BernaFatalError):
            c.freeze()


class TestCellInfo:
    def test_info_keys(self, sample_K):
        c = KnowledgeCell("math", "base-1", "base", 0)
        c.update(sample_K, utilization=0.5)
        info = c.info()
        expected = {"cell_id", "domain", "state", "S", "K",
                    "omega", "utilization", "age", "parent_cell"}
        assert set(info.keys()) == expected
