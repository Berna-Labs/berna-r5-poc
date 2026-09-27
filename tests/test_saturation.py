"""Tests for saturation module (Theory 1)."""

import pytest

from src.config import BernaFatalError, S_STAR, DIM_ORDER
from src.saturation import (
    score_length, score_width, score_height,
    score_depth, score_time, score_encompassment,
    build_K, compute_saturation, dominant_dimension,
    uniform_omega, update_omega,
    should_split, should_go_dormant, can_split_with_E,
)


class TestDimensionScorers:
    def test_length_basic(self):
        assert score_length([10, 20, 30], 100) == pytest.approx(0.30)

    def test_length_empty(self):
        assert score_length([], 100) == 0.0

    def test_length_zero_seq_raises(self):
        with pytest.raises(BernaFatalError):
            score_length([10], 0)

    def test_width_basic(self):
        counts = {"a": 0.5, "b": 0.2, "c": 0.05}
        assert score_width(counts, 10, tau=0.1) == pytest.approx(0.20)

    def test_height_focused(self):
        # Low entropy = high H
        H = score_height(activation_entropy=0.1, n_cells=10)
        assert 0.0 <= H <= 1.0

    def test_height_uniform(self):
        import math
        H = score_height(math.log(10), 10)
        assert H == pytest.approx(0.0, abs=1e-6)

    def test_depth_basic(self):
        assert score_depth(5, 10) == pytest.approx(0.5)

    def test_depth_zero_max_raises(self):
        with pytest.raises(BernaFatalError):
            score_depth(5, 0)

    def test_time_fresh(self):
        T = score_time(age_steps=0, freshness=1.0)
        assert T == pytest.approx(1.0)

    def test_time_old(self):
        T = score_time(age_steps=10000, decay_lambda=1e-4)
        assert 0.0 <= T < 1.0

    def test_time_negative_age_raises(self):
        with pytest.raises(BernaFatalError):
            score_time(age_steps=-1)

    def test_encompassment_full(self):
        facets = {"a": 1.0, "b": 1.0, "c": 1.0}
        assert score_encompassment(facets) == pytest.approx(1.0)

    def test_encompassment_partial(self):
        facets = {"a": 0.5, "b": 0.5}
        assert score_encompassment(facets) == pytest.approx(0.5)

    def test_encompassment_empty(self):
        assert score_encompassment({}) == 0.0


class TestBuildK:
    def test_valid(self):
        K = build_K(0.5, 0.3, 0.7, 0.4, 0.9, 0.6)
        assert set(K.keys()) == set(DIM_ORDER)

    def test_out_of_range_raises(self):
        with pytest.raises(BernaFatalError):
            build_K(1.5, 0.3, 0.7, 0.4, 0.9, 0.6)


class TestSaturation:
    def test_uniform_omega_sums_to_1(self):
        w = uniform_omega()
        assert sum(w.values()) == pytest.approx(1.0)

    def test_compute_basic(self, sample_K, uniform_omega):
        S = compute_saturation(sample_K, uniform_omega)
        assert 0.0 <= S <= 1.0

    def test_compute_perfect(self):
        K = build_K(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        S = compute_saturation(K, uniform_omega())
        assert S == pytest.approx(1.0)

    def test_compute_zero(self):
        K = build_K(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        S = compute_saturation(K, uniform_omega())
        assert S == pytest.approx(0.0)

    def test_dominant_dimension(self, sample_K, uniform_omega):
        d = dominant_dimension(sample_K, uniform_omega)
        # T=0.9 is highest, so dominant should be T
        assert d == "T"


class TestOmegaUpdate:
    def test_update_sums_to_1(self, uniform_omega):
        delta = {"L": 0.1, "W": -0.05, "H": 0.2,
                 "D": 0.15, "T": 0.0, "E": 0.3}
        w2 = update_omega(uniform_omega, delta)
        assert sum(w2.values()) == pytest.approx(1.0)

    def test_higher_delta_wins(self, uniform_omega):
        delta = {"L": 0.0, "W": 0.0, "H": 0.0,
                 "D": 0.0, "T": 0.0, "E": 0.9}
        w2 = update_omega(uniform_omega, delta)
        # E should have highest weight
        assert max(w2, key=w2.get) == "E"

    def test_invalid_delta_raises(self, uniform_omega):
        delta = {"L": 5.0, "W": 0.0, "H": 0.0,
                 "D": 0.0, "T": 0.0, "E": 0.0}
        with pytest.raises(BernaFatalError):
            update_omega(uniform_omega, delta)


class TestTriggers:
    def test_should_split_true(self):
        history = [S_STAR + 0.01] * 3
        assert should_split(history) is True

    def test_should_split_false_few(self):
        history = [S_STAR + 0.01]
        assert should_split(history) is False

    def test_should_split_false_low(self):
        history = [0.5, 0.6, 0.7]
        assert should_split(history) is False

    def test_should_go_dormant_true(self):
        history = [0.05] * 50
        assert should_go_dormant(history, utilization=0.01) is True

    def test_should_go_dormant_false_high_util(self):
        history = [0.05] * 50
        assert should_go_dormant(history, utilization=0.5) is False

    def test_can_split_with_E_ok(self, saturated_K):
        assert can_split_with_E(saturated_K) is True

    def test_can_split_with_E_low(self):
        K = build_K(0.9, 0.9, 0.9, 0.9, 0.9, 0.3)
        assert can_split_with_E(K) is False
