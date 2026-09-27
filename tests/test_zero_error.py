"""Tests for zero_error module."""

import math

import pytest

from src.config import BernaFatalError
from src.zero_error import (
    assert_finite,
    assert_range,
    assert_simplex,
    assert_dim_vector,
    fail,
)


class TestAssertFinite:
    def test_int_ok(self):
        assert_finite(5, "x")

    def test_float_ok(self):
        assert_finite(3.14, "pi")

    def test_negative_ok(self):
        assert_finite(-1.5, "neg")

    def test_nan_raises(self):
        with pytest.raises(BernaFatalError):
            assert_finite(float("nan"), "bad")

    def test_inf_raises(self):
        with pytest.raises(BernaFatalError):
            assert_finite(float("inf"), "bad")

    def test_neg_inf_raises(self):
        with pytest.raises(BernaFatalError):
            assert_finite(float("-inf"), "bad")


class TestAssertRange:
    def test_mid_ok(self):
        assert_range(0.5, 0.0, 1.0, "x")

    def test_edges_ok(self):
        assert_range(0.0, 0.0, 1.0, "lo")
        assert_range(1.0, 0.0, 1.0, "hi")

    def test_below_raises(self):
        with pytest.raises(BernaFatalError):
            assert_range(-0.1, 0.0, 1.0, "x")

    def test_above_raises(self):
        with pytest.raises(BernaFatalError):
            assert_range(1.1, 0.0, 1.0, "x")

    def test_nan_raises(self):
        with pytest.raises(BernaFatalError):
            assert_range(float("nan"), 0.0, 1.0, "x")


class TestAssertSimplex:
    def test_uniform_ok(self):
        assert_simplex([1/6] * 6)

    def test_two_way_ok(self):
        assert_simplex([0.5, 0.5])

    def test_negative_raises(self):
        with pytest.raises(BernaFatalError):
            assert_simplex([0.5, -0.5, 1.0])

    def test_not_summing_raises(self):
        with pytest.raises(BernaFatalError):
            assert_simplex([0.5, 0.3])

    def test_empty_raises(self):
        with pytest.raises(BernaFatalError):
            assert_simplex([])


class TestAssertDimVector:
    def test_valid_6d(self):
        K = {"L": 0.5, "W": 0.3, "H": 0.7,
             "D": 0.4, "T": 0.9, "E": 0.6}
        assert_dim_vector(K, ["L", "W", "H", "D", "T", "E"])

    def test_missing_dim_raises(self):
        K = {"L": 0.5, "W": 0.3, "H": 0.7}
        with pytest.raises(BernaFatalError):
            assert_dim_vector(K, ["L", "W", "H", "D", "T", "E"])

    def test_out_of_range_raises(self):
        K = {"L": 1.5, "W": 0.3, "H": 0.7,
             "D": 0.4, "T": 0.9, "E": 0.6}
        with pytest.raises(BernaFatalError):
            assert_dim_vector(K, ["L", "W", "H", "D", "T", "E"])


class TestFail:
    def test_fail_raises(self):
        with pytest.raises(BernaFatalError, match="custom message"):
            fail("custom message")
