"""Tests for Incorporation (Theory 2)."""

import pytest

from src.config import BernaFatalError, TAU_ROUTE, EPSILON_NEG
from src.incorporation import (
    hash_text, cosine_sim, route,
    verify_logical, verify_6d, verify_consistency,
    compute_reliability, Incorporation, GapQueue,
)


class TestHash:
    def test_hash_deterministic(self):
        assert hash_text("hello") == hash_text("hello")

    def test_hash_different(self):
        assert hash_text("a") != hash_text("b")

    def test_hash_length(self):
        assert len(hash_text("x")) == 64  # SHA256 hex


class TestCosineSim:
    def test_identical(self):
        assert cosine_sim([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine_sim([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(BernaFatalError):
            cosine_sim([1, 0], [1, 0, 0])

    def test_zero_vector_raises(self):
        with pytest.raises(BernaFatalError):
            cosine_sim([0, 0], [1, 0])


class TestRouting:
    def test_route_correct(self, sample_specialists):
        r = route([1.0, 0.0, 0.0, 0.0], sample_specialists)
        assert r is not None
        assert r["specialist"]["specialist_id"] == "spec_math"

    def test_route_below_threshold(self, sample_specialists):
        r = route([0.25, 0.25, 0.25, 0.25], sample_specialists,
                  tau_route=0.99)
        assert r is None

    def test_route_empty_list(self):
        assert route([1, 0], []) is None

    def test_route_missing_embedding_raises(self):
        bad = [{"specialist_id": "x"}]
        with pytest.raises(BernaFatalError):
            route([1, 0], bad)


class TestVerification:
    def test_6d_good(self):
        K1 = {"L": 0.5, "W": 0.3, "H": 0.7, "D": 0.4, "T": 0.9, "E": 0.6}
        K2 = {"L": 0.5, "W": 0.3, "H": 0.7, "D": 0.4, "T": 0.9, "E": 0.65}
        assert verify_6d(K1, K2) is True

    def test_6d_drop_fails(self):
        K1 = {"L": 0.5, "W": 0.3, "H": 0.7, "D": 0.4, "T": 0.9, "E": 0.6}
        K2 = {"L": 0.5, "W": 0.3, "H": 0.7, "D": 0.4, "T": 0.9, "E": 0.3}
        assert verify_6d(K1, K2) is False

    def test_logical_ok(self):
        def scorer(t):
            return 0.9 if "yes" in t else 0.1
        assert verify_logical("yes", "no", scorer) is True

    def test_logical_contradiction(self):
        def scorer(t):
            return 0.9
        assert verify_logical("yes", "yes", scorer) is False

    def test_consistency_identical(self):
        assert verify_consistency(["a", "a", "a"]) is True

    def test_consistency_mixed(self):
        assert verify_consistency(["a", "b", "a"]) is False

    def test_consistency_single(self):
        assert verify_consistency(["a"]) is True


class TestReliability:
    def test_reliability_ok(self):
        K1 = {"L": 0.5, "W": 0.3, "H": 0.7, "D": 0.4, "T": 0.9, "E": 0.6}
        K2 = {"L": 0.5, "W": 0.3, "H": 0.7, "D": 0.4, "T": 0.9, "E": 0.65}
        r = compute_reliability(0.9, True, K1, K2)
        assert r == pytest.approx(0.9)

    def test_reliability_inconsistent(self):
        K1 = {"L": 0.5, "W": 0.3, "H": 0.7, "D": 0.4, "T": 0.9, "E": 0.6}
        r = compute_reliability(0.9, False, K1, K1)
        assert r == 0.0


class TestIncorporation:
    def test_full_incorporation_integrated(self, sample_specialists):
        K_before = {"L": 0.5, "W": 0.3, "H": 0.7,
                    "D": 0.4, "T": 0.9, "E": 0.6}

        def k_after(K, q, a):
            K2 = dict(K); K2["E"] += 0.05; return K2

        def scorer(t):
            return 0.9 if not t.startswith("not ") else 0.2

        def ans_fn(spec, q):
            return "answer_42"

        inc = Incorporation("r1", "test query")
        status = inc.process(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            specialists=sample_specialists,
            specialist_answer_fn=ans_fn,
            base_scorer=scorer,
            K_before=K_before,
            K_after_fn=k_after,
        )
        assert status == "integrated"
        assert inc.verified is True

    def test_incorporation_queued_no_route(self, sample_specialists):
        K_before = {"L": 0.5, "W": 0.3, "H": 0.7,
                    "D": 0.4, "T": 0.9, "E": 0.6}
        inc = Incorporation("r2", "unknown query")
        status = inc.process(
            query_embedding=[0.25, 0.25, 0.25, 0.25],
            specialists=sample_specialists,
            specialist_answer_fn=lambda s, q: None,
            base_scorer=lambda t: 0.5,
            K_before=K_before,
            K_after_fn=lambda K, q, a: K,
        )
        assert status == "queued"


class TestGapQueue:
    def test_add_gap(self):
        gq = GapQueue()
        gq.add("r1", "query", [1.0, 0.0])
        assert gq.stats()["size"] == 1

    def test_cluster_priority(self):
        gq = GapQueue(tau_prio=3.0)
        for i in range(5):
            gq.add(f"r{i}", f"q{i}", [1.0, 0.0, 0.0, 0.0])
        stats = gq.stats()
        assert stats["max_priority"] >= 5

    def test_trigger_ready(self):
        gq = GapQueue(tau_prio=3.0)
        for i in range(5):
            gq.add(f"r{i}", f"q{i}", [1.0, 0.0, 0.0, 0.0])
        assert gq.should_trigger_training() is True

    def test_top_priority_empty(self):
        gq = GapQueue()
        assert gq.top_priority() is None
