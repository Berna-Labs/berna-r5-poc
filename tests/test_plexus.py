"""Tests for Plexus (Theory 5)."""

import pytest

from src.plexus import Plexus, new_node_id
from src.config import BernaFatalError, W_MIN, W_MAX


class TestNodeManagement:
    def test_add_node(self):
        p = Plexus()
        p.add_node("a")
        assert "a" in p.nodes
        assert p.has_node("a")

    def test_duplicate_node_raises(self):
        p = Plexus()
        p.add_node("a")
        with pytest.raises(BernaFatalError):
            p.add_node("a")

    def test_remove_node(self):
        p = Plexus()
        p.add_node("a")
        p.remove_node("a")
        assert not p.has_node("a")

    def test_remove_unknown_raises(self):
        p = Plexus()
        with pytest.raises(BernaFatalError):
            p.remove_node("missing")

    def test_new_node_id_unique(self):
        a, b = new_node_id(), new_node_id()
        assert a != b


class TestEdgeManagement:
    def test_add_edge(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        p.add_edge("a", "b", 0.5)
        assert ("a", "b") in p.edges
        assert p.edges[("a", "b")] == 0.5

    def test_self_loop_raises(self):
        p = Plexus()
        p.add_node("a")
        with pytest.raises(BernaFatalError):
            p.add_edge("a", "a", 0.5)

    def test_missing_endpoint_raises(self):
        p = Plexus()
        p.add_node("a")
        with pytest.raises(BernaFatalError):
            p.add_edge("a", "b", 0.5)

    def test_negative_weight_raises(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        with pytest.raises(BernaFatalError):
            p.add_edge("a", "b", -0.1)

    def test_remove_edge(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        p.add_edge("a", "b", 0.5)
        p.remove_edge("a", "b")
        assert ("a", "b") not in p.edges

    def test_remove_unknown_edge_raises(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        with pytest.raises(BernaFatalError):
            p.remove_edge("a", "b")

    def test_remove_node_clears_edges(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b"); p.add_node("c")
        p.add_edge("a", "b", 0.5)
        p.add_edge("b", "c", 0.5)
        p.remove_node("b")
        assert ("a", "b") not in p.edges
        assert ("b", "c") not in p.edges


class TestNeighbors:
    def test_neighbors_out(self):
        p = Plexus()
        for n in "abc":
            p.add_node(n)
        p.add_edge("a", "b", 0.5)
        p.add_edge("a", "c", 0.5)
        assert set(p.neighbors("a")) == {"b", "c"}

    def test_neighbors_in(self):
        p = Plexus()
        for n in "abc":
            p.add_node(n)
        p.add_edge("b", "a", 0.5)
        p.add_edge("c", "a", 0.5)
        assert set(p.neighbors_in("a")) == {"b", "c"}

    def test_degree(self):
        p = Plexus()
        for n in "abc":
            p.add_node(n)
        p.add_edge("a", "b", 0.5)
        p.add_edge("c", "a", 0.5)
        assert p.degree("a") == 2


class TestCoactivation:
    def test_record_coactivation(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        p.add_edge("a", "b", 0.5)
        p.record_coactivation("a", "b", 0.8)
        assert p.coactivation_mean("a", "b") == pytest.approx(0.8)

    def test_record_on_missing_edge_raises(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        with pytest.raises(BernaFatalError):
            p.record_coactivation("a", "b", 0.8)

    def test_out_of_range_raises(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        p.add_edge("a", "b", 0.5)
        with pytest.raises(BernaFatalError):
            p.record_coactivation("a", "b", 1.5)

    def test_mean_empty(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        p.add_edge("a", "b", 0.5)
        assert p.coactivation_mean("a", "b") == 0.0


class TestEdgeUpdate:
    def test_update_edge_weight(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        p.add_edge("a", "b", 0.1)
        # High coactivation should increase weight
        for _ in range(20):
            p.record_coactivation("a", "b", 1.0)
        new_w = p.update_edge_weight("a", "b")
        assert new_w > 0.1

    def test_update_missing_edge_raises(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        with pytest.raises(BernaFatalError):
            p.update_edge_weight("a", "b")

    def test_update_all(self):
        p = Plexus()
        for n in "abc":
            p.add_node(n)
        p.add_edge("a", "b", 0.1)
        p.add_edge("b", "c", 0.1)
        n = p.update_all_edges()
        assert n == 2

    def test_weight_clamped(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        p.add_edge("a", "b", W_MAX)
        for _ in range(100):
            p.record_coactivation("a", "b", 1.0)
        p.update_edge_weight("a", "b")
        assert p.edges[("a", "b")] <= W_MAX


class TestPath:
    def test_path_weight(self):
        p = Plexus()
        for n in "abc":
            p.add_node(n)
        p.add_edge("a", "b", 0.5)
        p.add_edge("b", "c", 0.4)
        w = p.path_weight(["a", "b", "c"])
        assert w == pytest.approx(0.5 * 0.4)

    def test_path_missing_edge(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        assert p.path_weight(["a", "b"]) == 0.0

    def test_path_single_node(self):
        p = Plexus()
        p.add_node("a")
        assert p.path_weight(["a"]) == 1.0


class TestTopK:
    def test_top_k_neighbors(self):
        p = Plexus()
        for n in "abcd":
            p.add_node(n)
        p.add_edge("a", "b", 0.9)
        p.add_edge("a", "c", 0.5)
        p.add_edge("a", "d", 0.1)
        top = p.top_k_neighbors("a", k=2)
        assert top == ["b", "c"]

    def test_top_k_unknown_node_raises(self):
        p = Plexus()
        with pytest.raises(BernaFatalError):
            p.top_k_neighbors("missing")


class TestZeroError:
    def test_isolated_check_single_allowed(self):
        p = Plexus()
        p.add_node("a")
        p.assert_no_isolated()  # should not raise

    def test_isolated_check_pair_fails(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        with pytest.raises(BernaFatalError):
            p.assert_no_isolated()

    def test_isolated_check_connected_ok(self):
        p = Plexus()
        p.add_node("a"); p.add_node("b")
        p.add_edge("a", "b", 0.5)
        p.assert_no_isolated()


class TestStats:
    def test_stats_empty(self):
        p = Plexus()
        s = p.stats()
        assert s["nodes"] == 0
        assert s["edges"] == 0
        assert s["isolated"] == 0

    def test_stats_connected(self):
        p = Plexus()
        for n in "abc":
            p.add_node(n)
        p.add_edge("a", "b", 0.5)
        p.add_edge("b", "c", 0.3)
        s = p.stats()
        assert s["nodes"] == 3
        assert s["edges"] == 2
        assert s["isolated"] == 0
