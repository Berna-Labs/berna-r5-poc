"""Theory 5: Neural Connectivity in the Plexus."""

import math
import uuid
from typing import Optional

from .config import (
    ETA_E, LAMBDA_E, W_MIN, W_MAX, BernaFatalError,
)
from .zero_error import assert_finite, assert_range


def new_node_id() -> str:
    return str(uuid.uuid4())


class Plexus:
    """Dynamic graph connecting cells, specialists, and base."""

    def __init__(self):
        self.nodes: dict = {}          # node_id -> metadata dict
        self.edges: dict = {}          # (src, tgt) -> weight
        self.coact_history: dict = {}  # (src, tgt) -> list of coact values

    # ----------------------------------------------------------
    # Nodes
    # ----------------------------------------------------------
    def add_node(self, node_id: str, metadata: Optional[dict] = None) -> None:
        if node_id in self.nodes:
            raise BernaFatalError(f"Node already exists: {node_id}")
        self.nodes[node_id] = metadata or {}

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise BernaFatalError(f"Unknown node: {node_id}")
        # Remove all edges touching this node
        to_del = [k for k in self.edges if node_id in k]
        for k in to_del:
            del self.edges[k]
            self.coact_history.pop(k, None)
        del self.nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    # ----------------------------------------------------------
    # Edges
    # ----------------------------------------------------------
    def add_edge(self, src: str, tgt: str, weight: float = 0.0) -> None:
        if src not in self.nodes or tgt not in self.nodes:
            raise BernaFatalError(f"Edge endpoints must exist: {src}, {tgt}")
        if src == tgt:
            raise BernaFatalError("Self-loops not allowed")
        assert_range(weight, 0.0, W_MAX, "edge weight")
        self.edges[(src, tgt)] = weight

    def remove_edge(self, src: str, tgt: str) -> None:
        if (src, tgt) not in self.edges:
            raise BernaFatalError(f"Edge missing: ({src}, {tgt})")
        del self.edges[(src, tgt)]
        self.coact_history.pop((src, tgt), None)

    def neighbors(self, node_id: str) -> list:
        return [tgt for (src, tgt) in self.edges if src == node_id]

    def neighbors_in(self, node_id: str) -> list:
        return [src for (src, tgt) in self.edges if tgt == node_id]

    def degree(self, node_id: str) -> int:
        return len(self.neighbors(node_id)) + len(self.neighbors_in(node_id))

    # ----------------------------------------------------------
    # Co-activation & edge update (Theory 5)
    # ----------------------------------------------------------
    def record_coactivation(self, src: str, tgt: str, value: float) -> None:
        """Store a co-activation observation for (src, tgt)."""
        if (src, tgt) not in self.edges:
            raise BernaFatalError(f"No edge to co-activate: ({src}, {tgt})")
        assert_range(value, 0.0, 1.0, "coactivation")
        key = (src, tgt)
        self.coact_history.setdefault(key, []).append(value)
        # Keep last 100
        if len(self.coact_history[key]) > 100:
            self.coact_history[key] = self.coact_history[key][-100:]

    def coactivation_mean(self, src: str, tgt: str) -> float:
        key = (src, tgt)
        hist = self.coact_history.get(key, [])
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    def update_edge_weight(self, src: str, tgt: str,
                           eta_e: float = ETA_E,
                           lambda_e: float = LAMBDA_E) -> float:
        """W(t+1) = W(t) + eta_e * (A(t) - lambda_e * W(t))."""
        key = (src, tgt)
        if key not in self.edges:
            raise BernaFatalError(f"Edge missing: {key}")
        W = self.edges[key]
        A = self.coactivation_mean(src, tgt)
        new_W = W + eta_e * (A - lambda_e * W)
        assert_finite(new_W, "new edge weight")
        new_W = max(W_MIN, min(W_MAX, new_W))
        self.edges[key] = new_W
        return new_W

    def update_all_edges(self) -> int:
        """Update all edges based on accumulated co-activations."""
        n = 0
        for (src, tgt) in list(self.edges.keys()):
            self.update_edge_weight(src, tgt)
            n += 1
        return n

    # ----------------------------------------------------------
    # Path / routing (Theory 5)
    # ----------------------------------------------------------
    def path_weight(self, path: list) -> float:
        """Product of edge weights along path."""
        if len(path) < 2:
            return 1.0
        w = 1.0
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            if e not in self.edges:
                return 0.0
            w *= self.edges[e]
        return w

    def top_k_neighbors(self, src: str, k: int = 5) -> list:
        """Return top-k outgoing neighbors by edge weight."""
        if src not in self.nodes:
            raise BernaFatalError(f"Unknown node: {src}")
        neigh = self.neighbors(src)
        scored = [(self.edges[(src, t)], t) for t in neigh]
        scored.sort(reverse=True)
        return [t for (_, t) in scored[:k]]

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------
    def stats(self) -> dict:
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        degrees = [self.degree(n) for n in self.nodes] if n_nodes else [0]
        avg_deg = sum(degrees) / len(degrees)
        isolated = sum(1 for n in self.nodes if self.degree(n) == 0)
        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "avg_degree": round(avg_deg, 3),
            "isolated": isolated,
            "max_weight": round(max(self.edges.values()), 4) if self.edges else 0.0,
        }

    def assert_no_isolated(self, allow_single: bool = True) -> None:
        """Zero-Error: every node must have degree >= 1.

        Set allow_single=True to permit a single-node plexus
        (transitional state during boot).
        """
        if allow_single and len(self.nodes) <= 1:
            return
        isolated = [n for n in self.nodes if self.degree(n) == 0]
        if isolated:
            raise BernaFatalError(f"Isolated nodes: {isolated}")
