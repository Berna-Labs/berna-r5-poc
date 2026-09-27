# Theory 5: Neural Connectivity in the Plexus

## 1. Motivation

Cells do not exist in isolation. Berna R5 requires a plexus:
a dynamic graph connecting cells, specialists, and the base.
Connectivity enables knowledge flow, routing, and emergent
specialization through co-activation.

## 2. Definitions

Definition 2.1 (Plexus).
    P = (V, E, W)
- V = nodes (cells, specialists, base)
- E = edges (directed or undirected)
- W: E -> R_+ = edge weights

Definition 2.2 (Neighbor Set).
    N(v) = { u in V : (v, u) in E }

Definition 2.3 (Co-Activation).
    A_uv(t) = correlation of activation signals over window T

Definition 2.4 (Edge Weight Update).
    W_uv(t+1) = W_uv(t) + eta_e * ( A_uv(t) - lambda * W_uv(t) )
- eta_e = learning rate
- lambda = decay coefficient

Definition 2.5 (Path).
    path(u, v) = (u = v0, v1, ..., vk = v)
    W(path) = product of edge weights along path

Definition 2.6 (Shortest Weighted Path).
    d(u, v) = -log( max W(path) ) over all paths

## 3. Plexus Properties

Property 3.1 (Small-World).
Short average path length: O(log |V|).
High clustering coefficient: C >> C_random.

Property 3.2 (Scale-Free).
Degree distribution: P(k) ~ k^(-gamma), gamma in [2, 3].

Property 3.3 (Dynamic Reconfiguration).
Edges can be added, removed, or reweighted at every K steps.

## 4. Main Theorem

Theorem 4.1 (Connectivity Convergence).
Under edge update rule with:
    (H1) A_uv stationary and bounded in [0, 1]
    (H2) eta_e < 2 / lambda_max(A)
    (H3) lambda > 0
Then W converges to fixed point W* = A / lambda.

Proof sketch:
Update is linear system:
    W(t+1) = (I - eta_e * lambda) W(t) + eta_e * A
Unique fixed point when eigenvalues of (I - eta_e*lambda)
are within unit disk. By H2, holds. Rate O((1-eta_e*lambda)^t).

Corollary 4.2 (Stable Routing). Given convergent W, shortest
path routing is stable.

Corollary 4.3 (Emergent Hubs). High co-activation nodes
become hubs; topology is scale-free.

## 5. Knowledge Flow Dynamics

Request r at base B:
    step 1: B computes query embedding e_q
    step 2: B computes similarity to each neighbor v
    step 3: B selects top-K neighbors by sim(e_q, e_v)
    step 4: B routes r to neighbors, collects answers
    step 5: B aggregates answers by edge weights
    step 6: B incorporates result (Theory 2)
    step 7: B updates W (edge learning)

Zero-Error: any missing neighbor => skip silently, log warning.

## 6. Integration with Other Theories

With Theory 1 (Saturation):
    Saturated cells are hubs in the plexus.
    Their edges are strong due to high co-activation.

With Theory 2 (Incorporation):
    Incorporation uses plexus paths to reach specialists.
    Edge weights determine routing preference.

With Theory 3 (Zero-Forgetting):
    Frozen cells maintain their edges.
    Edges are never deleted, only reweighted.

With Theory 4 (Balanced Growth):
    New cells attach via k-NN edges to most similar existing
    cells. Growth preserves small-world and scale-free.

## 7. Zero-Error Guarantees

Property 7.1 (No Isolated Cells).
Every active cell has at least one edge (degree >= 1).
Assert after every growth event.

Property 7.2 (Bounded Edge Weight).
W_uv in [W_min, W_max] = [1e-6, 1.0] always.
Clamp after every update.

Property 7.3 (Acyclic Assumption).
Primary graph is a DAG. Cycles allowed only on back-edges
for feedback.

Property 7.4 (Audit Trail).
Every edge addition or removal logged to registry.plexus_log.

## 8. Implementation Notes

Graph storage:
- Adjacency list in memory (dict of dicts)
- Persisted to disk every K steps
- On restart: reload from disk

Edge update frequency:
- Every 100 steps (matches checkpointing)
- Full re-computation every 1000 steps

Routing:
- Pre-compute top-K neighbors for common query clusters
- Cache routing decisions for hot queries

Zero-Error code (Python):
    def update_edge(W, A, eta_e, lambda_):
        new_W = W + eta_e * (A - lambda_ * W)
        if not np.isfinite(new_W):
            raise BernaFatalError("Edge weight non-finite")
        return np.clip(new_W, W_MIN, W_MAX)

## 9. Relation to Prior Work

Transformer: fixed, dense topology, gradient learning.
MoE: fixed, sparse, gradient + router.
GNN: fixed graph, message passing.
RAG: external store, retrieval.
This work: dynamic plexus, co-activation learning.

Key difference: plexus is not fixed. It evolves with
co-activation patterns, similar to biological synapses.

## 10. Open Questions

1. Is gamma (scale-free exponent) universal across domains?
2. What is the optimal eta_e?
3. Can edges be created between distant nodes (long-range)?
4. How does plexus size affect VRAM and latency?
5. Does emergent topology correlate with knowledge quality?

## 11. Summary

The plexus is a dynamic graph connecting cells, specialists,
and base. Edges strengthen through co-activation, following
a provably convergent update rule. The plexus exhibits
small-world and scale-free properties, enabling efficient
knowledge flow and routing. It integrates with all other
theories: saturation (hubs), incorporation (routing),
zero-forgetting (frozen edges), and balanced growth
(attachment rules). Zero-error guarantees ensure no isolated
cells, bounded weights, and auditable evolution.
