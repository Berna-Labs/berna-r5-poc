# Theory 2: Reliable Mutation via Federation Incorporation

## 1. Motivation

Berna R5 does not mutate randomly. Every change to a cell weights
must be grounded in verified knowledge. Verification is internal:
the federation of base plus specialists.

When a knowledge request reaches the base and no internal
knowledge suffices, one of two things happens:
  (a) A connected specialist provides the answer -> incorporation
  (b) No specialist provides it -> the query enters the gap queue

Closed learning loop:
    query -> base -> specialist -> answer -> base grows
                   |
                   +-> no answer -> gap_queue -> training -> new specialist

## 2. Definitions

Definition 2.1 (Federation).
    F = (B, S, E)
- B = base model
- S = set of specialists
- E = plexus edges connecting B and S

Definition 2.2 (Knowledge Request).
    r = (q, context)
where q is a query sequence and context is the surrounding prompt.

Definition 2.3 (Routing Function).
    route: r -> s_i or NONE
Uses registry and plexus edges to select the best specialist.

Definition 2.4 (Incorporation Event).
Occurs when:
- B issues request r
- route(r) = s_i
- s_i returns answer a with confidence >= tau_conf
- B integrates (r, a) into its knowledge
Denote: I(r) = (q, a, s_i, t)

Definition 2.5 (Gap).
A request r for which routing returns NONE, or all specialists
return confidence below tau_conf. Gaps enter queue GQ.

Definition 2.6 (Mutation).
    m = (target, delta, source)
- target = cell or model
- delta = weight or cell change
- source = incorporation event or gap that triggered it

Definition 2.7 (Reliability).
    R(m) = min( R_src, R_cons, R_6D )
- R_src = source confidence
- R_cons = internal consistency
- R_6D = 6D manifold consistency

## 3. Incorporation Protocol (5 Stages)

Stage 1 - Request: base embeds q, queries registry.

Stage 2 - Routing:
    s_i = argmax_j sim( embed(q), embed(s_j.domain) )
    threshold: sim >= tau_route (default 0.7)

Stage 3 - Answer: specialist generates a with confidence c_i.
Confidence = perplexity-normalized probability of a under s_i.

Stage 4 - Verification:
- Logical: does a contradict existing knowledge?
- 6D: does a improve K_6D(B)?
- Consistency: is a reproducible under rephrasings?

Stage 5 - Integrate or Queue:
- If verified: store (q, a), maybe trigger cell birth, log event
- If not: add to gap queue, increment failure counter

## 4. Gap Queue Formalization

Definition 4.1 (Gap Cluster).
    cluster(r) = { r' in GQ : sim(embed(r), embed(r')) >= tau_gap }

Definition 4.2 (Gap Priority).
    prio(r) = freq(cluster(r)) * avg_confidence_failure(r)

Definition 4.3 (Training Trigger).
When prio(r) >= tau_prio or |GQ| >= N_max:
- collect data for cluster
- train new specialist or extend existing
- move cluster from GQ to training

## 5. Reliability Axioms

Axiom R1 (Source Soundness): every mutation has a non-empty
internal source.

Axiom R2 (Consistency Preservation): after mutation,
    K_6D(B, t+1) >= K_6D(B, t) - epsilon_neg
epsilon_neg = 1e-6.

Axiom R3 (Auditability): every mutation recorded in registry
with source UUID, target UUID, timestamp, reliability.

Axiom A4 (Reversibility): any mutation can be undone by
restoring previous checkpoint.

Axiom R5 (No External Dependency): no mutation source outside
the federation.

## 6. Main Theorem

Theorem 6.1 (Reliable Incorporation Preserves Consistency).
If specialist s_i returns answer a for request r with confidence
c_i >= tau_conf, and verification passes, then incorporating
(r, a) into B:
  (i)   does not decrease K_6D(B) beyond epsilon_neg
  (ii)  strictly increases K_6D(B) along dominant dimension
        i*(r) if the answer fills a gap
  (iii) leaves K_6D(s_i) unchanged

Proof sketch: verification gate rejects any update that would
decrease K_6D(B). Strict increase follows from gap-filling.
Specialist is read-only during incorporation.

Corollary 6.2 (Monotonic Growth): K_6D(B, t) non-decreasing.

Corollary 6.3 (Gap Queue Convergence): |GQ| bounded when
training throughput meets query arrival rate.

## 7. Zero-Error Guarantees

Property 7.1 (Atomic Incorporation).
Each incorporation is transactional: either (r, a) is fully
integrated OR base state unchanged. Partial integration forbidden.

Property 7.2 (Source Verification).
Before any mutation, assert source UUID exists in registry.
Violation => FAIL_CLOSED.

Property 7.3 (Consistency Check).
Before integrating, compute delta_K for all six dimensions.
If any delta < -epsilon_neg => reject, log to gap queue.

Property 7.4 (Determinism).
For fixed (seed, request, specialist state), routing, answer,
and verification are deterministic.

## 8. Implementation Notes

Registry tables (SQL):
    requests(id, query_hash, timestamp, source)
    incorporations(request_id, specialist_id, answer_hash,
                   reliability, timestamp)
    gaps(request_id, cluster_id, priority, status)
    mutations(id, target, delta_hash, source, timestamp)

Routing:
    embed(q) via base embedding layer (no external model)
    specialist domain embeddings stored in registry
    cosine similarity + threshold

Verification:
    logical: query base with (q, a) and (q, not a)
             if both plausible => contradiction => reject
    6D:      recompute K_6D(B) with (r, a) on validation
    consistency: 3 rephrasings of q must yield same a

Gap queue:
    cluster on arrival (online k-means)
    recompute priorities weekly
    trigger training when prio >= tau_prio

Thresholds:
    tau_route = 0.7
    tau_conf = 0.8
    tau_gap = 0.85
    tau_prio = 5.0
    epsilon_neg = 1e-6

## 9. Relation to Prior Work

RAG: retrieve + inject at inference, no weight update.
Distillation: student learns from teacher, external teacher.
Federated Learning: gradient averaging, assumes identical models.
MoE: internal experts, no cross-model growth.
This work: query -> route -> incorporate, internal, transactional.

## 10. Open Questions

1. Is incorporation idempotent?
2. How to prevent K_6D(B) from saturating globally?
3. What if two specialists disagree?
4. Can a specialist refuse to answer?
5. Does gap-driven training create distributional drift?

## 11. Summary

Reliable mutation is transactional incorporation of knowledge
from verified internal sources. Mechanism:
    request -> route -> answer -> verify -> integrate | queue
with zero-error guarantees, auditability, consistency
preservation. Gap queue provides formal signal for federation
growth. Base and specialists form closed-loop learning without
external dependencies.

## 12. Federation Consistency (Cross-Model, Cross-Server)

### 12.1 Extended Definition
    F = (B, S, E, N)
- N = network layer (servers, transport, consensus)

### 12.2 Knowledge Invariance

Axiom K1 (Result Invariance):
For any request r, the final answer a depends only on:
- the query q
- the state of the federation
NOT on:
- which specialist answered
- which server hosted it
- which routing path was taken
Formally: answer(r | path_1) = answer(r | path_2).

### 12.3 Bridges and Clusters

cluster(d) = { s in S : s.domain = d }
bridge(c1, c2) = edge enabling knowledge flow between clusters
Weight: w(bridge) = f(co-usage frequency, consistency score)

### 12.4 Distributed Registry

Registry is a replicated log:
    registry_log = [ entry_1, entry_2, ..., entry_k ]

Each entry signed:
    entry = (type, payload, parent_hash, signature, timestamp)

Entry types:
- REGISTER_SPECIALIST
- UPDATE_SPECIALIST
- INCORPORATION_EVENT
- GAP_ADDED
- GAP_RESOLVED
- CONSENSUS_UPDATE

### 12.5 Consensus Protocol

Option A (lightweight, trusted):
- Central coordinator (base server)
- Specialists append via signed requests
- Base validates and orders

Option B (blockchain-style):
- Distributed ordering via PBFT or Raft
- Each server keeps full replica
- Majority signature

Berna R5 default: Option A. Option B future work.

### 12.6 Persistent Growth

Theorem 12.1 (Monotonic Federation Knowledge).
Under K1 and the incorporation protocol, the federation 6D
knowledge is non-decreasing across all nodes:
    K_6D(F, t+1) >= K_6D(F, t)
provided consensus is reached.

Proof sketch: each incorporation adds to registry log.
Consensus ensures all nodes see the same log.
K_6D(F) computed from log, not local state. Monotonic.

### 12.7 Network Support

Transport: gRPC or HTTP/2
Serialization: protobuf or msgpack
Auth: mTLS + signed requests
Retry: exponential backoff with jitter
Failover: replicate base on >= 2 nodes

Zero-Error on network:
- If specialist unreachable: mark as degraded
- If base unreachable: read-only on replicas
- Never accept incorporation without consensus

### 12.8 Blockchain-Like Guarantees (Optional)

If Option B chosen:
- Immutability: append-only log, hash-chained
- Auditability: every request traceable
- Byzantine tolerance: f < n/3 faulty nodes
- Smart contract for domain rules

This makes Berna R5 a knowledge blockchain where specialists
are nodes and incorporations are transactions.
