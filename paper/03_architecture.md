# 3. Architecture

Berna R5 is organized in six layers, from sensors to registry.
This section describes each layer and its role.

## 3.1 Overview

    Input (text, vision, audio)
         |
         v
    [Sensors]           raw signal encoding
         |
         v
    [Spinal Cord]       fast rule-based routing
         |
         v
    [Plexus]            dynamic graph of cells
         |
         v
    [Knowledge Cells]   born, split, die, interconnect
         |
         v
    [DNA Kernel]        100 chromosomes, regulatory
         |
         v
    [Registry]          SQL, UUIDs, lineage, routing

Each layer has a clear interface and can be tested in
isolation (see src/ for implementation).

## 3.2 Sensors

The sensor layer converts raw input into token sequences.

- Text: byte-pair encoding (BPE) with 200k vocabulary.
  The vocabulary is partitioned:
  - text tokens: 0 to 63,999
  - vision tokens: 64,000 to 163,999
  - audio tokens: 164,000 to 199,999
- Vision: CLIP ViT-L/14, frozen. Produces 256 image tokens.
- Audio: Whisper Large v3, frozen. Produces up to 1500
  audio tokens at 16 kHz.

The sensor layer does not learn. It is a deterministic
transformation from raw input to token IDs. Zero-Error
Invariant applies: any out-of-range token ID triggers
FAIL_CLOSED.

## 3.3 Spinal Cord

The spinal cord is a rule-based fast router. It does not
use learned weights. Its purpose is to select the relevant
cell or specialist before the slower learned components
activate.

Rules are of the form:
    if input_contains(token_group_A):
        route_to(cell_group_A)

Rules are stored as JSON and loaded at boot. They are
deterministic and auditable. New rules can be added without
retraining.

Why a spinal cord? Learned routers (as in MoE) require
training and can drift. A rule-based router is faster,
cheaper, and provably correct.

## 3.4 Plexus

The plexus is a dynamic graph P = (V, E, W) where:
- V = nodes (cells, specialists, base)
- E = edges (directed or undirected)
- W: E -> R_+ = edge weights

Edges strengthen through co-activation (Definition 2.4,
Theory 5):

    W_uv(t+1) = W_uv(t) + eta_e * (A_uv(t) - lambda * W_uv(t))

where A_uv(t) is the correlation of activations of u and v
over a window. The update is a linear time-invariant system
with provable convergence (Theorem 5.1).

Properties:
- Small-world: short path length O(log |V|)
- Scale-free: degree distribution P(k) ~ k^(-gamma), gamma
  in [2, 3]
- Dynamic: edges can be added, removed, reweighted

## 3.5 Knowledge Cells

A knowledge cell is the fundamental unit of knowledge. Each
cell has:

- cell_id: UUID
- domain: e.g., "math", "code", "arabic"
- state: active, dormant, frozen, dead
- K_6D: 6-dimensional knowledge vector (L, W, H, D, T, E)
- omega: adaptive weight vector in the simplex
- S: saturation score, S = ||K||_omega
- weights: parameter tensors (attached at birth)

State transitions:

    active  --(S >= S* for 3 steps)--> split into two daughters
    active  --(S < S_low for 50 steps)--> dormant
    dormant --(S < S_low for 200 steps)--> dead
    active  --(manual)--> frozen

Split: The parent becomes dormant (not deleted). Each
daughter inherits K_6D with suffix along the dominant
dimension (Corollary 5.2 of Theorem 1).

Frozen cells: requires_grad = False. They cannot be
modified by future training. This is the key mechanism for
zero-forgetting (Lemma 1 of Theorem 3).

## 3.6 DNA Kernel

The DNA Kernel is a regulatory layer of 100 chromosomes,
organized in 10 groups:

    Group         Purpose                       Count
    language      text processing               10
    math          arithmetic, algebra           10
    code          programming                   10
    science       physics, chemistry, biology   10
    sensory       multimodal integration        10
    reasoning     logical inference             10
    memory        long-term storage             10
    coordination  routing, scheduling           10
    adaptation    learning rate control         10
    growth        cell birth/split triggers     10

Each chromosome has 4 genes:
- activation: readiness to express
- sensitivity: threshold for triggering
- persistence: duration of expression
- plasticity: mutation rate

Gene expression is computed as:
    expr = sigmoid(activation + mean_signal - sensitivity)

The DNA Kernel is shared between the base and all
specialists. When a cell splits, the kernel is replicated
with small perturbations (mutation_rate = 0.005).

## 3.7 Federation

The federation is the base model plus N specialists.
Communication is via the SQL registry.

Base model: trained from scratch. Holds the DNA Kernel,
the plexus, and the initial cells.

Specialists: descendants of the base. Each has:
- A UUID
- A parent_id (the base or another specialist)
- A domain_key (SHA256 of domain|version|param_hash)
- A capability fingerprint (canonical capability vector)
- A signature (base's signature over domain_key)

Routing: Base receives a request, embeds it, and looks up
the appropriate specialist in the registry. Routing uses
exact keys, not approximate similarity, when possible.

Incorporation: When a specialist answers a request, the
base verifies the answer (logical, consistency, 6D checks)
and integrates it. Verification guarantees no decrease in
K_6D(B) (Theorem 2.6.1).

## 3.8 Hardware-Adaptive Deployment

Boot sequence (7 stages):
1. Hardware probe: GPU, VRAM, RAM
2. Registry query: fetch candidate specialists
3. Load planning: greedy selection under budget
4. Parallel load: thread pool, up to 4 workers
5. Router warmup: domain embedding index
6. Gap queue restore: pending knowledge requests
7. Ready: base + loaded specialists

Modes: bf16, int8, int4. Selected per specialist based on
available VRAM. Safe ratio = 0.85.

LRU eviction: if a requested specialist is not loaded
and VRAM is insufficient, the least-recently-used specialist
is evicted. Zero-Error: never evict the last loaded
specialist during an in-flight request.

## 3.9 Zero-Error Invariant

Every layer enforces the Zero-Error Invariant:

- Sensors: token ID in [0, vocab_size)
- Spinal Cord: rule match is deterministic
- Plexus: edge weight in [W_min, W_max], no isolated nodes
- Cells: K_6D in [0,1]^6, omega in simplex, S in [0,1]
- DNA Kernel: gene values in [0, 1]
- Federation: UUID unique, signature valid
- Registry: foreign keys enforced, append-only log

Any violation triggers FAIL_CLOSED. This is a runtime
guarantee, not a mathematical claim (see fail_closed_integrity.md).

## 3.10 What This Architecture Is Not

- Not a transformer replacement. Cells can use transformers
  internally.
- Not a MoE variant. Cells are born dynamically, not fixed.
- Not a RAG system. Knowledge is in weights, not in an
  external store.
- Not a NAS method. Growth is triggered by knowledge
  saturation, not by loss.
- Not a completion of AGI. It is a step toward more
  adaptive architectures.

## 3.11 Parameter Counts (Target)

    Component        PoC          Final
    Base model       50M          1.5B
    Each specialist  50M          1.5B
    DNA Kernel       ~1000        ~1000
    Plexus           0 (graph)    0 (graph)
    Registry         0 (SQL)      0 (SQL)

The DNA Kernel and plexus add negligible parameter cost
but substantial behavioral richness.

## 3.12 Summary

Berna R5 is a 6-layer architecture: sensors, spinal cord,
plexus, cells, DNA kernel, registry. Each layer has a clear
interface and is independently testable. Growth is dynamic
(birth, split, death). Zero-forgetting is proven. Federation
is SQL-based with cryptographic keys. Hardware-adaptive
deployment supports 8 GB VRAM and up.
