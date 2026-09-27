# 2. Related Work

Berna R5 sits at the intersection of five research areas:
Mixture-of-Experts, continual learning, neural architecture
search, retrieval-augmented generation, and biology-inspired
neural networks. We review each and identify the gap.

## 2.1 Mixture-of-Experts (MoE)

MoE models activate a subset of parameters per input, reducing
compute while scaling capacity (Jacobs et al., 1991; Shazeer
et al., 2017; Fedus et al., 2022). Recent work includes
Mixtral (Jiang et al., 2024), DeepSeek-MoE (Dai et al., 2024),
and Switch Transformer (Fedus et al., 2022).

**Key limitation:** experts are fixed at initialization. Adding
a new expert requires retraining the router. MoE also does not
address catastrophic forgetting (each expert is trained
jointly, not sequentially) or localized correction (editing
one expert affects routing).

Berna R5 differs in three ways: (1) cells are born dynamically,
not fixed; (2) routing uses exact keys, not learned similarity;
(3) zero-forgetting is proven, not just measured.

## 2.2 Continual Learning

Continual learning addresses catastrophic forgetting through:

- **Regularization:** EWC (Kirkpatrick et al., 2017),
  SI (Zenke et al., 2017)
- **Parameter isolation:** PackNet (Mallya & Lazebnik, 2018),
  Progressive Nets (Rusu et al., 2016)
- **Replay:** GEM (Lopez-Paz & Ranzato, 2017),
  Experience Replay (Rolnick et al., 2019)
- **Low-rank adaptation:** LoRA (Hu et al., 2021),
  Adapters (Houlsby et al., 2019)

**Key limitation:** these methods reduce forgetting but cannot
prove zero. EWC has a tunable lambda; PackNet depends on
pruning schedule; LoRA adds capacity but does not isolate.

Berna R5 proves F_j <= 1e-6 (Theorem 3) under explicit
assumptions. The proof is falsifiable: a single task with
F_j > 1e-6 refutes it.

## 2.3 Neural Architecture Search (NAS)

NAS automates architecture design (Zoph & Le, 2017;
Elsken et al., 2019). Dynamic variants include:

- **Net2Net:** Net2WiderNet, Net2DeeperNet (Chen et al., 2016)
- **DEN:** Dynamically Expandable Networks (Yoon et al., 2018)
- **Firefly:** (Wu et al., 2020)
- **Once-for-All:** (Cai et al., 2020)

**Key limitation:** growth is triggered by loss or validation
metrics, not by knowledge saturation. No NAS method uses a
6-dimensional knowledge manifold or a DNA-like regulatory
kernel.

Berna R5 differs: growth is triggered by S >= S* (Theorem 1),
constrained by capacity and diversity (Theorem 4).

## 2.4 Retrieval-Augmented Generation (RAG)

RAG augments LLMs with external retrieval (Lewis et al., 2020;
Guu et al., 2020; Borgeaud et al., 2022; Izacard et al., 2023).

**Key limitation:** RAG does not modify model weights. It
surfaces existing text but cannot learn new capabilities.
It also requires an external store (vector DB), adding
deployment complexity.

Berna R5 differs: knowledge is incorporated into the base
through transactional updates (Theorem 2). No external
store is required at inference; the registry is only for
federation management.

## 2.5 Biology-Inspired Neural Networks

Biological inspiration has a long history:

- **NEAT:** NeuroEvolution of Augmenting Topologies
  (Stanley & Miikkulainen, 2002)
- **CPPN:** Compositional Pattern Producing Networks
  (Stanley, 2007)
- **HyperNEAT:** (Gauci & Stanley, 2010)
- **NCA:** Neural Cellular Automata (Mordvintsev et al., 2020)
- **GRN-based ANN:** Gene Regulatory Networks for ML
  (Cangelosi et al., 2010)
- **Darwin Series:** (2023-2024)

**Key limitation:** these methods have not been applied to
billion-parameter language models. Most operate on small
networks (< 1M params). No prior work uses a 100-chromosome
DNA Kernel for LLM regulation.

Berna R5 differs: the DNA Kernel regulates cells at 50M-1.5B
scale, with explicit mapping to cognitive domains.

## 2.6 Distributed LLMs

Federated learning (McMahan et al., 2017; Kairouz et al.,
2021) enables distributed training without sharing raw data.
Model Soup (Wortsman et al., 2022) averages fine-tuned
models. HuggingGPT (Shen et al., 2023) orchestrates multiple
models via an LLM controller.

**Key limitation:** federated learning assumes identical model
architectures and synchronizes via gradient averaging.
Model Soup requires identical architectures. HuggingGPT uses
external models, not specialists derived from a common base.

Berna R5 differs: specialists are descended from the same
base, share the same tokenizer and vocabulary layout, and
communicate via a SQL registry with cryptographic keys.

## 2.7 The Gap

| Area | Growth? | Zero-Forgetting? | Federation? | Proven? |
|---|---|---|---|---|
| MoE | No | No | No | No |
| Continual Learning | No | Approximate | No | No |
| NAS | Yes (loss-based) | No | No | No |
| RAG | No | N/A | No | No |
| Biology-inspired | Yes (small) | No | No | No |
| Federated Learning | No | No | Yes (identical) | No |
| **Berna R5** | **Yes (saturation)** | **Yes** | **Yes (lineage)** | **Yes** |

No existing work combines all four properties. This is the
gap Berna R5 fills.

## 2.8 What We Do Not Claim

We do not claim that Berna R5 is better than any specific
method on any specific benchmark. We claim that it is the
first architecture that combines:

1. Dynamic growth (Theory 1, 4)
2. Proven zero-forgetting (Theory 3)
3. Federation with lineage (Theory 2)
4. Formal convergence guarantees (all 5 theorems)

Whether this combination yields better downstream performance
than existing methods is an empirical question (Section 5).
