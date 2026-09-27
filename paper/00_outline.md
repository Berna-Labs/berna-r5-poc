# Berna R5: Paper Outline

## Title
**DNA-Kernel Plexus: A Bio-Inspired Architecture for Growing,
Self-Correcting Language Models with Bounded Forgetting**

## Target Venues
- Primary: NeurIPS 2027
- Alternate: ICLR 2027, MLSys 2027
- Fallback: arXiv preprint + workshop

## Abstract (150 words)
We present Berna R5, a language model architecture in which
knowledge is stored in dynamic weighted cells that are born,
split, and interconnect according to a 6-dimensional
knowledge-saturation manifold. Unlike Mixture-of-Experts, our
cells are not fixed at initialization; they grow in response
to knowledge saturation and interconnect via a co-activation
plexus. We present: (1) saturation equilibrium in
6D knowledge state space; (2) reliable mutation via federation
incorporation; (3) zero-forgetting (F_j <= 1e-6); (4) balanced
growth equilibrium; (5) plexus connectivity convergence.
We empirically evaluate the associated predictions on a 50M-parameter PoC
trained on 800M tokens, comparing against Transformer, MoE,
EWC, and PackNet baselines. Berna R5 achieves competitive
loss while reducing forgetting by 5 orders of magnitude.

## Paper Structure (10 pages + appendix)

### 1. Introduction (1 page)
- Problem: LLMs have fixed architectures; cannot grow,
  split, or correct locally.
- Gap: existing methods (MoE, continual learning, RAG) address
  parts, not the whole.
- Contribution: 5 theorems + 7 experiments + open-source code.

### 2. Related Work (1.5 pages)
- 2.1 Mixture-of-Experts (Switch, Mixtral, DeepSeek-MoE)
- 2.2 Continual Learning (EWC, PackNet, Progressive Nets)
- 2.3 Neural Architecture Search (NAS, Net2Net, DEN)
- 2.4 Knowledge Retrieval (RAG, RETRO, Atlas)
- 2.5 Distributed LLMs (Federated Learning, Model Soup)
- 2.6 Biology-Inspired NN (NCA, NEAT, CPPN)

### 3. Architecture (2 pages)
- 3.1 Overview: sensors -> spinal cord -> plexus -> cells
- 3.2 The 6D Knowledge Manifold
- 3.3 DNA Kernel (100 chromosomes, 10 groups)
- 3.4 Knowledge Cells (state machine)
- 3.5 Plexus (dynamic graph)
- 3.6 Federation (base + specialists)
- 3.7 Hardware-Adaptive Deployment

### 4. Theory (2.5 pages)
- 4.1 Hypothesis 1: Saturation-Triggered Splitting Equilibrium (6D)
- 4.2 Invariant 1: Transactional Incorporation
- 4.3 Hypothesis 2: Bounded Forgetting
- 4.4 Theorem 2: Balanced Growth Equilibrium
- 4.5 Theorem 1: Plexus Connectivity Convergence Connectivity

### 5. Experiments (2 pages)
- 5.1 Setup: 50M params, 800M tokens, 1x RTX 5090
- 5.2 Exp 1: Saturation dynamics
- 5.3 Exp 2: Splitting behavior
- 5.4 Exp 3: Incorporation
- 5.5 Exp 4: Zero-forgetting
- 5.6 Exp 5: Baselines
- 5.7 Exp 6: Plexus dynamics
- 5.8 Exp 7: Ablations

### 6. Discussion (0.5 page)
### 7. Conclusion (0.3 page)
### 8. Limitations (0.5 page)
### 9. Broader Impact (0.2 page)
### 10. Reproducibility Statement (0.2 page)

## Appendix
- A. Full Proofs (5 files)
- B. Implementation Details
- C. Additional Experiments
- D. Datasets
- E. Failure Cases

## Figures (planned)
| Fig | Description |
|---|---|
| 1 | Architecture overview |
| 2 | 6D knowledge state space visualization |
| 3 | Saturation curves S(t) |
| 4 | Growth over time N(t) |
| 5 | Loss curves vs baselines |
| 6 | Forgetting comparison |
| 7 | Plexus graph |

## Tables (planned)
| Tab | Description |
|---|---|
| 1 | Comparison with related work |
| 2 | Baselines results |
| 3 | Ablation study |
| 4 | Hyperparameters |
| 5 | Theorem-to-experiment mapping |

## Writing Schedule
- Week 1-2: Introduction + Related Work
- Week 3-4: Architecture + Theory
- Week 5-6: Experiments
- Week 7: Discussion + Limitations
- Week 8: Abstract + polish
- Week 9: Internal review
- Week 10: Submission

## Companion Artifacts
- Code: github.com/berna-labs/berna-r5
- Checkpoints: HuggingFace Hub
- Registry: SQL dump in repo
- Proofs: PDF + markdown

## What This Paper Does NOT Claim
- Not SOTA on any leaderboard
- Not a replacement for GPT-4-class models
- Not proven beyond 50M params
- Not theoretically complete

## What This Paper DOES Claim
- Novel architecture combining 5 proven mechanisms
- 5 theorems with explicit proofs and constants
- 7 reproducible experiments on 1 GPU
- Bounded forgetting observed and validated
- Open-source, hardware-adaptive
