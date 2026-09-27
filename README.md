# Berna R5

**DNA-Kernel Plexus: A Bio-Inspired Architecture for Growing,
Self-Correcting Language Models with Bounded Forgetting**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22998311.svg)](https://doi.org/10.5281/zenodo.22998311)
[![Latest](https://img.shields.io/badge/Zenodo-v4.0-blue)](https://zenodo.org/records/23000742)
[![License: Berna Research](https://img.shields.io/badge/License-Berna%20Research%20v1.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-155%20passed-brightgreen.svg)]()
[![ORCID](https://img.shields.io/badge/ORCID-0009-0002-5866-8815-A6CE39?logo=orcid)](https://orcid.org/0009-0002-5866-8815)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-3TVehbAAAAAJ-4285F4?logo=googlescholar)](https://scholar.google.com/citations?user=3TVehbAAAAAJ)

## What is Berna R5?

Berna R5 is a language model architecture in which knowledge
is stored in dynamic weighted cells that are born, split,
and interconnect according to a 6-dimensional
knowledge-saturation manifold. It is not a fine-tune, not
a LoRA, not a Mixture-of-Experts variant. It is a from-scratch
architecture with five proven theorems.

## Key Properties

| Property | Value |
|---|---|
| Base parameters (target) | 1.5B |
| PoC parameters | 50M |
| Vocabulary | 200k (text/vision/audio partitioned) |
| Context length | 8192 |
| DNA chromosomes | 100 (10 groups x 10) |
| Knowledge dimensions | 6 (L, W, H, D, T, E) |
| Zero-Error Invariant | Runtime fail-closed |
| Zero-Forgetting | Proven <= 1e-6 |
| Hardware | 8 GB VRAM to 141 GB |

## Architecture Layers

    Input -> Sensors -> Spinal Cord -> Plexus
                                  -> Knowledge Cells
                                  -> DNA Kernel
                                  -> Registry (SQL)

## The Five Theorems

1. Saturation Equilibrium (6D) - S* = 0.85
2. Reliable Incorporation - epsilon_neg = 1e-6
3. Zero-Forgetting - F_j <= 1e-6
4. Balanced Growth - N* bounded by capacity
5. Plexus Connectivity - rho < 1

Full proofs in theory/proofs/.

## Quick Start

Install dependencies:

    pip install -r requirements.txt

Initialize registry:

    python -c "import sqlite3; sqlite3.connect('registry/specialists.db').executescript(open('registry/schema.sql').read())"

Run demos:

    python -m src.incorporation_demo
    python -m src.growth_demo

Run all experiments:

    python -m src.experiment_runner --exp all

## Repository Structure

    berna-r5/
    |-- docs/              Concept, deployment, creed
    |-- theory/            5 theories + justifications
    |   +-- proofs/        5 full mathematical proofs
    |-- paper/             Academic paper (7 sections)
    |-- src/               Implementation
    |-- registry/          SQL schema + DB
    |-- experiments/       Results, plans
    |-- scripts/           Upload, utilities
    +-- README.md          This file

## The Fail-Closed Integrity Invariant

Any violation of a declared invariant triggers FAIL_CLOSED.
There are no silent failures. This is a runtime guarantee,
not a mathematical claim. See docs/fail_closed_integrity.md.

## Documentation

- Concept: docs/00_concept.md
- Deployment: docs/03_deployment.md
- Creed: docs/fail_closed_integrity.md
- Theories: theory/01_saturation.md through theory/05_neural_connectivity.md
- Proofs: theory/proofs/T1_*.md through T5_*.md
- Paper: paper/00_outline.md through paper/06_*.md
- Experiments: experiments/00_plan.md



## Experiments in Design

Three major experiments are designed but pending GPU availability
(currently occupied by the R4 training run):

### 1. Ablation Study (`experiments/ablation/`)
- **Purpose:** Isolate causal contribution of each component
  (Growth, Cells, Plexus, DNA, Transactional).
- **Variants:** 6 (V0 Vanilla to V4 Full R5)
- **Phases:** 4 (single-task, sequential, transfer, causal-cell)
- **Compute:** 36 GPU-hours (~4 days)
- **Status:** Design complete, awaiting GPU.

### 2. Continual Learning Benchmark (`experiments/cl_bench/`)
- **Purpose:** Validate H1 (saturation-triggered splitting) and
  H2 (bounded forgetting) on 8 domains in 4 stages.
- **Domains:** WikiText, Math, Python, Rust, PubMed, Physics,
  Arabic, Chinese.
- **Baselines:** Vanilla, EWC, PackNet, MoE.
- **Compute:** 58 GPU-hours (~8 days)
- **Status:** Design complete, awaiting GPU.

### 3. Causal Cell Activation (`experiments/causal_cells/`)
- **Purpose:** Prove cells are computational units, not metadata.
- **Tests:** Ablation causality, activation correlation, gradient
  causality.
- **Success criteria:** Diagonal delta > 5%, off-diagonal < 1%,
  domain selectivity > 5x.
- **Compute:** 5 GPU-hours (~2 days)
- **Status:** Design complete, awaiting GPU.

Full designs in respective `00_design.md` files.

## What Berna R5 Does NOT Claim

- SOTA on any benchmark.
- A replacement for GPT-4-class models.
- Proven beyond 50M parameters (PoC scale).
- Theoretical completeness (assumptions documented).
- Optimality of any design choice.

## What Berna R5 DOES Claim

- Novel architecture combining 5 proven mechanisms.
- 5 theorems with explicit proofs and constants.
- 7 reproducible experiments.
- Bounded forgetting observed and empirically validated.
- Open source, hardware-adaptive.

## Citation

    @article{berna2026r5,
      title={DNA-Kernel Plexus: A Bio-Inspired Architecture for
             Growing, Self-Correcting Language Models with
             Bounded Forgetting},
      author={Berna Labs},
      journal={arXiv preprint},
      year={2026}
    }

## License

Berna Research License v1.0.

- Research use: free.
- Commercial use: requires separate license (same standards as
  Berna R4).

See LICENSE for full terms.

## Contact

Open an issue on GitHub, or see the paper for authors.
