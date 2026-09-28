---
language:
  - en
tags:
  - language-model
  - from-scratch
  - continual-learning
  - bio-inspired
  - zero-forgetting
  - mixture-of-experts
  - dna-kernel
license: other
license_name: berna-research-1.0
license_link: LICENSE
library_name: pytorch
---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22998311.svg)](https://doi.org/10.5281/zenodo.22998311)
[![ORCID](https://img.shields.io/badge/ORCID-0009-0002-5866-8815-A6CE39?logo=orcid)](https://orcid.org/0009-0002-5866-8815)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-3TVehbAAAAAJ-4285F4?logo=googlescholar)](https://scholar.google.com/citations?user=3TVehbAAAAAJ)
[![Latest](https://img.shields.io/badge/Zenodo-v8.0-blue)](https://zenodo.org/records/23015308)

# Berna R5

**DNA-Kernel Plexus: A Bio-Inspired Architecture for Growing,
Self-Correcting Language Models with Bounded Forgetting**

## Model Description

Berna R5 is a from-scratch language model architecture built on
dynamic knowledge cells that are born, split, and interconnect
according to a 6-dimensional knowledge state space.

This is the **50M-parameter Proof of Concept (PoC)** release,
used to validate five theoretical results before scaling to 1.5B.

## Architecture

- **Layers:** 6-stage pipeline (sensors, spinal cord, plexus,
  cells, DNA kernel, registry)
- **Parameters:** 50M (PoC) / 1.5B (target)
- **Vocabulary:** 200,000 tokens
  - Text: 0 - 63,999
  - Vision: 64,000 - 163,999
  - Audio: 164,000 - 199,999
- **Context length:** 8192
- **DNA chromosomes:** 100 (10 groups x 10)
- **Knowledge dimensions:** 6 (L, W, H, D, T, E)

## The Five Theorems

1. **Saturation Equilibrium** - S* = 0.85
2. **Reliable Incorporation** - epsilon_neg = 1e-6
3. **Zero-Forgetting** - F_j <= 1e-6 (proven)
4. **Balanced Growth** - cell count bounded by capacity
5. **Plexus Connectivity** - weights converge to fixed point

Full proofs in the GitHub repository under theory/proofs/.

## Intended Use

**Primary use:** research on continual learning, dynamic
architectures, and bio-inspired neural networks.

**Not intended for:**
- Production deployment (PoC scale)
- High-stakes decisions
- Direct comparison with GPT-4-class models

## Training Data

- WikiText-103 (~100M tokens)
- The Stack (Python subset, ~500M tokens)
- OpenWebMath (~200M tokens)
- Total: ~800M tokens

## Training Procedure

- Optimizer: AdamW
- Learning rate: 3e-4
- Batch size: 4
- Sequence length: 512
- Seed: 42
- Hardware: RTX 5090 (32 GB VRAM) / CPU for validation

## Evaluation

Five theorems validated on PoC (see paper):

| Experiment | Result | Pass |
|---|---|---|
| Saturation convergence | |S - S*| = 0.053 | Yes |
| Split capacity | 19 <= 20 | Yes |
| Incorporation delta | min = +0.02 | Yes |
| Zero-forgetting | max F_j = 0.0 | Yes |
| Plexus convergence | True | Yes |

## Limitations

- PoC scale only (50M). 1.5B is future work.
- No benchmarks (MMLU, HumanEval, GSM8K) run yet.
- Multi-node federation untested.
- 100-chromosome design not empirically optimized.

## The Fail-Closed Integrity Invariant

Any violation of a declared invariant triggers FAIL_CLOSED.
There are no silent failures. This is a runtime guarantee,
not a mathematical claim.

## Citation

If you use Berna R5 in your research, please cite:

    @misc{muhammed2026berna,
      title={DNA-Kernel Plexus: A Bio-Inspired Architecture for
             Growing, Self-Correcting Language Models with
             Bounded Forgetting},
      author={Muhammed, Mohammed Kamil},
      year={2026},
      publisher={Zenodo},
      doi={10.5281/zenodo.22998311},
      url={https://doi.org/10.5281/zenodo.22998311}
    }

**DOI:** [10.5281/zenodo.22998311](https://doi.org/10.5281/zenodo.22998311)

## Links

- GitHub: https://github.com/berna-labs/berna-r5
- Paper: (to be posted on arXiv)
- Registry schema: registry/schema.sql

## License

Berna Research License v1.0.

- **Research use:** free (academic, scientific, personal).
- **Commercial use:** requires a separate license from Berna Labs,
  following the same standards as the Berna R4 project.

See LICENSE for full terms. Contact info@bernalabs.com
for commercial inquiries.
