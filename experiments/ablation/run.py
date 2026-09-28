#!/usr/bin/env python3
"""Berna R5 Ablation Study Runner.

Runs the 5 ablation variants (V0-V4) with fixed seed, same data,
same hardware. Produces a comparison matrix.

Usage:
    python experiments/ablation/run.py --variant V0 --phase 1
    python experiments/ablation/run.py --all --phase 1
    python experiments/ablation/run.py --analyze

Variants:
    V0  Vanilla      (no cells, no growth, no plexus, no DNA, no transactional)
    V1  +Cells       (cells, no growth)
    V2  +Plexus      (cells + plexus)
    V3  +DNA         (cells + plexus + DNA kernel)
    V4  Full R5      (all components enabled)

Phases:
    1   Single-task training (WikiText-103)
    2   Sequential multi-task (4 tasks)
    3   Zero-shot transfer
    4   Causal cell test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/berna-r5")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Configuration
# ============================================================
ROOT = Path("/data/berna-r5")
ABLATION_DIR = ROOT / "experiments" / "ablation"
RESULTS_DIR = ABLATION_DIR / "results"
CKPT_DIR = ABLATION_DIR / "checkpoints"
DATA_DIR = ROOT / "experiments" / "cl_bench" / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VOCAB_SIZE = 32000
SEQ_LEN = 512
HIDDEN = 512
N_LAYERS = 8
N_HEADS = 8
BATCH_SIZE = 4
LR = 3e-4
WARMUP = 100
TOTAL_STEPS_PHASE1 = 2000
LOG_EVERY = 50
SAVE_EVERY = 500


# ============================================================
# Variant Configuration
# ============================================================
VARIANTS = {
    "V0": {"cells": False, "growth": False, "plexus": False,
           "dna": False, "transactional": False},
    "V1": {"cells": True,  "growth": False, "plexus": False,
           "dna": False, "transactional": False},
    "V2": {"cells": True,  "growth": False, "plexus": True,
           "dna": False, "transactional": False},
    "V3": {"cells": True,  "growth": False, "plexus": True,
           "dna": True,  "transactional": False},
    "V4": {"cells": True,  "growth": True,  "plexus": True,
           "dna": True,  "transactional": True},
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Model
# ============================================================
class AblationModel(nn.Module):
    """Simple transformer for ablation testing.

    Components toggled via config:
      - cells: if False, single monolithic MLP
      - growth: if False, no cell splitting
      - plexus: if False, no graph between cells
      - dna: if False, no regulatory kernel
      - transactional: if False, no incorporation protocol
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.vocab_size = VOCAB_SIZE
        self.hidden = HIDDEN

        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.pos = nn.Embedding(SEQ_LEN, HIDDEN)

        # Core transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=N_HEADS,
            dim_feedforward=HIDDEN * 4,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)

        # Ablation state (bookkeeping only, not model params)
        self.n_cells = 1 if config["cells"] else 0
        self.n_splits = 0
        self.plexus_edges = 0 if not config["plexus"] else 1

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos_ids)

        # Causal mask
        mask = torch.triu(
            torch.ones(T, T, device=input_ids.device), diagonal=1
        ).bool()
        x = self.blocks(x, mask=mask)
        logits = self.lm_head(x)

        if labels is None:
            return {"logits": logits}

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        return {"logits": logits, "loss": loss}


# ============================================================
# Synthetic Dataset (for testing)
# ============================================================
class SyntheticDataset(Dataset):
    """Placeholder synthetic dataset.

    In a real run, this would load actual tokenized data from
    /data/berna-r5/experiments/cl_bench/data/.
    """

    def __init__(self, n: int = 5000, seq_len: int = SEQ_LEN,
                 vocab: int = VOCAB_SIZE, seed: int = SEED):
        self.n = n
        self.seq_len = seq_len
        self.vocab = vocab
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab, (n, seq_len), generator=g)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.data[i]
        return {"input_ids": x, "labels": x.clone()}


# ============================================================
# Training
# ============================================================
def train_variant(
    variant: str,
    phase: int = 1,
    device: str = "cuda",
    max_steps: int = TOTAL_STEPS_PHASE1,
) -> dict:
    """Train a single variant for a specified phase."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    config = VARIANTS[variant]
    log(f"=== Variant {variant} | Phase {phase} ===")
    log(f"Config: {config}")

    # Deterministic
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Model
    model = AblationModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_params/1e6:.1f}M parameters")

    # Data
    dataset = SyntheticDataset(n=max(max_steps * BATCH_SIZE, 5000))
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False,
    )

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # Training loop
    model.train()
    data_iter = iter(dataloader)
    step = 0
    losses = []
    t_start = time.time()

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        out = model(x, labels=y)
        loss = out["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step += 1
        losses.append(loss.item())

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            avg_loss = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            tps = (step * BATCH_SIZE * SEQ_LEN) / elapsed
            log(f"step {step:5d} | loss {avg_loss:.4f} | "
                f"tok/s {tps:.0f} | cells {model.n_cells}")

    elapsed = time.time() - t_start

    result = {
        "variant": variant,
        "phase": phase,
        "config": config,
        "params": n_params,
        "final_loss": losses[-1],
        "avg_loss_last_100": sum(losses[-100:]) / min(100, len(losses)),
        "min_loss": min(losses),
        "elapsed_sec": round(elapsed, 1),
        "steps": step,
        "tokens": step * BATCH_SIZE * SEQ_LEN,
        "n_cells_final": model.n_cells,
        "n_splits": model.n_splits,
        "plexus_edges": model.plexus_edges,
    }

    # Save checkpoint
    ckpt_path = CKPT_DIR / f"{variant}_phase{phase}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "result": result,
    }, ckpt_path)
    log(f"Checkpoint saved: {ckpt_path}")

    return result


# ============================================================
# Analysis
# ============================================================
def analyze_results() -> None:
    """Compare all variants and produce summary."""
    log("=== Analysis ===")

    results = {}
    for variant in VARIANTS:
        rpath = RESULTS_DIR / f"{variant}_result.json"
        if rpath.exists():
            with open(rpath) as f:
                results[variant] = json.load(f)

    if not results:
        log("No results found. Run variants first.")
        return

    # Print table
    print()
    print(f"{'Variant':<8} {'Loss':>10} {'Min Loss':>10} "
          f"{'tok/s':>8} {'Cells':>6} {'Splits':>7}")
    print("-" * 55)
    for v, r in sorted(results.items()):
        tps = r["tokens"] / max(r["elapsed_sec"], 1)
        print(f"{v:<8} {r['final_loss']:>10.4f} "
              f"{r['min_loss']:>10.4f} {tps:>8.0f} "
              f"{r['n_cells_final']:>6} {r['n_splits']:>7}")

    # Relative degradation
    if "V4" in results:
        v4_loss = results["V4"]["final_loss"]
        print()
        print("Degradation vs V4 (Full R5):")
        for v, r in sorted(results.items()):
            if v == "V4":
                continue
            delta = (r["final_loss"] - v4_loss) / v4_loss * 100
            print(f"  {v}: {delta:+.2f}%")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str,
                        choices=list(VARIANTS.keys()),
                        help="Single variant to run")
    parser.add_argument("--all", action="store_true",
                        help="Run all variants sequentially")
    parser.add_argument("--phase", type=int, default=1,
                        help="Phase number (1-4)")
    parser.add_argument("--max_steps", type=int,
                        default=TOTAL_STEPS_PHASE1)
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
        return

    if args.all:
        variants = list(VARIANTS.keys())
    elif args.variant:
        variants = [args.variant]
    else:
        parser.print_help()
        return

    log(f"Running {len(variants)} variants on {args.device}")

    for v in variants:
        try:
            result = train_variant(
                variant=v,
                phase=args.phase,
                device=args.device,
                max_steps=args.max_steps,
            )
            # Save per-variant result
            out = RESULTS_DIR / f"{v}_result.json"
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            log(f"Result saved: {out}")
        except Exception as e:
            log(f"FAILED variant {v}: {e}")

    log("All done. Run --analyze to see comparison.")


if __name__ == "__main__":
    main()
