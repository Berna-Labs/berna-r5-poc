#!/usr/bin/env python3
"""Berna R5 Causal Cell Activation Test.

Proves that knowledge cells are computational units, not metadata.

Three tests:
  A. Ablation Causality: remove one cell, measure loss delta on its domain
  B. Activation Correlation: cell activation correlates with its domain
  C. Gradient Causality: cell receives gradients only from its domain

Usage:
    python experiments/causal_cells/run.py --test ablation
    python experiments/causal_cells/run.py --test activation
    python experiments/causal_cells/run.py --test gradient
    python experiments/causal_cells/run.py --all
    python experiments/causal_cells/run.py --analyze
"""

import argparse
import json
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
CAUSAL_DIR = ROOT / "experiments" / "causal_cells"
RESULTS_DIR = CAUSAL_DIR / "results"
CKPT_DIR = CAUSAL_DIR / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VOCAB_SIZE = 32000
SEQ_LEN = 256
HIDDEN = 512
N_LAYERS = 8
N_HEADS = 8
N_CELLS = 4
BATCH_SIZE = 4
LR = 3e-4

DOMAINS = ["math", "code", "text", "science"]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Model with Causal Cells
# ============================================================
class CausalCell(nn.Module):
    """A computational cell: small MLP with activation tracking."""

    def __init__(self, hidden: int, domain: str, cell_id: int):
        super().__init__()
        self.cell_id = cell_id
        self.domain = domain
        self.hidden = hidden

        # The cell IS a computation (not metadata):
        self.up = nn.Linear(hidden, hidden * 2, bias=False)
        self.down = nn.Linear(hidden * 2, hidden, bias=False)
        self.act = nn.GELU()

        # Activation tracking
        self.last_activation = None
        self.register_buffer("activation_sum", torch.zeros(1))

    def forward(self, x):
        """x: (B, T, H). Returns transformed x."""
        h = self.act(self.up(x))
        out = self.down(h)
        # Track activation magnitude
        with torch.no_grad():
            self.last_activation = out.abs().mean().item()
            self.activation_sum += self.last_activation
        return out


class CausalModel(nn.Module):
    """Transformer with N causal cells. Each cell specializes in a domain."""

    def __init__(self, n_cells: int = N_CELLS):
        super().__init__()
        self.n_cells = n_cells
        self.hidden = HIDDEN
        self.vocab_size = VOCAB_SIZE

        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.pos = nn.Embedding(SEQ_LEN, HIDDEN)

        # Base transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=N_HEADS,
            dim_feedforward=HIDDEN * 4,
            batch_first=True, activation="gelu",
        )
        self.base = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Cells (one per domain)
        self.cells = nn.ModuleList([
            CausalCell(HIDDEN, DOMAINS[i % len(DOMAINS)], i)
            for i in range(n_cells)
        ])

        # Router (soft gating)
        self.router = nn.Linear(HIDDEN, n_cells, bias=False)

        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)

    def forward(self, input_ids, labels=None, return_gates=False):
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos_ids)

        mask = torch.triu(
            torch.ones(T, T, device=input_ids.device), diagonal=1
        ).bool()
        x = self.base(x, mask=mask)

        # Compute routing gates from mean-pooled hidden state
        pooled = x.mean(dim=1)  # (B, H)
        gates = torch.softmax(self.router(pooled), dim=-1)  # (B, n_cells)

        # Apply each cell, weighted by its gate
        cell_outputs = []
        for i, cell in enumerate(self.cells):
            out_i = cell(x)  # (B, T, H)
            # Weight by gate: (B, 1, 1) * (B, T, H)
            weighted = out_i * gates[:, i:i+1, None]
            cell_outputs.append(weighted)

        # Sum of weighted cell outputs
        if cell_outputs:
            x = x + sum(cell_outputs)

        logits = self.lm_head(x)

        if labels is None:
            return {"logits": logits, "gates": gates}

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        if return_gates:
            return {"logits": logits, "loss": loss, "gates": gates}
        return {"logits": logits, "loss": loss}


# ============================================================
# Synthetic Dataset (placeholder)
# ============================================================
class DomainDataset(Dataset):
    """Synthetic data with domain labels."""

    def __init__(self, n: int = 500, seq_len: int = SEQ_LEN,
                 vocab: int = VOCAB_SIZE, domain_id: int = 0, seed: int = SEED):
        self.n = n
        self.seq_len = seq_len
        self.domain_id = domain_id
        # Each domain has a distinct token distribution
        g = torch.Generator().manual_seed(seed + domain_id)
        low = (domain_id * 1000) % (vocab // 2)
        self.data = torch.randint(low, low + vocab // 4,
                                   (n, seq_len), generator=g)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.data[i]
        return {"input_ids": x, "labels": x.clone(),
                "domain_id": self.domain_id}


# ============================================================
# Test A: Ablation Causality
# ============================================================
def test_ablation(model, dataloaders, device):
    """Remove one cell, measure loss delta per domain."""
    log("=== Test A: Ablation Causality ===")
    model.eval()

    # Baseline: all cells intact
    baseline_losses = {}
    for d_id, d_name in enumerate(DOMAINS):
        losses = []
        for batch in dataloaders[d_name]:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            with torch.no_grad():
                out = model(x, labels=y)
            losses.append(out["loss"].item())
        baseline_losses[d_name] = sum(losses) / len(losses)
        log(f"  Baseline loss on {d_name}: {baseline_losses[d_name]:.4f}")

    # Ablate each cell
    delta_matrix = {}
    for cell_id, cell in enumerate(model.cells):
        cell_domain = cell.domain
        log(f"  Ablating cell {cell_id} (domain: {cell_domain})")

        # Save original weights
        orig_up = cell.up.weight.data.clone()
        orig_down = cell.down.weight.data.clone()

        # Zero out
        cell.up.weight.data.zero_()
        cell.down.weight.data.zero_()

        deltas = {}
        for d_name in DOMAINS:
            losses = []
            for batch in dataloaders[d_name]:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                with torch.no_grad():
                    out = model(x, labels=y)
                losses.append(out["loss"].item())
            new_loss = sum(losses) / len(losses)
            delta = new_loss - baseline_losses[d_name]
            deltas[d_name] = round(delta, 4)
        delta_matrix[cell_id] = deltas
        log(f"    Deltas: {deltas}")

        # Restore
        cell.up.weight.data.copy_(orig_up)
        cell.down.weight.data.copy_(orig_down)

    return {
        "baseline_losses": baseline_losses,
        "delta_matrix": delta_matrix,
    }


# ============================================================
# Test B: Activation Correlation
# ============================================================
def test_activation(model, dataloaders, device):
    """Measure cell activation correlation with domain label."""
    log("=== Test B: Activation Correlation ===")
    model.eval()

    # Collect activations per cell per domain
    activations = {i: {d: [] for d in DOMAINS} for i in range(model.n_cells)}

    for d_name in DOMAINS:
        for batch in dataloaders[d_name]:
            x = batch["input_ids"].to(device)
            with torch.no_grad():
                out = model(x, return_gates=True)
            gates = out["gates"]  # (B, n_cells)
            for i in range(model.n_cells):
                activations[i][d_name].extend(
                    gates[:, i].cpu().tolist()
                )

    # Compute per-cell, per-domain mean activation
    corr_matrix = {}
    for i in range(model.n_cells):
        cell_domain = model.cells[i].domain
        mean_act = {
            d: sum(activations[i][d]) / max(len(activations[i][d]), 1)
            for d in DOMAINS
        }
        corr_matrix[i] = {
            "cell_domain": cell_domain,
            "mean_activation_by_domain": {d: round(v, 4) for d, v in mean_act.items()},
            "own_domain": round(mean_act.get(cell_domain, 0), 4),
            "max_other": round(max(v for d, v in mean_act.items()
                                    if d != cell_domain), 4),
        }
        log(f"  Cell {i} (domain={cell_domain}): "
            f"own={corr_matrix[i]['own_domain']:.4f}, "
            f"max_other={corr_matrix[i]['max_other']:.4f}")

    return corr_matrix


# ============================================================
# Test C: Gradient Causality
# ============================================================
def test_gradient(model, dataloaders, device):
    """Measure gradient norms on each cell for each domain."""
    log("=== Test C: Gradient Causality ===")
    model.train()

    grad_matrix = {}
    for d_name in DOMAINS:
        # Zero all gradients
        for p in model.parameters():
            p.grad = None

        # Compute gradient on one batch of this domain
        for batch in dataloaders[d_name]:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            out = model(x, labels=y)
            out["loss"].backward()
            break  # one batch is enough

        # Collect gradient norms per cell
        cell_grads = {}
        for i, cell in enumerate(model.cells):
            g_up = cell.up.weight.grad
            g_down = cell.down.weight.grad
            norm = 0.0
            if g_up is not None:
                norm += g_up.norm().item()
            if g_down is not None:
                norm += g_down.norm().item()
            cell_grads[i] = round(norm, 6)
        grad_matrix[d_name] = cell_grads
        log(f"  Grad norms on {d_name}: {cell_grads}")

    return grad_matrix


# ============================================================
# Analysis
# ============================================================
def analyze():
    log("=== Analysis ===")

    # Load all test results
    results = {}
    for test in ["ablation", "activation", "gradient"]:
        rpath = RESULTS_DIR / f"{test}.json"
        if rpath.exists():
            with open(rpath) as f:
                results[test] = json.load(f)

    if not results:
        log("No results found. Run tests first.")
        return

    # Summary
    print()
    print("=" * 70)
    print("CAUSAL CELL ANALYSIS SUMMARY")
    print("=" * 70)

    if "ablation" in results:
        print()
        print("Test A: Ablation Causality")
        print("-" * 70)
        dm = results["ablation"]["delta_matrix"]
        # Check diagonal dominance
        diagonal_sum = 0
        off_diagonal_sum = 0
        for cell_id_str, deltas in dm.items():
            cell_id = int(cell_id_str)
            cell_domain = DOMAINS[cell_id % len(DOMAINS)]
            own = deltas.get(cell_domain, 0)
            other = max((v for d, v in deltas.items() if d != cell_domain),
                        default=0)
            diagonal_sum += own
            off_diagonal_sum += other
            # Use absolute values for ratio
            ratio = abs(own) / max(abs(other), 1e-6)
            print(f"  Cell {cell_id} ({cell_domain}): "
                  f"own={own:.6f}, max_other={other:.6f}, |ratio|={ratio:.2f}x")
        total_abs = abs(diagonal_sum) + abs(off_diagonal_sum)
        sparsity = abs(diagonal_sum) / max(total_abs, 1e-6)
        print(f"  Sparsity: {sparsity:.2f} (0=uniform, 1=diagonal-only)")

    if "activation" in results:
        print()
        print("Test B: Activation Correlation")
        print("-" * 70)
        am = results["activation"]
        own_avg = sum(int(k) >= 0 and am[k]["own_domain"]
                       for k in am) / max(len(am), 1)
        other_avg = sum(int(k) >= 0 and am[k]["max_other"]
                         for k in am) / max(len(am), 1)
        print(f"  Mean own-domain activation:  {own_avg:.4f}")
        print(f"  Mean max-other activation:   {other_avg:.4f}")
        print(f"  Selectivity ratio:           {own_avg / max(other_avg, 1e-6):.2f}x")

    if "gradient" in results:
        print()
        print("Test C: Gradient Causality")
        print("-" * 70)
        gm = results["gradient"]
        for d_name, cell_grads in gm.items():
            total = sum(cell_grads.values())
            print(f"  Domain {d_name}: total_grad={total:.6f}")
            for cid, gn in cell_grads.items():
                pct = 100 * gn / max(total, 1e-9)
                print(f"    cell {cid}: {gn:.6f} ({pct:.1f}%)")

    print()
    print("=" * 70)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str,
                        choices=["ablation", "activation", "gradient"],
                        help="Which test to run")
    parser.add_argument("--all", action="store_true",
                        help="Run all three tests")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.analyze:
        analyze()
        return

    if not (args.all or args.test):
        parser.print_help()
        return

    # Setup
    torch.manual_seed(SEED)
    log(f"Device: {args.device}")

    model = CausalModel(n_cells=N_CELLS).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_params/1e6:.1f}M params, {N_CELLS} cells")

    # Load checkpoint if exists
    ckpt = CKPT_DIR / "latest.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        log(f"Loaded checkpoint: {ckpt}")

    # Build dataloaders
    dataloaders = {}
    for d_id, d_name in enumerate(DOMAINS):
        ds = DomainDataset(n=100, domain_id=d_id)
        dataloaders[d_name] = DataLoader(ds, batch_size=BATCH_SIZE)

    # Run tests
    tests_to_run = []
    if args.all:
        tests_to_run = ["ablation", "activation", "gradient"]
    else:
        tests_to_run = [args.test]

    for test_name in tests_to_run:
        if test_name == "ablation":
            result = test_ablation(model, dataloaders, args.device)
        elif test_name == "activation":
            result = test_activation(model, dataloaders, args.device)
        elif test_name == "gradient":
            result = test_gradient(model, dataloaders, args.device)

        out_path = RESULTS_DIR / f"{test_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log(f"Saved: {out_path}")

    log("Done. Run --analyze for summary.")


if __name__ == "__main__":
    main()
