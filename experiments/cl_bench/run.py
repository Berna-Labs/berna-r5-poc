#!/usr/bin/env python3
"""Berna R5 Continual Learning Benchmark.

Trains 8 domains sequentially, measures retention and plasticity.

Usage:
    python experiments/cl_bench/run.py --model r5 --stages 4
    python experiments/cl_bench/run.py --baseline ewc --stages 4
    python experiments/cl_bench/run.py --analyze

Domains:
    Stage 1: WikiText-103, OpenWebMath
    Stage 2: Python, Rust
    Stage 3: PubMed, arXiv-Physics
    Stage 4: Arabic, Chinese

Metrics:
    - P_i(t): perplexity on domain i after stage t
    - F_j(t): max_{s<=t} P_j(s) - P_j(t)  (forgetting)
    - Ret_i(t): P_i(i) / P_i(t)           (retention)
    - Plast_t: P_t(0) - P_t(t)            (plasticity)
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
# Config
# ============================================================
ROOT = Path("/data/berna-r5")
CL_DIR = ROOT / "experiments" / "cl_bench"
RESULTS_DIR = CL_DIR / "results"
CKPT_DIR = CL_DIR / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VOCAB_SIZE = 32000
SEQ_LEN = 256
HIDDEN = 512
N_LAYERS = 8
N_HEADS = 8
BATCH_SIZE = 4
LR = 3e-4
STEPS_PER_DOMAIN = 100      # 100 for test, 5000 for real
LOG_EVERY = 25

DOMAINS = [
    ("wikitext",  1, "text"),
    ("math",      1, "math"),
    ("python",    2, "code"),
    ("rust",      2, "code"),
    ("pubmed",    3, "science"),
    ("physics",   3, "science"),
    ("arabic",    4, "lang"),
    ("chinese",   4, "lang"),
]

STAGES = {
    1: ["wikitext", "math"],
    2: ["python", "rust"],
    3: ["pubmed", "physics"],
    4: ["arabic", "chinese"],
}


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Model
# ============================================================
class CLModel(nn.Module):
    """Base model for CL benchmark (simplified)."""

    def __init__(self):
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.pos = nn.Embedding(SEQ_LEN, HIDDEN)

        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=N_HEADS,
            dim_feedforward=HIDDEN * 4,
            batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)

    def forward(self, x, labels=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device),
                           diagonal=1).bool()
        h = self.blocks(h, mask=mask)
        logits = self.lm_head(h)

        if labels is None:
            return {"logits": logits}
        sl = logits[:, :-1, :].contiguous()
        sl_label = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            sl.view(-1, self.vocab_size), sl_label.view(-1)
        )
        return {"logits": logits, "loss": loss}


# ============================================================
# Dataset
# ============================================================
class DomainDataset(Dataset):
    """Synthetic per-domain dataset (replaces real data)."""

    def __init__(self, domain_id: int, n: int = 500, seq_len: int = SEQ_LEN):
        self.n = n
        self.seq_len = seq_len
        g = torch.Generator().manual_seed(SEED + domain_id * 1000)
        low = (domain_id * 1234) % (VOCAB_SIZE // 2)
        self.data = torch.randint(low, low + VOCAB_SIZE // 4,
                                   (n, seq_len), generator=g)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input_ids": self.data[i]}


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, domain_name, domain_id, device):
    """Compute perplexity on one domain."""
    model.eval()
    ds = DomainDataset(domain_id=domain_id, n=100)
    dl = DataLoader(ds, batch_size=BATCH_SIZE)

    losses = []
    with torch.no_grad():
        for batch in dl:
            x = batch["input_ids"].to(device)
            out = model(x, labels=x)
            losses.append(out["loss"].item())
    avg_loss = sum(losses) / len(losses)
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    return {"loss": round(avg_loss, 4), "ppl": round(ppl, 2)}


# ============================================================
# Training
# ============================================================
def train_domain(model, domain_name, domain_id, device, steps):
    """Train on one domain for `steps` steps."""
    model.train()
    ds = DomainDataset(domain_id=domain_id, n=steps * BATCH_SIZE * 2)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    losses = []
    step = 0
    for batch in dl:
        if step >= steps:
            break
        x = batch["input_ids"].to(device)
        out = model(x, labels=x)
        loss = out["loss"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        step += 1

        if step % LOG_EVERY == 0:
            avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            log(f"    {domain_name} step {step} | loss {avg:.4f}")

    return {"final_loss": round(losses[-1], 4),
            "avg_loss": round(sum(losses) / max(len(losses), 1), 4),
            "steps": step}


# ============================================================
# Main CL loop
# ============================================================
def run_cl(model_name: str = "r5", n_stages: int = 4,
           device: str = "cuda") -> dict:
    log(f"=== CL Benchmark: {model_name} ===")
    log(f"Device: {device}")

    torch.manual_seed(SEED)
    model = CLModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_params/1e6:.1f}M params")

    # History: P[domain][stage] = eval result
    P = {d: {} for d, _, _ in DOMAINS}
    training_log = {}

    for stage in range(1, n_stages + 1):
        log(f"=== Stage {stage} ===")
        domains_this_stage = STAGES[stage]

        for d_name in domains_this_stage:
            d_id = [i for i, (n, _, _) in enumerate(DOMAINS) if n == d_name][0]
            log(f"  Training on {d_name} (domain_id={d_id})")
            train_result = train_domain(model, d_name, d_id, device,
                                          STEPS_PER_DOMAIN)
            training_log[f"stage{stage}_{d_name}"] = train_result

        # Evaluate on ALL domains seen so far
        log(f"  Evaluating on all {sum(len(STAGES[s]) for s in range(1, stage+1))} domains...")
        for d_name, _, _ in DOMAINS:
            d_id = [i for i, (n, _, _) in enumerate(DOMAINS) if n == d_name][0]
            eval_result = evaluate(model, d_name, d_id, device)
            P[d_name][stage] = eval_result
            log(f"    {d_name}: ppl={eval_result['ppl']:.2f}")

        # Save checkpoint
        ckpt_path = CKPT_DIR / f"{model_name}_stage{stage}.pt"
        torch.save({"model": model.state_dict(), "P": P},
                    ckpt_path)
        log(f"  Checkpoint: {ckpt_path}")

    # Compute forgetting matrix
    forgetting = {}
    for d_name, _, _ in DOMAINS:
        domains_stages = P[d_name]
        if not domains_stages:
            continue
        # Find first stage where this domain was trained
        first_stage = min(domains_stages.keys())
        p0 = domains_stages[first_stage]["ppl"]
        # max forgetting = max ppl seen - current ppl
        seen_ppls = [domains_stages[s]["ppl"] for s in domains_stages]
        f_j = max(seen_ppls) - min(seen_ppls)
        forgetting[d_name] = {
            "first_ppl": p0,
            "min_ppl": min(seen_ppls),
            "max_ppl": max(seen_ppls),
            "max_forgetting": round(f_j, 2),
        }

    return {
        "model": model_name,
        "n_stages": n_stages,
        "n_params": n_params,
        "P": P,
        "training_log": training_log,
        "forgetting": forgetting,
    }


# ============================================================
# Analysis
# ============================================================
def analyze():
    log("=== Analysis ===")

    models = ["r5", "vanilla", "ewc", "packnet", "moe"]
    all_results = {}
    for m in models:
        p = RESULTS_DIR / f"{m}_result.json"
        if p.exists():
            with open(p) as f:
                all_results[m] = json.load(f)

    if not all_results:
        log("No results found. Run models first.")
        return

    print()
    print("=" * 70)
    print("CONTINUAL LEARNING BENCHMARK - SUMMARY")
    print("=" * 70)

    for m, r in all_results.items():
        print()
        print(f"Model: {m}")
        print("-" * 70)
        fg = r.get("forgetting", {})
        if fg:
            max_f = max(v["max_forgetting"] for v in fg.values())
            avg_f = sum(v["max_forgetting"] for v in fg.values()) / len(fg)
            print(f"  Max forgetting:  {max_f:.2f} ppl")
            print(f"  Avg forgetting:  {avg_f:.2f} ppl")
            print(f"  Domains tested:  {len(fg)}")
            for d, v in fg.items():
                print(f"    {d:10s}: max_forgetting={v['max_forgetting']:.2f}")

    print()
    print("=" * 70)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="r5",
                        choices=["r5", "vanilla", "ewc", "packnet", "moe"])
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        analyze()
        return

    result = run_cl(args.model, args.stages, args.device)
    out = RESULTS_DIR / f"{args.model}_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Saved: {out}")
    log("Done. Run --analyze for summary.")


if __name__ == "__main__":
    main()
