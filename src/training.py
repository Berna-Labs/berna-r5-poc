"""PoC training loop integrating DNA Kernel, cells, and plexus."""

import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from .config import (
    S_STAR, T_CONSECUTIVE_SPLIT, BernaFatalError,
)
from .zero_error import assert_finite
from .dna_kernel import DNAKernel
from .cell import KnowledgeCell
from .plexus import Plexus
from .saturation import (
    build_K, compute_saturation, uniform_omega,
    dominant_dimension,
)
from . import registry as reg


# ============================================================
# Simple PoC model (for validation only)
# ============================================================
class PoCModel(nn.Module):
    """Minimal transformer-like model for theorem validation.

    NOT the final Berna R5 architecture. Only used to test
    growth, saturation, incorporation at small scale.
    """

    def __init__(self, vocab_size: int = 32000, hidden: int = 256,
                 n_layers: int = 4, n_heads: int = 4, max_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(max_len, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 4,
            batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        if T > self.max_len:
            raise BernaFatalError(f"Sequence too long: {T} > {self.max_len}")

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
# Training state
# ============================================================
class TrainingState:
    def __init__(self, model: nn.Module, base_id: str):
        self.model = model
        self.base_id = base_id
        self.step = 0
        self.tokens_seen = 0
        self.dna = DNAKernel()
        self.cells = []
        self.plexus = Plexus()
        self.last_log_time = time.time()
        self.last_log_tokens = 0
        self.loss_history = []

    def add_cell(self, domain: str):
        c = KnowledgeCell(
            domain=domain, owner_id=self.base_id,
            owner_type="base", birth_step=self.step,
        )
        self.cells.append(c)
        self.plexus.add_node(c.cell_id, {"domain": domain})
        # Connect to all existing cells
        for other in self.cells[:-1]:
            self.plexus.add_edge(c.cell_id, other.cell_id, weight=0.1)
            self.plexus.add_edge(other.cell_id, c.cell_id, weight=0.1)

    def update_cells_from_loss(self, loss: float):
        """Update cell K_6D based on training signal."""
        for c in self.cells:
            # Simplified update: derive K from loss trend
            # L (length): inverse of loss improvement
            recent = self.loss_history[-10:] if len(self.loss_history) >= 10 else [loss]
            improvement = max(0.0, (recent[0] - recent[-1]) / max(recent[0], 1e-9))
            # Placeholder metrics for PoC (real implementation would measure directly)
            K = build_K(
                L=min(1.0, max(0.0, improvement)),
                W=0.3,
                H=min(1.0, max(0.0, 1.0 - loss / 10.0)),
                D=0.4,
                T=1.0,
                E=min(1.0, max(0.0, self.step / 10000.0)),
            )
            c.update(K, utilization=0.5)
            c.tick()


# ============================================================
# Training loop
# ============================================================
def train_poc(
    model: nn.Module,
    dataloader,
    base_id: str,
    max_steps: int = 50_000,
    lr: float = 3e-4,
    log_every: int = 50,
    save_every: int = 100,
    checkpoint_dir: Optional[Path] = None,
    device: str = "cuda",
) -> TrainingState:
    """Run PoC training with DNA/cell/plexus integration."""
    state = TrainingState(model, base_id)
    state.add_cell("general")

    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    start_time = time.time()

    data_iter = iter(dataloader)
    accum_loss = 0.0
    accum_count = 0

    while state.step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]

        if not torch.isfinite(loss):
            raise BernaFatalError(f"Non-finite loss at step {state.step}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise BernaFatalError(f"Non-finite grad norm at step {state.step}")
        opt.step()

        state.step += 1
        state.tokens_seen += input_ids.numel()
        loss_val = float(loss.item())
        state.loss_history.append(loss_val)
        accum_loss += loss_val
        accum_count += 1

        # Update cells from training signal
        if state.step % 10 == 0:
            state.update_cells_from_loss(loss_val)

        # Logging
        if state.step % log_every == 0:
            now = time.time()
            dt = now - state.last_log_time
            d_tok = state.tokens_seen - state.last_log_tokens
            tps = d_tok / dt if dt > 0 else 0.0
            avg_loss = accum_loss / accum_count
            print(
                f"step {state.step:6d} | loss {avg_loss:.4f} | "
                f"tok/s {tps:.0f} | cells {len(state.cells)}",
                flush=True,
            )
            state.last_log_time = now
            state.last_log_tokens = state.tokens_seen
            accum_loss = 0.0
            accum_count = 0

        # Checkpointing (in-process, atomic at file level)
        if checkpoint_dir and state.step % save_every == 0:
            save_poc_checkpoint(state, checkpoint_dir)

    return state


def save_poc_checkpoint(state: TrainingState, ckpt_dir: Path) -> Path:
    """Atomic save of PoC state."""
    import tempfile, shutil
    final_dir = Path(ckpt_dir) / f"step_{state.step:08d}"
    if final_dir.exists():
        return final_dir
    tmp = Path(tempfile.mkdtemp(dir=str(ckpt_dir)))
    try:
        torch.save(state.model.state_dict(), tmp / "model.pt")
        with open(tmp / "state.json", "w") as f:
            import json
            json.dump({
                "step": state.step,
                "tokens_seen": state.tokens_seen,
                "n_cells": len(state.cells),
            }, f)
        tmp.rename(final_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final_dir
