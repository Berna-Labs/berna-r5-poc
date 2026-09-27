"""Hardware-Adaptive Deployment: 7-stage boot for Berna R5."""

import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

from .config import (
    REGISTRY_DB, SAFE_RATIO, BernaFatalError,
)
from .zero_error import assert_range, assert_finite
from . import registry as reg


# ============================================================
# Stage 1: Hardware Probe
# ============================================================
def probe_hardware() -> dict:
    """Detect GPU, VRAM, RAM, disk."""
    profile = {
        "gpu": "unknown",
        "vram_gb": 0.0,
        "num_gpus": 0,
        "ram_gb": 0.0,
        "cuda_available": False,
    }

    try:
        import torch
        if torch.cuda.is_available():
            profile["cuda_available"] = True
            profile["num_gpus"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            profile["gpu"] = props.name
            profile["vram_gb"] = props.total_memory / 1e9
    except ImportError:
        pass

    # RAM from /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    profile["ram_gb"] = kb / 1e6
                    break
    except (OSError, ValueError):
        pass

    # Validate
    for k, v in profile.items():
        if isinstance(v, (int, float)):
            assert_finite(v, k)

    if profile["cuda_available"] and profile["vram_gb"] <= 0:
        raise BernaFatalError("CUDA available but VRAM = 0")

    return profile


# ============================================================
# Stage 2: Registry Query
# ============================================================
def query_specialists(base_id: str) -> list:
    """Fetch candidate specialists from registry."""
    try:
        return reg.list_specialists(parent_id=base_id)
    except BernaFatalError as e:
        # Empty registry is allowed; missing DB is not
        if "missing" in str(e).lower():
            raise
        return []


# ============================================================
# Stage 3: Load Planning
# ============================================================
def load_factor(mode: str) -> float:
    return {"bf16": 1.0, "int8": 0.5, "int4": 0.25}.get(mode, 1.0)


def plan_load(candidates: list, vram_gb: float,
              safe_ratio: float = SAFE_RATIO) -> dict:
    """Greedy selection of specialists to load."""
    budget = vram_gb * safe_ratio
    assert_finite(budget, "budget")
    if budget <= 0:
        return {"selected": [], "modes": {}, "remaining_gb": 0.0}

    selected = []
    modes = {}
    remaining = budget
    for s in sorted(candidates, key=lambda x: -x.get("accuracy_score", 0.0)):
        size_gb = s["size_bytes"] / 1e9
        mode = "bf16"
        footprint = size_gb * load_factor(mode)
        if footprint <= remaining:
            selected.append(s["specialist_id"])
            modes[s["specialist_id"]] = mode
            remaining -= footprint
    return {
        "selected": selected,
        "modes": modes,
        "remaining_gb": round(remaining, 3),
    }


# ============================================================
# Stage 4: Parallel Load (pluggable loader)
# ============================================================
def parallel_load(selected_ids: list,
                  loader_fn: Callable[[str], object],
                  max_workers: int = 4) -> dict:
    """Load specialists in parallel via thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    loaded = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(loader_fn, sid): sid for sid in selected_ids}
        for f in as_completed(futures):
            sid = futures[f]
            try:
                loaded[sid] = f.result()
            except Exception as e:
                errors[sid] = str(e)
    return {"loaded": loaded, "errors": errors}


# ============================================================
# Stage 6: Gap Queue Restore
# ============================================================
def restore_gaps() -> int:
    """Load pending gap count from registry."""
    try:
        with reg.connect() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM gaps WHERE status='pending'"
            ).fetchone()
            return row[0] if row else 0
    except BernaFatalError:
        return 0


# ============================================================
# Full Boot Sequence (7 stages)
# ============================================================
def boot(base_id: str,
         loader_fn: Optional[Callable[[str], object]] = None,
         logger: Callable[[str], None] = print) -> dict:
    """Execute 7-stage boot. Returns deployment record."""
    t0 = time.time()

    # Stage 1: Hardware
    hw = probe_hardware()
    logger(f"[1/7] HW probe: {hw['gpu']}, "
           f"{hw['vram_gb']:.1f} GB VRAM, {hw['ram_gb']:.1f} GB RAM")

    # Stage 2: Registry query
    candidates = query_specialists(base_id)
    logger(f"[2/7] Registry: {len(candidates)} candidates")

    # Stage 3: Plan
    plan = plan_load(candidates, hw["vram_gb"])
    logger(f"[3/7] Plan: load {len(plan['selected'])} specialists, "
           f"{plan['remaining_gb']} GB spare")

    # Stage 4: Load
    loaded, errors = {}, {}
    if plan["selected"] and loader_fn is not None:
        result = parallel_load(plan["selected"], loader_fn)
        loaded = result["loaded"]
        errors = result["errors"]
    logger(f"[4/7] Load: {len(loaded)} loaded, {len(errors)} errors")

    # Stage 5: Router warmup
    router_ready = bool(loaded)
    logger(f"[5/7] Router: ready={router_ready}")

    # Stage 6: Gap queue
    gap_count = restore_gaps()
    logger(f"[6/7] Gap queue: {gap_count} pending")

    # Stage 7: Ready
    elapsed = round(time.time() - t0, 3)
    logger(f"[7/7] Ready in {elapsed}s")

    return {
        "hw": hw,
        "plan": plan,
        "loaded": list(loaded.keys()),
        "errors": errors,
        "gap_count": gap_count,
        "elapsed_s": elapsed,
        "ready": True,
    }


def save_deployment(record: dict, path: Optional[Path] = None) -> str:
    """Persist deployment record to registry."""
    dep_id = reg.new_uuid()
    with reg.connect() as c:
        c.execute(
            """INSERT INTO deployments
               (deployment_id, base_id, hw_profile_hash,
                loaded_ids, mode_map)
               VALUES (?, ?, ?, ?, ?)""",
            (dep_id,
             record.get("base_id", "unknown"),
             json.dumps(record["hw"], sort_keys=True)[:64],
             json.dumps(record["loaded"]),
             json.dumps(record["plan"]["modes"])),
        )
    return dep_id
