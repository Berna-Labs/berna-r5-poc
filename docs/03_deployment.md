# Berna R5: Deployment Protocol (Hardware-Adaptive Federation)

## 1. Purpose

Define how a Berna R5 base model discovers, plans, and loads
its federation of specialists on any hardware, from a single
RTX 5090 upward. Zero manual configuration.

## 2. Boot Sequence (7 stages)

### Stage 1: Hardware Probe
At boot, the base model detects:
    - GPU model and count
    - VRAM per GPU (total + free)
    - System RAM
    - Disk I/O throughput
    - Network bandwidth (if multi-node)

Profile is stored:
    hw_profile = {
        "gpu": "RTX 5090",
        "vram_gb": 32,
        "num_gpus": 1,
        "ram_gb": 92,
        "disk_io_mbps": ...,
        "net_mbps": ...,
    }

Zero-Error: if any value is missing or non-finite => FAIL_CLOSED.

### Stage 2: Registry Query
Base queries the SQL registry for available specialists:
    SELECT specialist_id, domain, checkpoint_path,
           size_bytes, accuracy_score
    FROM specialists
    WHERE parent_id = :base_id
    ORDER BY accuracy_score DESC

Result: candidate list C = { s_1, ..., s_N }.

Zero-Error: if registry is unreachable => FAIL_CLOSED.
If registry is empty => base runs standalone.

### Stage 3: Load Planning
For each candidate s_i, compute:
    footprint(s_i) = size_bytes * load_factor(hw_profile)

where load_factor is:
    - 1.0 for bf16
    - 0.5 for int8
    - 0.25 for int4

Budget:
    budget = vram_gb * safe_ratio  (default 0.85)

Greedy selection:
    selected = []
    remaining = budget
    for s in sorted(C, by accuracy desc):
        if footprint(s) <= remaining:
            selected.append(s)
            remaining -= footprint(s)

Result: load_plan = (selected, mode_per_specialist).

### Stage 4: Parallel Load
Load selected specialists in parallel (up to 4 threads):
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(load_specialist, s) for s in selected]
        for f in futures:
            f.result()   # raises on failure

Zero-Error: if any load fails => unload all, run base standalone.

### Stage 5: Router Warmup
Build domain embedding index for routing:
    for s in selected:
        registry.update_embedding(s.id, embed(s.domain))

Router is now ready to route incoming requests.

### Stage 6: Gap Queue Restore
Load gap queue from disk:
    gaps = load_jsonl("/data/berna-r5/registry/gaps.jsonl")
Log count and top clusters.

### Stage 7: Ready
Log:
    "Berna R5 ready on {gpu}, {vram_gb} GB VRAM"
    "Loaded {len(selected)} specialists: {ids}"
    "Gap queue: {len(gaps)} pending"
    "Idle VRAM: {remaining:.1f} GB"

## 3. Runtime Management

### 3.1 Request Handling
For each incoming request r:
    1. Compute query embedding via base
    2. Query router for best specialist s_i
    3. If s_i is loaded: route request
    4. If s_i is not loaded:
        a. If remaining VRAM suffices: load on demand
        b. Else: evict least-recently-used specialist
        c. Then load s_i
    5. If routing fails (no specialist): add to gap queue

### 3.2 LRU Eviction
Maintain LRU cache of loaded specialists.
Eviction policy:
    victim = argmin(last_used(s) for s in loaded)
Unload victim before loading new.

Zero-Error: never evict the last loaded specialist while a
request for it is in flight.

### 3.3 Concurrent Requests
Support up to K concurrent requests (default K=4):
    - Each request gets its own KV cache
    - Specialists are shared (read-only)
    - Base is shared (read-only during inference)

Zero-Error: if concurrent VRAM exceeds budget => queue requests.

## 4. Hardware-Adaptive Modes

The same weights run in different modes depending on hardware:

| Hardware | VRAM | Base Mode | Specialist Mode | Concurrent |
|---|---|---|---|---|
| RTX 5090 | 32 GB | bf16 | bf16 | 4 |
| RTX 4090 | 24 GB | bf16 | int8 | 3 |
| RTX 3090 | 24 GB | int8 | int8 | 2 |
| RTX 3060 | 12 GB | int8 | int4 | 1 |
| CPU + 32 GB RAM | 0 | int4 | int4 (offload) | 1 |

Mode is selected at load time, not hardcoded.

## 5. Zero-Error Guarantees

Property 5.1 (Atomic Boot).
If any stage fails, base reverts to standalone mode.
No partial federation is exposed to the user.

Property 5.2 (Deterministic Planning).
For fixed (hw_profile, registry, config), load_plan is
deterministic and reproducible.

Property 5.3 (Resource Safety).
At every step, assert:
    used_vram + pending_alloc <= budget
Violation => FAIL_CLOSED.

Property 5.4 (Graceful Degradation).
If loading a specialist fails, base continues without it
and logs the failure. No cascade failure.

## 6. Registry Schema (extended)

Additional table for deployment tracking:

    CREATE TABLE deployments (
        deployment_id   TEXT PRIMARY KEY,
        base_id         TEXT NOT NULL,
        hw_profile_hash TEXT NOT NULL,
        loaded_ids      TEXT NOT NULL,  -- JSON array
        mode_map        TEXT NOT NULL,  -- JSON dict
        started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (base_id) REFERENCES base_models(base_id)
    );

    CREATE TABLE routing_log (
        log_id          TEXT PRIMARY KEY,
        request_hash    TEXT NOT NULL,
        selected_id     TEXT,
        latency_ms      INTEGER,
        confidence      REAL,
        timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

## 7. Multi-Node Extension (future)

If hw_profile has num_gpus > 1 or multi-node flag:
    - Split specialists across GPUs by domain affinity
    - Use NCCL for base-specialist communication
    - Same registry, same routing protocol
    - Fall back to single-node if NCCL unavailable

## 8. Configuration File

    /data/berna-r5/config/deployment.yaml

    base_model: /data/berna-r5/checkpoints/base
    registry_db: /data/berna-r5/registry/specialists.db
    safe_ratio: 0.85
    max_concurrent: 4
    lru_capacity: 8
    log_level: INFO
    fail_closed: true

## 9. Open Questions

1. What if two specialists have the same domain?
   (Tie-break by accuracy_score, then by version.)
2. What if VRAM is exactly equal to footprint?
   (Reserve 5% margin; refuse to load if margin < 5%.)
3. How often to re-probe hardware?
   (On boot only; manual refresh via CLI.)
4. What if registry grows beyond VRAM capacity?
   (LRU + on-demand loading handles this.)
5. Cross-node federation with heterogeneous GPUs?
   (Future work; protocol designed to support it.)

## 10. Summary

Deployment is a 7-stage boot sequence:
    probe -> query -> plan -> load -> warmup -> restore -> ready
with LRU eviction, hardware-adaptive modes, and zero-error
guarantees. The same weights run from 32 GB VRAM down to
CPU-only, without retraining or manual configuration.

