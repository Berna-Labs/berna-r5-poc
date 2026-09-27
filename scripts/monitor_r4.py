#!/usr/bin/env python3
"""Berna R4 Real-Time Monitoring & Analysis.

Usage:
    python monitor_r4.py             # one-shot snapshot
    python monitor_r4.py --watch     # continuous refresh every 30s
    python monitor_r4.py --full      # full analysis + history
    python monitor_r4.py --save      # save snapshot to file
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


LOG_FILE = "/data/berna-r4/logs/train.log"
CKPT_DIR = "/data/berna-r4/training/checkpoints"
SNAPSHOT_DIR = "/data/berna-r5/experiments/monitoring"


# ============================================================
# Log Parsing
# ============================================================
LOG_PATTERN = re.compile(
    r"\[(?P<ts>[\d\-: ]+)\] step\s+(?P<step>\d+) \| "
    r"loss (?P<loss>[\d.]+) \| lr (?P<lr>[\d.e-]+) \| "
    r"tok/s (?P<tps>\d+) \| VRAM (?P<vram>[\d.]+)GB \| "
    r"tokens (?P<tokens>[\d,]+)"
)


def parse_log(path: str) -> list:
    """Parse training log; return list of dicts."""
    records = []
    try:
        with open(path) as f:
            for line in f:
                m = LOG_PATTERN.search(line)
                if m:
                    records.append({
                        "ts": m.group("ts").strip(),
                        "step": int(m.group("step")),
                        "loss": float(m.group("loss")),
                        "lr": float(m.group("lr")),
                        "tps": int(m.group("tps")),
                        "vram": float(m.group("vram")),
                        "tokens": int(m.group("tokens").replace(",", "")),
                    })
    except FileNotFoundError:
        pass
    return records


# ============================================================
# Process / GPU Info
# ============================================================
def get_process_info() -> dict:
    """Query the train.py process."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "train.py"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return {"running": False}

    if not out:
        return {"running": False}

    pid = out.split()[0]
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        # utime + stime in clock ticks
        ticks = os.sysconf("SC_CLK_TCK")
        cpu_sec = (int(stat[13]) + int(stat[14])) / ticks
    except (FileNotFoundError, IndexError, ValueError):
        cpu_sec = 0.0

    try:
        etime = subprocess.check_output(
            ["ps", "-o", "etime=", "-p", pid], text=True
        ).strip()
    except subprocess.CalledProcessError:
        etime = "?"

    try:
        rss_kb = int(Path(f"/proc/{pid}/status")
                     .read_text().split("VmRSS:")[1].split()[0])
        rss_gb = rss_kb / 1e6
    except (FileNotFoundError, IndexError, ValueError):
        rss_gb = 0.0

    return {
        "running": True,
        "pid": pid,
        "elapsed": etime,
        "cpu_sec": round(cpu_sec, 1),
        "rss_gb": round(rss_gb, 2),
    }


def get_gpu_info() -> dict:
    """Query nvidia-smi."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,power.limit,clocks.current.sm",
            "--format=csv,noheader,nounits",
        ], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False}

    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 8:
        return {"available": False}

    return {
        "available": True,
        "name": parts[0],
        "util_pct": int(parts[1]),
        "vram_used_mb": int(parts[2]),
        "vram_total_mb": int(parts[3]),
        "temp_c": int(parts[4]),
        "power_w": float(parts[5]),
        "power_limit_w": float(parts[6]),
        "sm_clock_mhz": int(parts[7]),
    }


def get_disk_info() -> dict:
    """Query disk usage for /data."""
    try:
        st = os.statvfs("/data")
        total_gb = (st.f_blocks * st.f_frsize) / 1e9
        free_gb = (st.f_bavail * st.f_frsize) / 1e9
        used_gb = total_gb - free_gb
        return {
            "total_gb": round(total_gb, 1),
            "used_gb": round(used_gb, 1),
            "free_gb": round(free_gb, 1),
            "used_pct": round(100 * used_gb / total_gb, 1),
        }
    except (FileNotFoundError, PermissionError):
        return {}


def get_checkpoints() -> list:
    """List active checkpoints."""
    try:
        ckpts = sorted(Path(CKPT_DIR).glob("step_*"))
        return [c.name for c in ckpts]
    except (FileNotFoundError, PermissionError):
        return []


def get_service_status() -> dict:
    """Query systemd service."""
    try:
        out = subprocess.check_output([
            "systemctl", "show", "berna-train.service",
            "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts",
        ], text=True)
        d = dict(line.split("=", 1) for line in out.strip().split("\n") if "=" in line)
        return d
    except Exception:
        return {}


# ============================================================
# Analysis
# ============================================================
def analyze(records: list) -> dict:
    """Derive statistics from parsed log records."""
    if len(records) < 2:
        return {"insufficient_data": True}

    first = records[0]
    last = records[-1]

    # Time delta
    t0 = datetime.strptime(first["ts"], "%Y-%m-%d %H:%M:%S")
    t1 = datetime.strptime(last["ts"], "%Y-%m-%d %H:%M:%S")
    wall_sec = (t1 - t0).total_seconds()

    # Token delta
    token_delta = last["tokens"] - first["tokens"]
    step_delta = last["step"] - first["step"]

    # Throughput (real) -- use only recent records to avoid resume skew
    if len(records) >= 10:
        recent = records[-10:]
        t_r0 = datetime.strptime(recent[0]["ts"], "%Y-%m-%d %H:%M:%S")
        t_r1 = datetime.strptime(recent[-1]["ts"], "%Y-%m-%d %H:%M:%S")
        dt_recent = (t_r1 - t_r0).total_seconds()
        dtoken_recent = recent[-1]["tokens"] - recent[0]["tokens"]
        real_tps = dtoken_recent / dt_recent if dt_recent > 0 else 0.0
    else:
        real_tps = token_delta / wall_sec if wall_sec > 0 else 0.0

    # Loss trend (linear fit on last N points)
    losses = [r["loss"] for r in records[-20:]]
    steps = [r["step"] for r in records[-20:]]
    if len(losses) >= 2:
        n = len(losses)
        sx = sum(steps); sy = sum(losses)
        sxy = sum(s * l for s, l in zip(steps, losses))
        sxx = sum(s * s for s in steps)
        denom = (n * sxx - sx * sx)
        slope = ((n * sxy - sx * sy) / denom) if denom != 0 else 0.0
        # slope in loss units per step; convert to per-100 steps
        slope_100 = slope * 100
    else:
        slope_100 = 0.0

    # ETA
    TOTAL_TOKENS = 10_919_550_781
    remaining_tokens = TOTAL_TOKENS - last["tokens"]
    eta_sec = remaining_tokens / real_tps if real_tps > 0 else 0
    eta_days = eta_sec / 86400

    # Completion
    pct = 100 * last["tokens"] / TOTAL_TOKENS

    # TPS stability (stddev of last 10)
    tps_recent = [r["tps"] for r in records[-10:]]
    if tps_recent:
        mean_tps = sum(tps_recent) / len(tps_recent)
        var = sum((t - mean_tps) ** 2 for t in tps_recent) / len(tps_recent)
        std_tps = var ** 0.5
    else:
        mean_tps = 0.0
        std_tps = 0.0

    # Loss spike detection (skip first 200 steps to avoid init)
    filtered = [r for r in records if r["step"] > 200]
    recent_losses = [r["loss"] for r in filtered[-50:]]
    spikes = []
    if recent_losses:
        median = sorted(recent_losses)[len(recent_losses) // 2]
        for r in filtered[-50:]:
            if r["loss"] > median * 1.4:
                spikes.append({"step": r["step"], "loss": r["loss"]})

    return {
        "first_step": first["step"],
        "last_step": last["step"],
        "step_delta": step_delta,
        "token_delta": token_delta,
        "wall_sec": round(wall_sec, 1),
        "real_tps": round(real_tps, 1),
        "reported_tps_mean": round(mean_tps, 1),
        "reported_tps_std": round(std_tps, 1),
        "loss_first": first["loss"],
        "loss_last": last["loss"],
        "loss_slope_per_100_steps": round(slope_100, 4),
        "current_tokens": last["tokens"],
        "total_tokens": TOTAL_TOKENS,
        "completion_pct": round(pct, 3),
        "remaining_tokens": remaining_tokens,
        "eta_days": round(eta_days, 2),
        "eta_date": (datetime.now() + timedelta(days=eta_days)).strftime("%Y-%m-%d"),
        "recent_loss_spikes": spikes[:5],
    }


# ============================================================
# Rendering
# ============================================================
def bar(pct: float, width: int = 40) -> str:
    filled = int(pct / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def render(records: list, show_full: bool = False) -> str:
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 70)
    lines.append(f"  Berna R4 -- Monitoring Snapshot @ {now}")
    lines.append("=" * 70)

    # --- Process ---
    proc = get_process_info()
    lines.append("")
    lines.append("[PROCESS]")
    if proc["running"]:
        lines.append(f"  Status    : RUNNING (PID {proc['pid']})")
        lines.append(f"  Elapsed   : {proc['elapsed']}")
        lines.append(f"  CPU time  : {proc['cpu_sec']} sec")
        lines.append(f"  RSS       : {proc['rss_gb']} GB")
    else:
        lines.append("  Status    : NOT RUNNING")

    # --- GPU ---
    gpu = get_gpu_info()
    lines.append("")
    lines.append("[GPU]")
    if gpu.get("available"):
        lines.append(f"  Name      : {gpu['name']}")
        lines.append(f"  Util      : {gpu['util_pct']}%")
        lines.append(f"  VRAM      : {gpu['vram_used_mb']} / {gpu['vram_total_mb']} MB")
        lines.append(f"  Temp      : {gpu['temp_c']} C")
        lines.append(f"  Power     : {gpu['power_w']} / {gpu['power_limit_w']} W")
        lines.append(f"  SM Clock  : {gpu['sm_clock_mhz']} MHz")
    else:
        lines.append("  nvidia-smi unavailable")

    # --- Service ---
    svc = get_service_status()
    lines.append("")
    lines.append("[SYSTEMD]")
    for k in ("ActiveState", "SubState", "NRestarts"):
        if k in svc:
            lines.append(f"  {k:12s}: {svc[k]}")

    # --- Disk ---
    disk = get_disk_info()
    if disk:
        lines.append("")
        lines.append("[DISK]")
        lines.append(f"  /data     : {disk['used_gb']} / {disk['total_gb']} GB "
                     f"({disk['used_pct']}%)")
        lines.append(f"  Free      : {disk['free_gb']} GB")

    # --- Checkpoints ---
    ckpts = get_checkpoints()
    lines.append("")
    lines.append(f"[CHECKPOINTS] ({len(ckpts)})")
    for c in ckpts:
        lines.append(f"  - {c}")

    # --- Analysis ---
    if records:
        a = analyze(records)
        if not a.get("insufficient_data"):
            lines.append("")
            lines.append("[TRAINING]")
            lines.append(f"  Step      : {a['last_step']} "
                         f"(first logged: {a['first_step']})")
            lines.append(f"  Tokens    : {a['current_tokens']:,} / "
                         f"{a['total_tokens']:,}")
            lines.append(f"  Progress  : {bar(a['completion_pct'])} "
                         f"{a['completion_pct']:.2f}%")
            lines.append(f"  Loss      : {a['loss_last']:.4f} "
                         f"(first {a['loss_first']:.4f})")
            lines.append(f"  Loss trend: {a['loss_slope_per_100_steps']:+.4f} "
                         f"per 100 steps")
            lines.append(f"  TPS real  : {a['real_tps']:.1f} tok/s")
            lines.append(f"  TPS log   : {a['reported_tps_mean']:.1f} "
                         f"+/- {a['reported_tps_std']:.1f}")
            lines.append(f"  ETA       : {a['eta_days']:.1f} days "
                         f"(~{a['eta_date']})")

            if a["recent_loss_spikes"]:
                lines.append("")
                lines.append("[SPIKES]")
                for s in a["recent_loss_spikes"]:
                    lines.append(f"  step {s['step']}: loss {s['loss']:.4f}")

    # --- Full history ---
    if show_full and records:
        lines.append("")
        lines.append("[HISTORY]  (last 20 logged steps)")
        lines.append(f"  {'step':>8s}  {'loss':>8s}  {'lr':>10s}  "
                     f"{'tps':>7s}  {'tokens':>14s}")
        for r in records[-20:]:
            lines.append(f"  {r['step']:>8d}  {r['loss']:>8.4f}  "
                         f"{r['lr']:>10.2e}  {r['tps']:>7d}  "
                         f"{r['tokens']:>14,d}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true",
                   help="Continuous refresh")
    p.add_argument("--interval", type=int, default=30,
                   help="Refresh interval seconds (default 30)")
    p.add_argument("--full", action="store_true",
                   help="Include full history table")
    p.add_argument("--save", action="store_true",
                   help="Save snapshot to JSON")
    args = p.parse_args()

    Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

    def once():
        records = parse_log(LOG_FILE)
        out = render(records, show_full=args.full)
        print(out, flush=True)
        if args.save:
            a = analyze(records) if records else {}
            snap = {
                "timestamp": datetime.now().isoformat(),
                "records": records[-50:],
                "analysis": a,
                "process": get_process_info(),
                "gpu": get_gpu_info(),
                "disk": get_disk_info(),
                "checkpoints": get_checkpoints(),
            }
            fname = SNAPSHOT_DIR / f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            fname.write_text(json.dumps(snap, indent=2))
            print(f"Snapshot saved: {fname}")

    if not args.watch:
        once()
        return

    try:
        while True:
            os.system("clear")
            once()
            print(f"\nRefreshing every {args.interval}s "
                  f"(Ctrl+C to stop)...")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
