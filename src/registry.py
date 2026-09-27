"""Registry: SQL interface for Berna R5 federation."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import REGISTRY_DB, BernaFatalError


def new_uuid() -> str:
    return str(uuid.uuid4())


@contextmanager
def connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Context-managed SQLite connection with foreign keys ON."""
    path = db_path or REGISTRY_DB
    if not Path(path).exists():
        raise BernaFatalError(f"Registry DB missing: {path}")
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================================================
# Base Models
# ============================================================
def register_base(
    name: str,
    version: str,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    param_count: int,
    config_hash: str,
    checkpoint_path: str,
) -> str:
    base_id = new_uuid()
    with connect() as c:
        c.execute(
            """INSERT INTO base_models
               (base_id, name, version, vocab_size, hidden_size,
                num_layers, param_count, config_hash, checkpoint_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (base_id, name, version, vocab_size, hidden_size,
             num_layers, param_count, config_hash, checkpoint_path),
        )
    return base_id


def get_base(base_id: str) -> Optional[dict]:
    with connect() as c:
        row = c.execute("SELECT * FROM base_models WHERE base_id = ?",
                        (base_id,)).fetchone()
    return dict(row) if row else None


# ============================================================
# Specialists
# ============================================================
def register_specialist(
    parent_id: str,
    domain: str,
    version: str,
    param_count: int,
    size_bytes: int,
    checkpoint_path: str,
    accuracy_score: float = 0.0,
) -> str:
    sid = new_uuid()
    with connect() as c:
        c.execute(
            """INSERT INTO specialists
               (specialist_id, parent_id, domain, version, param_count,
                size_bytes, accuracy_score, checkpoint_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (sid, parent_id, domain, version, param_count,
             size_bytes, accuracy_score, checkpoint_path),
        )
    return sid


def list_specialists(parent_id: Optional[str] = None,
                     domain: Optional[str] = None) -> list:
    q = "SELECT * FROM specialists WHERE 1=1"
    args = []
    if parent_id:
        q += " AND parent_id = ?"; args.append(parent_id)
    if domain:
        q += " AND domain = ?"; args.append(domain)
    q += " ORDER BY accuracy_score DESC"
    with connect() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(r) for r in rows]


# ============================================================
# Cells
# ============================================================
def register_cell(
    owner_id: str,
    owner_type: str,
    domain: str,
    birth_step: int,
    parent_cell: Optional[str] = None,
) -> str:
    if owner_type not in ("base", "specialist"):
        raise BernaFatalError(f"Invalid owner_type: {owner_type}")
    cid = new_uuid()
    with connect() as c:
        c.execute(
            """INSERT INTO cells
               (cell_id, owner_id, owner_type, domain, parent_cell, birth_step)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cid, owner_id, owner_type, domain, parent_cell, birth_step),
        )
    return cid


def update_cell_k(cell_id: str, K: dict, saturation: float,
                  utilization: float = 0.0) -> None:
    with connect() as c:
        c.execute(
            """UPDATE cells SET
                 k_length=?, k_width=?, k_height=?, k_depth=?,
                 k_time=?, k_encompass=?, saturation=?, utilization=?
               WHERE cell_id=?""",
            (K["L"], K["W"], K["H"], K["D"], K["T"], K["E"],
             saturation, utilization, cell_id),
        )


def set_cell_state(cell_id: str, state: str) -> None:
    if state not in ("active", "dormant", "dead", "frozen"):
        raise BernaFatalError(f"Invalid cell state: {state}")
    with connect() as c:
        c.execute("UPDATE cells SET state=? WHERE cell_id=?",
                  (state, cell_id))


# ============================================================
# Plexus Edges
# ============================================================
def add_edge(source_cell: str, target_cell: str,
             weight: float = 0.0) -> str:
    eid = new_uuid()
    with connect() as c:
        c.execute(
            """INSERT OR REPLACE INTO plexus_edges
               (edge_id, source_cell, target_cell, weight)
               VALUES (?, ?, ?, ?)""",
            (eid, source_cell, target_cell, weight),
        )
    return eid


def update_edge_weight(edge_id: str, weight: float) -> None:
    with connect() as c:
        c.execute("UPDATE plexus_edges SET weight=?, updated_at=CURRENT_TIMESTAMP WHERE edge_id=?",
                  (weight, edge_id))


# ============================================================
# Mutations
# ============================================================
def log_mutation(cell_id: str, trigger_source: str,
                 reliability: float, delta_hash: str,
                 source_uuid: Optional[str] = None) -> str:
    mid = new_uuid()
    with connect() as c:
        c.execute(
            """INSERT INTO mutations
               (mutation_id, cell_id, trigger_source, source_uuid,
                reliability, delta_hash)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (mid, cell_id, trigger_source, source_uuid,
             reliability, delta_hash),
        )
    return mid


# ============================================================
# Requests / Incorporations / Gaps (Theory 2)
# ============================================================
def log_request(query_hash: str, context_hash: Optional[str] = None,
                source: str = "user") -> str:
    rid = new_uuid()
    with connect() as c:
        c.execute(
            """INSERT INTO requests (request_id, query_hash, context_hash, source)
               VALUES (?, ?, ?, ?)""",
            (rid, query_hash, context_hash, source),
        )
    return rid


def log_incorporation(request_id: str, answer_hash: str,
                      confidence: float, reliability: float,
                      specialist_id: Optional[str] = None,
                      verified: bool = False) -> str:
    iid = new_uuid()
    with connect() as c:
        c.execute(
            """INSERT INTO incorporations
               (incorporation_id, request_id, specialist_id, answer_hash,
                confidence, reliability, verified)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (iid, request_id, specialist_id, answer_hash,
             confidence, reliability, 1 if verified else 0),
        )
    return iid


def add_gap(request_id: str, cluster_id: Optional[str] = None,
            priority: float = 0.0) -> str:
    gid = new_uuid()
    with connect() as c:
        c.execute(
            """INSERT INTO gaps (gap_id, request_id, cluster_id, priority)
               VALUES (?, ?, ?, ?)""",
            (gid, request_id, cluster_id, priority),
        )
    return gid


# ============================================================
# Registry Log (append-only)
# ============================================================
def append_registry_log(entry_type: str, payload_hash: str,
                        parent_hash: Optional[str] = None,
                        signature: Optional[str] = None) -> int:
    with connect() as c:
        cur = c.execute(
            """INSERT INTO registry_log (entry_type, payload_hash, parent_hash, signature)
               VALUES (?, ?, ?, ?)""",
            (entry_type, payload_hash, parent_hash, signature),
        )
        return cur.lastrowid


# ============================================================
# Diagnostics
# ============================================================
def stats() -> dict:
    with connect() as c:
        out = {}
        for tbl in ("base_models", "specialists", "cells",
                    "plexus_edges", "mutations", "requests",
                    "incorporations", "gaps", "growth_events",
                    "deployments", "routing_log", "task_metrics"):
            row = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
            out[tbl] = row[0]
    return out
