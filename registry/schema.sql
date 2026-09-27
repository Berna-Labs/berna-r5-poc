-- Berna R5 Registry Schema
-- Tracks lineage, routing, and versioning of all cells, specialists, and deployments.

PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. Base Models
-- ============================================================
CREATE TABLE IF NOT EXISTS base_models (
    base_id         TEXT PRIMARY KEY,           -- UUID
    name            TEXT NOT NULL,              -- "berna-r5-base"
    version         TEXT NOT NULL,              -- "v1.0.0"
    vocab_size      INTEGER NOT NULL,
    hidden_size     INTEGER NOT NULL,
    num_layers      INTEGER NOT NULL,
    param_count     INTEGER NOT NULL,
    config_hash     TEXT NOT NULL,              -- SHA256 of config.json
    checkpoint_path TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 2. Specialists
-- ============================================================
CREATE TABLE IF NOT EXISTS specialists (
    specialist_id   TEXT PRIMARY KEY,           -- UUID
    parent_id       TEXT NOT NULL,              -- FK to base_models
    domain          TEXT NOT NULL,              -- "blockchain", "medical"
    version         TEXT NOT NULL,
    param_count     INTEGER NOT NULL,
    size_bytes      INTEGER NOT NULL,
    accuracy_score  REAL,
    checkpoint_path TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES base_models(base_id)
);

-- ============================================================
-- 3. Knowledge Cells
-- ============================================================
CREATE TABLE IF NOT EXISTS cells (
    cell_id         TEXT PRIMARY KEY,           -- UUID
    owner_id        TEXT NOT NULL,              -- base or specialist UUID
    owner_type      TEXT NOT NULL CHECK (owner_type IN ('base','specialist')),
    domain          TEXT NOT NULL,
    parent_cell     TEXT,                       -- for split lineage
    birth_step      INTEGER NOT NULL,
    saturation      REAL DEFAULT 0.0,           -- S(c, t) from Theory 1
    k_length        REAL DEFAULT 0.0,           -- L dimension
    k_width         REAL DEFAULT 0.0,           -- W dimension
    k_height        REAL DEFAULT 0.0,           -- H dimension
    k_depth         REAL DEFAULT 0.0,           -- D dimension
    k_time          REAL DEFAULT 1.0,           -- T dimension
    k_encompass     REAL DEFAULT 0.0,           -- E dimension
    utilization     REAL DEFAULT 0.0,
    state           TEXT DEFAULT 'active' CHECK (state IN ('active','dormant','dead','frozen')),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 4. Plexus Edges
-- ============================================================
CREATE TABLE IF NOT EXISTS plexus_edges (
    edge_id         TEXT PRIMARY KEY,
    source_cell     TEXT NOT NULL,
    target_cell     TEXT NOT NULL,
    weight          REAL NOT NULL DEFAULT 0.0,
    co_activation   REAL DEFAULT 0.0,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_cell) REFERENCES cells(cell_id),
    FOREIGN KEY (target_cell) REFERENCES cells(cell_id),
    UNIQUE (source_cell, target_cell)
);

-- ============================================================
-- 5. Mutations (audit trail)
-- ============================================================
CREATE TABLE IF NOT EXISTS mutations (
    mutation_id     TEXT PRIMARY KEY,
    cell_id         TEXT NOT NULL,
    trigger_source  TEXT NOT NULL,              -- incorporation / gap / internal
    source_uuid     TEXT,                       -- reference to source event
    reliability     REAL NOT NULL,              -- 0.0 - 1.0
    delta_hash      TEXT NOT NULL,              -- SHA256 of delta
    applied_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cell_id) REFERENCES cells(cell_id)
);

-- ============================================================
-- 6. Requests (Theory 2)
-- ============================================================
CREATE TABLE IF NOT EXISTS requests (
    request_id      TEXT PRIMARY KEY,
    query_hash      TEXT NOT NULL,
    context_hash    TEXT,
    source          TEXT,                       -- user / internal
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 7. Incorporations (Theory 2)
-- ============================================================
CREATE TABLE IF NOT EXISTS incorporations (
    incorporation_id TEXT PRIMARY KEY,
    request_id       TEXT NOT NULL,
    specialist_id    TEXT,
    answer_hash      TEXT NOT NULL,
    confidence       REAL NOT NULL,
    reliability      REAL NOT NULL,
    verified         INTEGER NOT NULL DEFAULT 0, -- 0/1
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (request_id) REFERENCES requests(request_id),
    FOREIGN KEY (specialist_id) REFERENCES specialists(specialist_id)
);

-- ============================================================
-- 8. Gap Queue (Theory 2)
-- ============================================================
CREATE TABLE IF NOT EXISTS gaps (
    gap_id          TEXT PRIMARY KEY,
    request_id      TEXT NOT NULL,
    cluster_id      TEXT,
    priority        REAL DEFAULT 0.0,
    status          TEXT DEFAULT 'pending' CHECK (status IN ('pending','training','resolved','abandoned')),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at     TIMESTAMP,
    FOREIGN KEY (request_id) REFERENCES requests(request_id)
);

-- ============================================================
-- 9. Growth Events (Theory 4)
-- ============================================================
CREATE TABLE IF NOT EXISTS growth_events (
    event_id        TEXT PRIMARY KEY,
    parent_id       TEXT NOT NULL,
    d1_id           TEXT NOT NULL,
    d2_id           TEXT NOT NULL,
    benefit         REAL NOT NULL,
    cost            REAL NOT NULL,
    divergence      REAL NOT NULL,
    accepted        INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 10. Deployments
-- ============================================================
CREATE TABLE IF NOT EXISTS deployments (
    deployment_id   TEXT PRIMARY KEY,
    base_id         TEXT NOT NULL,
    hw_profile_hash TEXT NOT NULL,
    loaded_ids      TEXT NOT NULL,              -- JSON array of specialist_ids
    mode_map        TEXT NOT NULL,              -- JSON dict: specialist_id -> mode
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (base_id) REFERENCES base_models(base_id)
);

-- ============================================================
-- 11. Routing Log
-- ============================================================
CREATE TABLE IF NOT EXISTS routing_log (
    log_id          TEXT PRIMARY KEY,
    request_hash    TEXT NOT NULL,
    selected_id     TEXT,
    latency_ms      INTEGER,
    confidence      REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 12. Task Metrics (Theory 3, zero-forgetting)
-- ============================================================
CREATE TABLE IF NOT EXISTS task_metrics (
    metric_id       TEXT PRIMARY KEY,
    task_name       TEXT NOT NULL,
    step            INTEGER NOT NULL,
    performance     REAL NOT NULL,
    forgetting      REAL DEFAULT 0.0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 13. Plexus Log (audit trail for edge changes)
-- ============================================================
CREATE TABLE IF NOT EXISTS plexus_log (
    log_id          TEXT PRIMARY KEY,
    action          TEXT NOT NULL CHECK (action IN ('add','remove','reweight')),
    source_cell     TEXT NOT NULL,
    target_cell     TEXT NOT NULL,
    old_weight      REAL,
    new_weight      REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 14. Registry Log (append-only, hash-chained)
-- ============================================================
CREATE TABLE IF NOT EXISTS registry_log (
    entry_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_type      TEXT NOT NULL,
    payload_hash    TEXT NOT NULL,
    parent_hash     TEXT,
    signature       TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Indexes
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_specialists_domain    ON specialists(domain);
CREATE INDEX IF NOT EXISTS idx_specialists_parent    ON specialists(parent_id);
CREATE INDEX IF NOT EXISTS idx_cells_domain          ON cells(domain);
CREATE INDEX IF NOT EXISTS idx_cells_owner           ON cells(owner_id);
CREATE INDEX IF NOT EXISTS idx_cells_state           ON cells(state);
CREATE INDEX IF NOT EXISTS idx_edges_source          ON plexus_edges(source_cell);
CREATE INDEX IF NOT EXISTS idx_edges_target          ON plexus_edges(target_cell);
CREATE INDEX IF NOT EXISTS idx_mutations_cell        ON mutations(cell_id);
CREATE INDEX IF NOT EXISTS idx_gaps_status           ON gaps(status);
CREATE INDEX IF NOT EXISTS idx_gaps_priority         ON gaps(priority DESC);
CREATE INDEX IF NOT EXISTS idx_routing_request       ON routing_log(request_hash);
CREATE INDEX IF NOT EXISTS idx_task_metrics_step     ON task_metrics(step);


-- ============================================================
-- 15. Specialist Keys (precise binding to base)
-- ============================================================
CREATE TABLE IF NOT EXISTS specialist_keys (
    key_id          TEXT PRIMARY KEY,
    specialist_id   TEXT NOT NULL UNIQUE,
    domain_key      TEXT NOT NULL UNIQUE,   -- hash(domain + version + param_hash)
    capability_vec  TEXT NOT NULL,          -- JSON: canonical capability embedding
    public_key      TEXT NOT NULL,          -- ed25519 public key (hex)
    parent_signature TEXT NOT NULL,          -- base signature over domain_key
    issued_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked_at      TIMESTAMP,
    FOREIGN KEY (specialist_id) REFERENCES specialists(specialist_id)
);

-- ============================================================
-- 16. Signed Answers (audit trail)
-- ============================================================
CREATE TABLE IF NOT EXISTS signed_answers (
    answer_id       TEXT PRIMARY KEY,
    request_id      TEXT NOT NULL,
    specialist_id   TEXT NOT NULL,
    answer_hash     TEXT NOT NULL,
    signature       TEXT NOT NULL,          -- specialist signature (hex)
    verified        INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (request_id) REFERENCES requests(request_id),
    FOREIGN KEY (specialist_id) REFERENCES specialists(specialist_id)
);

CREATE INDEX IF NOT EXISTS idx_keys_domain ON specialist_keys(domain_key);
CREATE INDEX IF NOT EXISTS idx_keys_spec   ON specialist_keys(specialist_id);
CREATE INDEX IF NOT EXISTS idx_signed_req  ON signed_answers(request_id);
