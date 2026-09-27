"""Berna R5 configuration constants."""

from pathlib import Path

# Paths
ROOT = Path("/data/berna-r5")
REGISTRY_DB = ROOT / "registry" / "specialists.db"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
EXPERIMENTS = ROOT / "experiments"

# Theory 1 - Saturation
S_STAR = 0.85
S_LOW = 0.15
E_MIN_FOR_SPLIT = 0.5
T_CONSECUTIVE_SPLIT = 3
T_CONSECUTIVE_DORMANCY = 50
T_CONSECUTIVE_DEATH = 200
ETA_OMEGA = 0.01

# Theory 1 - 6D dimension priority (tie-break order)
DIM_PRIORITY = {"E": 0, "D": 1, "L": 2, "H": 3, "W": 4, "T": 5}
DIM_ORDER = ["L", "W", "H", "D", "T", "E"]

# Theory 2 - Incorporation
TAU_ROUTE = 0.7
TAU_CONF = 0.8
TAU_GAP = 0.85
TAU_PRIO = 5.0
EPSILON_NEG = 1e-6

# Theory 4 - Growth
TAU_DIV = 0.3
SAFE_RATIO = 0.85

# Theory 5 - Plexus
ETA_E = 0.01
LAMBDA_E = 0.001
W_MIN = 1e-6
W_MAX = 1.0

# Zero-Error
FAIL_CLOSED = True


class BernaFatalError(RuntimeError):
    """Raised on any Zero-Error violation. Always FAIL_CLOSED."""
    pass
