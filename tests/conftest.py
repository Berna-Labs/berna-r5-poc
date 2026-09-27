"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_K():
    """Standard 6D knowledge vector for tests."""
    return {"L": 0.5, "W": 0.3, "H": 0.7, "D": 0.4, "T": 0.9, "E": 0.6}


@pytest.fixture
def saturated_K():
    """Saturated 6D vector (should trigger split)."""
    return {"L": 0.9, "W": 0.85, "H": 0.9, "D": 0.88, "T": 1.0, "E": 0.92}


@pytest.fixture
def uniform_omega():
    """Uniform omega weights (1/6 each)."""
    return {d: 1.0 / 6 for d in ["L", "W", "H", "D", "T", "E"]}


@pytest.fixture
def sample_specialists():
    """Three mock specialists for routing tests."""
    return [
        {
            "specialist_id": "spec_math",
            "domain": "math",
            "domain_embedding": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "specialist_id": "spec_code",
            "domain": "code",
            "domain_embedding": [0.0, 1.0, 0.0, 0.0],
        },
        {
            "specialist_id": "spec_med",
            "domain": "medical",
            "domain_embedding": [0.0, 0.0, 1.0, 0.0],
        },
    ]


@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite registry for tests."""
    import sqlite3
    schema_path = ROOT / "registry" / "schema.sql"
    db_path = tmp_path / "test_registry.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema_path.read_text())
    conn.commit()
    conn.close()
    return db_path
