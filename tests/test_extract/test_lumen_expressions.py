"""Tests for primitive_history extraction from anima.db."""

import json
import sqlite3
from pathlib import Path

import pytest

from eisv_lumen.extract.lumen_expressions import extract_primitive_history

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHEMA = """\
CREATE TABLE primitive_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    tokens TEXT NOT NULL,
    category_pattern TEXT,
    warmth REAL,
    brightness REAL,
    stability REAL,
    presence REAL,
    score REAL,
    feedback_signals TEXT,
    UNIQUE(timestamp, tokens)
);
"""

ROWS = [
    (
        "2025-06-01T12:00:00",
        "sense,bright,more",
        "PRESENCE,STATE,CHANGE",
        0.72,
        0.85,
        0.60,
        0.91,
        0.78,
        json.dumps({"state_coherence": 0.8, "drift": 0.1}),
    ),
    (
        "2025-06-01T12:05:00",
        "hold,still",
        "ACTION,STATE",
        0.55,
        0.40,
        0.88,
        0.63,
        0.45,
        json.dumps({"state_coherence": 0.5}),
    ),
    (
        "2025-06-01T12:10:00",
        "reach,far,open",
        "ACTION,CHANGE,STATE",
        0.80,
        0.92,
        0.70,
        0.95,
        0.90,
        json.dumps({"state_coherence": 0.9, "resonance": 0.7}),
    ),
]

INSERT_SQL = """\
INSERT INTO primitive_history
    (timestamp, tokens, category_pattern, warmth, brightness,
     stability, presence, score, feedback_signals)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with sample primitive_history rows."""
    db_path = tmp_path / "anima.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.executemany(INSERT_SQL, ROWS)
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractPrimitiveHistory:
    """Tests for extract_primitive_history()."""

    def test_returns_all_records(self, sample_db: Path) -> None:
        """All rows are returned when no filters are applied."""
        records = extract_primitive_history(sample_db)
        assert len(records) == 3

    def test_tokens_parsed_as_list(self, sample_db: Path) -> None:
        """The comma-separated tokens column is returned as a list."""
        records = extract_primitive_history(sample_db)
        first = records[0]
        assert isinstance(first["tokens"], list)
        assert first["tokens"] == ["sense", "bright", "more"]

    def test_category_pattern_parsed_as_list(self, sample_db: Path) -> None:
        """The comma-separated category_pattern column is returned as a list."""
        records = extract_primitive_history(sample_db)
        first = records[0]
        assert isinstance(first["category_pattern"], list)
        assert first["category_pattern"] == ["PRESENCE", "STATE", "CHANGE"]

    def test_feedback_signals_parsed_as_dict(self, sample_db: Path) -> None:
        """The JSON feedback_signals column is returned as a dict."""
        records = extract_primitive_history(sample_db)
        first = records[0]
        assert isinstance(first["feedback_signals"], dict)
        assert first["feedback_signals"]["state_coherence"] == pytest.approx(0.8)
        assert first["feedback_signals"]["drift"] == pytest.approx(0.1)

    def test_state_at_generation(self, sample_db: Path) -> None:
        """A state_at_generation dict is formed from warmth/brightness/stability/presence."""
        records = extract_primitive_history(sample_db)
        first = records[0]
        state = first["state_at_generation"]
        assert isinstance(state, dict)
        assert state == {
            "warmth": pytest.approx(0.72),
            "brightness": pytest.approx(0.85),
            "stability": pytest.approx(0.60),
            "presence": pytest.approx(0.91),
        }

    def test_min_score_filter(self, sample_db: Path) -> None:
        """Only records with score >= min_score are returned."""
        records = extract_primitive_history(sample_db, min_score=0.7)
        assert len(records) == 2
        scores = [r["score"] for r in records]
        assert all(s >= 0.7 for s in scores)

    def test_min_score_excludes_all(self, sample_db: Path) -> None:
        """A very high min_score returns no records."""
        records = extract_primitive_history(sample_db, min_score=99.0)
        assert records == []

    def test_limit(self, sample_db: Path) -> None:
        """The limit parameter caps the number of returned records."""
        records = extract_primitive_history(sample_db, limit=2)
        assert len(records) == 2

    def test_limit_and_min_score_combined(self, sample_db: Path) -> None:
        """limit and min_score work together correctly."""
        records = extract_primitive_history(sample_db, min_score=0.7, limit=1)
        assert len(records) == 1
        assert records[0]["score"] >= 0.7

    def test_record_contains_expected_keys(self, sample_db: Path) -> None:
        """Each record dict contains the expected top-level keys."""
        records = extract_primitive_history(sample_db)
        expected_keys = {
            "id",
            "timestamp",
            "tokens",
            "category_pattern",
            "state_at_generation",
            "score",
            "feedback_signals",
        }
        for rec in records:
            assert set(rec.keys()) == expected_keys
