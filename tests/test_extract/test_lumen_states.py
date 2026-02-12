"""Tests for extracting Lumen state_history from anima.db."""

import json
import sqlite3
from pathlib import Path

import pytest

from eisv_lumen.extract.lumen_states import anima_to_eisv, extract_state_history


SAMPLE_STATES = [
    {
        "timestamp": "2025-12-01T08:00:00",
        "warmth": 0.72, "clarity": 0.85, "stability": 0.90, "presence": 0.80,
        "sensors": {"temp_c": 21.3, "humidity": 45.0, "light_lux": 320},
    },
    {
        "timestamp": "2025-12-01T08:05:00",
        "warmth": 0.65, "clarity": 0.78, "stability": 0.82, "presence": 0.75,
        "sensors": {"temp_c": 21.5, "humidity": 46.0, "light_lux": 310},
    },
    {
        "timestamp": "2025-12-01T08:10:00",
        "warmth": 0.80, "clarity": 0.92, "stability": 0.95, "presence": 0.88,
        "sensors": {"temp_c": 22.0, "humidity": 44.0, "light_lux": 350},
    },
    {
        "timestamp": "2025-12-01T08:15:00",
        "warmth": 0.55, "clarity": 0.60, "stability": 0.70, "presence": 0.65,
        "sensors": {"temp_c": 21.8, "humidity": 47.0, "light_lux": 290},
    },
    {
        "timestamp": "2025-12-01T08:20:00",
        "warmth": 0.90, "clarity": 0.88, "stability": 0.85, "presence": 0.92,
        "sensors": {"temp_c": 22.1, "humidity": 43.0, "light_lux": 360},
    },
]


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "anima.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE state_history ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp TEXT NOT NULL, "
        "warmth REAL, clarity REAL, stability REAL, presence REAL, "
        "sensors TEXT DEFAULT '{}')"
    )
    for row in SAMPLE_STATES:
        cur.execute(
            "INSERT INTO state_history "
            "(timestamp, warmth, clarity, stability, presence, sensors) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row["timestamp"], row["warmth"], row["clarity"],
             row["stability"], row["presence"], json.dumps(row["sensors"])),
        )
    conn.commit()
    conn.close()
    return db_path


class TestExtractStateHistory:

    def test_returns_all_records(self, sample_db):
        records = extract_state_history(sample_db)
        assert len(records) == 5

    def test_record_keys(self, sample_db):
        records = extract_state_history(sample_db)
        expected = {"id", "timestamp", "warmth", "clarity", "stability", "presence", "sensors"}
        assert set(records[0].keys()) == expected

    def test_sensors_parsed_as_dict(self, sample_db):
        records = extract_state_history(sample_db)
        for rec in records:
            assert isinstance(rec["sensors"], dict)

    def test_sensor_values_correct(self, sample_db):
        records = extract_state_history(sample_db)
        first = records[0]
        assert first["sensors"]["temp_c"] == pytest.approx(21.3)
        assert first["sensors"]["humidity"] == pytest.approx(45.0)
        assert first["sensors"]["light_lux"] == pytest.approx(320)

    def test_numeric_fields_correct(self, sample_db):
        records = extract_state_history(sample_db)
        first = records[0]
        assert first["warmth"] == pytest.approx(0.72)
        assert first["clarity"] == pytest.approx(0.85)
        assert first["stability"] == pytest.approx(0.90)
        assert first["presence"] == pytest.approx(0.80)

    def test_ordering_by_timestamp(self, sample_db):
        records = extract_state_history(sample_db)
        timestamps = [r["timestamp"] for r in records]
        assert timestamps == sorted(timestamps)

    def test_limit_parameter(self, sample_db):
        records = extract_state_history(sample_db, limit=3)
        assert len(records) == 3

    def test_limit_none_returns_all(self, sample_db):
        records = extract_state_history(sample_db, limit=None)
        assert len(records) == 5


class TestAnimaToEISV:

    def test_basic_mapping(self):
        result = anima_to_eisv(warmth=0.72, clarity=0.85, stability=0.90, presence=0.80)
        assert result["E"] == pytest.approx(0.72)
        assert result["I"] == pytest.approx(0.85)
        assert result["S"] == pytest.approx(0.10)
        assert result["V"] == pytest.approx(0.06)

    def test_all_zeros(self):
        result = anima_to_eisv(warmth=0.0, clarity=0.0, stability=0.0, presence=0.0)
        assert result["E"] == pytest.approx(0.0)
        assert result["I"] == pytest.approx(0.0)
        assert result["S"] == pytest.approx(1.0)
        assert result["V"] == pytest.approx(0.3)

    def test_all_ones(self):
        result = anima_to_eisv(warmth=1.0, clarity=1.0, stability=1.0, presence=1.0)
        assert result["E"] == pytest.approx(1.0)
        assert result["I"] == pytest.approx(1.0)
        assert result["S"] == pytest.approx(0.0)
        assert result["V"] == pytest.approx(0.0)

    def test_clamping_high(self):
        result = anima_to_eisv(warmth=1.5, clarity=1.2, stability=-0.5, presence=-0.3)
        assert result["E"] == pytest.approx(1.0)
        assert result["I"] == pytest.approx(1.0)
        assert result["S"] == pytest.approx(1.0)
        assert 0.0 <= result["V"] <= 1.0

    def test_clamping_low(self):
        result = anima_to_eisv(warmth=-0.5, clarity=-0.3, stability=1.5, presence=1.5)
        assert result["E"] == pytest.approx(0.0)
        assert result["I"] == pytest.approx(0.0)
        assert result["S"] == pytest.approx(0.0)
        assert result["V"] == pytest.approx(0.0)

    def test_returns_all_four_keys(self):
        result = anima_to_eisv(0.5, 0.5, 0.5, 0.5)
        assert set(result.keys()) == {"E", "I", "S", "V"}


class TestExtractWithEISV:

    def test_eisv_key_present(self, sample_db):
        records = extract_state_history(sample_db, compute_eisv=True)
        for rec in records:
            assert "eisv" in rec

    def test_eisv_values_correct(self, sample_db):
        records = extract_state_history(sample_db, compute_eisv=True)
        first = records[0]
        expected = anima_to_eisv(0.72, 0.85, 0.90, 0.80)
        assert first["eisv"]["E"] == pytest.approx(expected["E"])
        assert first["eisv"]["I"] == pytest.approx(expected["I"])
        assert first["eisv"]["S"] == pytest.approx(expected["S"])
        assert first["eisv"]["V"] == pytest.approx(expected["V"])

    def test_eisv_not_present_by_default(self, sample_db):
        records = extract_state_history(sample_db, compute_eisv=False)
        for rec in records:
            assert "eisv" not in rec

    def test_eisv_with_limit(self, sample_db):
        records = extract_state_history(sample_db, compute_eisv=True, limit=2)
        assert len(records) == 2
        for rec in records:
            assert "eisv" in rec
            assert set(rec["eisv"].keys()) == {"E", "I", "S", "V"}
