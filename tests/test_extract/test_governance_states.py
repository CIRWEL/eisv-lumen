"""Tests for extracting EISV state histories from the governance PostgreSQL database."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eisv_lumen.extract.governance_states import (
    _clamp,
    _row_to_record,
    extract_governance_history,
)


# ---------------------------------------------------------------------------
# Sample data matching core.agent_state schema
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {
        "state_id": 218,
        "identity_id": 393,
        "recorded_at": datetime(2025, 12, 13, 4, 22, 8, tzinfo=timezone.utc),
        "entropy": 0.702,
        "integrity": 0.809,
        "stability_index": 0.182,
        "volatility": -0.003,
        "regime": "EXPLORATION",
        "coherence": 0.499,
        "state_json": '{"E": 0.702, "I": 0.809, "S": 0.182, "V": -0.003}',
    },
    {
        "state_id": 219,
        "identity_id": 281,
        "recorded_at": datetime(2025, 12, 13, 5, 0, 0, tzinfo=timezone.utc),
        "entropy": 0.704,
        "integrity": 0.818,
        "stability_index": 0.164,
        "volatility": -0.006,
        "regime": "EXPLORATION",
        "coherence": 0.497,
        "state_json": '{"E": 0.704, "I": 0.818, "S": 0.164, "V": -0.006}',
    },
    {
        "state_id": 220,
        "identity_id": 393,
        "recorded_at": datetime(2025, 12, 14, 10, 0, 0, tzinfo=timezone.utc),
        "entropy": 0.85,
        "integrity": 0.60,
        "stability_index": 0.30,
        "volatility": 0.10,
        "regime": "PRODUCTION",
        "coherence": 0.48,
        "state_json": '{"E": 0.85, "I": 0.60, "S": 0.30, "V": 0.10}',
    },
]


class TestClamp:

    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_range(self):
        assert _clamp(-0.5) == 0.0

    def test_above_range(self):
        assert _clamp(1.5) == 1.0

    def test_at_boundaries(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0


class TestRowToRecord:

    def test_basic_conversion(self):
        record = _row_to_record(SAMPLE_ROWS[0])
        assert record["identity_id"] == 393
        assert record["regime"] == "EXPLORATION"
        assert record["coherence"] == pytest.approx(0.499)

    def test_eisv_extracted_from_state_json(self):
        record = _row_to_record(SAMPLE_ROWS[0])
        eisv = record["eisv"]
        assert eisv["E"] == pytest.approx(0.702)
        assert eisv["I"] == pytest.approx(0.809)
        assert eisv["S"] == pytest.approx(0.182)
        assert eisv["V"] == pytest.approx(0.0)  # -0.003 clamped to 0

    def test_negative_v_clamped_to_zero(self):
        record = _row_to_record(SAMPLE_ROWS[0])
        assert record["eisv"]["V"] == pytest.approx(0.0)

    def test_positive_v_preserved(self):
        record = _row_to_record(SAMPLE_ROWS[2])
        assert record["eisv"]["V"] == pytest.approx(0.10)

    def test_timestamp_is_float(self):
        record = _row_to_record(SAMPLE_ROWS[0])
        assert isinstance(record["timestamp"], float)

    def test_state_json_as_dict(self):
        """state_json can be a dict (already parsed) instead of a string."""
        row = dict(SAMPLE_ROWS[0])
        row["state_json"] = {"E": 0.702, "I": 0.809, "S": 0.182, "V": -0.003}
        record = _row_to_record(row)
        assert record["eisv"]["E"] == pytest.approx(0.702)

    def test_record_has_required_keys(self):
        record = _row_to_record(SAMPLE_ROWS[0])
        assert set(record.keys()) == {
            "timestamp", "identity_id", "regime", "coherence", "eisv",
        }

    def test_eisv_values_clamped_to_unit(self):
        """Values > 1 are clamped to 1.0."""
        row = dict(SAMPLE_ROWS[0])
        row["state_json"] = '{"E": 1.5, "I": -0.2, "S": 0.5, "V": 0.3}'
        record = _row_to_record(row)
        assert record["eisv"]["E"] == 1.0
        assert record["eisv"]["I"] == 0.0


class TestExtractGovernanceHistory:

    @patch("eisv_lumen.extract.governance_states._extract_async")
    def test_returns_list_of_records(self, mock_async):
        mock_async.return_value = [_row_to_record(r) for r in SAMPLE_ROWS]
        result = extract_governance_history("postgresql://fake")
        assert len(result) == 3
        assert all("eisv" in r for r in result)

    @patch("eisv_lumen.extract.governance_states._extract_async")
    def test_agent_id_filter_forwarded(self, mock_async):
        mock_async.return_value = []
        extract_governance_history("postgresql://fake", agent_id=393)
        mock_async.assert_called_once_with("postgresql://fake", 393, None, None)

    @patch("eisv_lumen.extract.governance_states._extract_async")
    def test_since_filter_forwarded(self, mock_async):
        mock_async.return_value = []
        extract_governance_history("postgresql://fake", since=1000.0)
        mock_async.assert_called_once_with("postgresql://fake", None, 1000.0, None)

    @patch("eisv_lumen.extract.governance_states._extract_async")
    def test_limit_forwarded(self, mock_async):
        mock_async.return_value = []
        extract_governance_history("postgresql://fake", limit=10)
        mock_async.assert_called_once_with("postgresql://fake", None, None, 10)

    @patch("eisv_lumen.extract.governance_states._extract_async")
    def test_records_compatible_with_assembler(self, mock_async):
        """Output records have 'timestamp' and 'eisv' keys needed by states_to_eisv_series."""
        mock_async.return_value = [_row_to_record(r) for r in SAMPLE_ROWS]
        records = extract_governance_history("postgresql://fake")

        for rec in records:
            assert "timestamp" in rec
            assert isinstance(rec["timestamp"], float)
            assert "eisv" in rec
            eisv = rec["eisv"]
            assert set(eisv.keys()) == {"E", "I", "S", "V"}
            for dim in "EISV":
                assert 0.0 <= eisv[dim] <= 1.0
