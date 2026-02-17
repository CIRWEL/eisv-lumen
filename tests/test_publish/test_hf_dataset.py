"""Tests for HuggingFace dataset publisher."""

import json
import pytest
from eisv_lumen.publish.hf_dataset import (
    trajectories_to_hf_format,
    generate_dataset_card,
)


def _make_records(n=3):
    """Create fake assembled trajectory records."""
    records = []
    for i in range(n):
        records.append({
            "shape": "settled_presence",
            "states": [{"t": float(j), "E": 0.7, "I": 0.6, "S": 0.2, "V": 0.03} for j in range(5)],
            "derivatives": [{"t": float(j+1), "dE": 0.0, "dI": 0.0, "dS": 0.0, "dV": 0.0} for j in range(4)],
            "second_derivatives": [],
            "t_start": 0.0,
            "t_end": 4.0,
            "provenance": "lumen_real",
            "expressions": [
                {"tokens": ["~stillness~", "~holding~"], "score": 0.8},
            ] if i % 2 == 0 else [],
        })
    return records


class TestTrajectoriesToHfFormat:
    def test_returns_dict_with_expected_columns(self):
        records = _make_records(3)
        result = trajectories_to_hf_format(records)
        expected_keys = {"shape", "eisv_states", "derivatives", "t_start", "t_end", "provenance", "tokens", "n_expressions"}
        assert set(result.keys()) == expected_keys

    def test_column_lengths_match(self):
        records = _make_records(5)
        result = trajectories_to_hf_format(records)
        for key, values in result.items():
            assert len(values) == 5, f"Column {key} has wrong length"

    def test_shape_column_is_strings(self):
        records = _make_records(3)
        result = trajectories_to_hf_format(records)
        assert all(isinstance(s, str) for s in result["shape"])

    def test_eisv_states_is_json(self):
        records = _make_records(2)
        result = trajectories_to_hf_format(records)
        parsed = json.loads(result["eisv_states"][0])
        assert isinstance(parsed, list)
        assert "E" in parsed[0]

    def test_tokens_is_json(self):
        records = _make_records(2)
        result = trajectories_to_hf_format(records)
        parsed = json.loads(result["tokens"][0])
        assert isinstance(parsed, list)

    def test_n_expressions_count(self):
        records = _make_records(3)
        result = trajectories_to_hf_format(records)
        # Records 0 and 2 have 1 expression, record 1 has 0
        assert result["n_expressions"][0] == 1
        assert result["n_expressions"][1] == 0
        assert result["n_expressions"][2] == 1

    def test_empty_records(self):
        result = trajectories_to_hf_format([])
        for key, values in result.items():
            assert len(values) == 0


class TestGenerateDatasetCard:
    def test_contains_dataset_name(self):
        card = generate_dataset_card("hikewa/unitares-eisv-trajectories", n_records=100)
        assert "unitares-eisv-trajectories" in card

    def test_contains_eisv_explanation(self):
        card = generate_dataset_card()
        assert "EISV" in card
        # Verify correct dimension names (not the old wrong ones)
        assert "Energy" in card
        assert "Information Integrity" in card
        assert "Entropy" in card
        assert "Void" in card
        # Ensure old wrong names are NOT present
        assert "Sigma" not in card
        assert "Valence" not in card

    def test_contains_license(self):
        card = generate_dataset_card()
        assert "Apache" in card or "apache" in card

    def test_contains_shape_counts(self):
        counts = {"settled_presence": 50, "rising_entropy": 30}
        card = generate_dataset_card(shape_counts=counts)
        assert "settled_presence" in card
        assert "50" in card

    def test_contains_record_count(self):
        card = generate_dataset_card(n_records=42)
        assert "42" in card
