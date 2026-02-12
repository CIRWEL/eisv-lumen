"""Tests for governance trajectory and dialectic data extraction."""

import pytest

from eisv_lumen.extract.governance_data import (
    extract_agent_trajectories,
    extract_dialectic_sessions,
    parse_agent_update,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_UPDATE = {
    "agent_id": "opus_test_20260101",
    "timestamp": "2026-01-01T12:00:00",
    "eisv": {"E": 0.7, "I": 0.8, "S": 0.2, "V": 0.05},
    "coherence": 0.49,
    "verdict": "proceed",
    "margin": "comfortable",
    "complexity": 0.5,
    "confidence": 0.8,
    "ethical_drift": [0.02, -0.01, 0.03],
}

SAMPLE_DIALECTIC = {
    "session_id": "abc123",
    "paused_agent_id": "agent_a",
    "reviewer_agent_id": "agent_b",
    "topic": "Design review",
    "thesis": {"root_cause": "Misalignment", "conditions": ["c1"], "reasoning": "r1"},
    "antithesis": {"concerns": ["concern1"], "reasoning": "r2"},
    "synthesis": {"conditions": ["c2"], "reasoning": "r3", "agrees": True},
    "status": "converged",
}


def _make_updates(agent_id, count, base_hour=12):
    """Helper: create *count* updates for a single agent."""
    updates = []
    for i in range(count):
        updates.append(
            {
                "agent_id": agent_id,
                "timestamp": f"2026-01-01T{base_hour + i:02d}:00:00",
                "eisv": {"E": 0.5 + i * 0.01, "I": 0.6, "S": 0.3, "V": 0.1},
                "coherence": 0.5,
                "verdict": "proceed",
                "margin": "comfortable",
                "complexity": 0.4,
                "confidence": 0.7,
                "ethical_drift": [0.0, 0.0, 0.0],
            }
        )
    return updates


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseAgentUpdate:
    def test_parse_agent_update(self):
        """Parse raw update into standardized record; verify provenance."""
        record = parse_agent_update(SAMPLE_UPDATE)

        assert record["agent_id"] == "opus_test_20260101"
        assert record["timestamp"] == "2026-01-01T12:00:00"
        assert record["eisv"] == {"E": 0.7, "I": 0.8, "S": 0.2, "V": 0.05}
        assert record["coherence"] == 0.49
        assert record["verdict"] == "proceed"
        assert record["margin"] == "comfortable"
        assert record["complexity"] == 0.5
        assert record["confidence"] == 0.8
        assert record["ethical_drift"] == [0.02, -0.01, 0.03]
        assert record["provenance"] == "governance_multi_agent"

    def test_parse_agent_update_handles_missing_fields(self):
        """Missing optional fields get sensible defaults."""
        minimal = {
            "agent_id": "minimal_agent",
            "timestamp": "2026-06-01T00:00:00",
            "eisv": {"E": 0.5, "I": 0.5, "S": 0.5, "V": 0.5},
        }
        record = parse_agent_update(minimal)

        assert record["agent_id"] == "minimal_agent"
        assert record["provenance"] == "governance_multi_agent"
        # Defaults for missing optional fields
        assert record["coherence"] is None
        assert record["verdict"] is None
        assert record["margin"] is None
        assert record["complexity"] is None
        assert record["confidence"] is None
        assert record["ethical_drift"] is None


class TestExtractAgentTrajectories:
    def test_extract_agent_trajectories(self):
        """Group updates by agent, filter by min_updates, verify timestamp sort."""
        # agent_a gets 4 updates (out of order), agent_b gets 3
        updates_a = _make_updates("agent_a", 4, base_hour=10)
        updates_b = _make_updates("agent_b", 3, base_hour=14)
        # Shuffle so input is not pre-sorted
        raw = [updates_a[2], updates_b[0], updates_a[0], updates_b[2], updates_a[3], updates_b[1], updates_a[1]]

        trajectories = extract_agent_trajectories(raw, min_updates=3)

        assert len(trajectories) == 2
        agent_ids = {t["agent_id"] for t in trajectories}
        assert agent_ids == {"agent_a", "agent_b"}

        for traj in trajectories:
            timestamps = [u["timestamp"] for u in traj["updates"]]
            assert timestamps == sorted(timestamps), "Updates should be sorted by timestamp"

    def test_extract_agent_trajectories_filters_sparse_agents(self):
        """Agents with fewer than min_updates are excluded."""
        updates_a = _make_updates("agent_a", 5)
        updates_b = _make_updates("agent_b", 2)  # below threshold
        raw = updates_a + updates_b

        trajectories = extract_agent_trajectories(raw, min_updates=3)

        assert len(trajectories) == 1
        assert trajectories[0]["agent_id"] == "agent_a"
        assert len(trajectories[0]["updates"]) == 5


class TestExtractDialecticSessions:
    def test_extract_dialectic_sessions(self):
        """Parse dialectic sessions into standardized records."""
        results = extract_dialectic_sessions([SAMPLE_DIALECTIC])

        assert len(results) == 1
        sess = results[0]
        assert sess["session_id"] == "abc123"
        assert sess["paused_agent_id"] == "agent_a"
        assert sess["reviewer_agent_id"] == "agent_b"
        assert sess["topic"] == "Design review"
        assert sess["thesis"]["root_cause"] == "Misalignment"
        assert sess["antithesis"]["concerns"] == ["concern1"]
        assert sess["synthesis"]["agrees"] is True
        assert sess["status"] == "converged"
        assert sess["provenance"] == "governance_multi_agent"
