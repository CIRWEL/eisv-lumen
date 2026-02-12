"""Extract governance trajectory and dialectic data from unitares-governance."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

_PROVENANCE = "governance_multi_agent"

_OPTIONAL_UPDATE_FIELDS = (
    "coherence",
    "verdict",
    "margin",
    "complexity",
    "confidence",
    "ethical_drift",
)


def parse_agent_update(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a raw governance update into a standardised trajectory record.

    Required keys in *raw*: agent_id, timestamp, eisv.
    Optional keys (default to None): coherence, verdict, margin, complexity,
    confidence, ethical_drift.

    The returned record always carries provenance='governance_multi_agent'.
    """
    record: Dict[str, Any] = {
        "agent_id": raw["agent_id"],
        "timestamp": raw["timestamp"],
        "eisv": raw["eisv"],
    }
    for field in _OPTIONAL_UPDATE_FIELDS:
        record[field] = raw.get(field, None)
    record["provenance"] = _PROVENANCE
    return record


def extract_agent_trajectories(
    updates: List[Dict[str, Any]],
    min_updates: int = 3,
) -> List[Dict[str, Any]]:
    """Group parsed updates into per-agent trajectories.

    Only agents with at least *min_updates* updates are included.  Updates
    within each trajectory are sorted by timestamp (ascending).
    """
    # Parse every raw update first.
    parsed = [parse_agent_update(u) for u in updates]

    # Group by agent.
    by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in parsed:
        by_agent[record["agent_id"]].append(record)

    trajectories: List[Dict[str, Any]] = []
    for agent_id, agent_updates in sorted(by_agent.items()):
        if len(agent_updates) < min_updates:
            continue
        agent_updates.sort(key=lambda r: r["timestamp"])
        trajectories.append(
            {
                "agent_id": agent_id,
                "updates": agent_updates,
                "num_updates": len(agent_updates),
                "provenance": _PROVENANCE,
            }
        )
    return trajectories


def extract_dialectic_sessions(
    sessions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract dialectic sessions (thesis / antithesis / synthesis).

    Each returned record mirrors the source structure and is tagged with
    provenance='governance_multi_agent'.
    """
    results: List[Dict[str, Any]] = []
    for sess in sessions:
        results.append(
            {
                "session_id": sess["session_id"],
                "paused_agent_id": sess["paused_agent_id"],
                "reviewer_agent_id": sess["reviewer_agent_id"],
                "topic": sess.get("topic"),
                "thesis": sess["thesis"],
                "antithesis": sess["antithesis"],
                "synthesis": sess["synthesis"],
                "status": sess.get("status"),
                "provenance": _PROVENANCE,
            }
        )
    return results
