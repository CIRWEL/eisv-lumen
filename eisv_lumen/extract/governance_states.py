"""Extract EISV state histories from the UNITARES governance PostgreSQL database.

Reads from the ``core.agent_state`` table and returns records in the same
format as :func:`eisv_lumen.extract.lumen_states.extract_state_history` with
``compute_eisv=True``, so the output plugs directly into
:func:`eisv_lumen.extract.assembler.assemble_dataset`.

Requires ``asyncpg`` (or ``psycopg2`` as sync fallback).

Usage::

    records = extract_governance_history("postgresql://postgres:postgres@localhost:5432/governance")
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _row_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a core.agent_state row to the assembler-compatible format.

    The ``state_json`` JSONB column contains E, I, S, V directly from the
    governance ODE.  We clamp to [0, 1] to match the Lumen observation range
    used by the assembler and classifier.
    """
    state_json = row["state_json"]
    if isinstance(state_json, str):
        state_json = json.loads(state_json)

    eisv = {
        "E": _clamp(float(state_json["E"])),
        "I": _clamp(float(state_json["I"])),
        "S": _clamp(float(state_json["S"])),
        "V": _clamp(float(state_json["V"])),
    }

    recorded_at = row["recorded_at"]
    if hasattr(recorded_at, "timestamp"):
        ts = recorded_at.timestamp()
    else:
        ts = float(recorded_at)

    return {
        "timestamp": ts,
        "identity_id": row["identity_id"],
        "regime": row.get("regime"),
        "coherence": row.get("coherence"),
        "eisv": eisv,
    }


async def _extract_async(
    pg_url: str,
    agent_id: Optional[int] = None,
    since: Optional[float] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Async implementation using asyncpg."""
    import asyncpg

    conn = await asyncpg.connect(pg_url)
    try:
        conditions = []
        params: list = []
        idx = 1

        if agent_id is not None:
            conditions.append(f"identity_id = ${idx}")
            params.append(agent_id)
            idx += 1

        if since is not None:
            conditions.append(f"EXTRACT(EPOCH FROM recorded_at) > ${idx}")
            params.append(since)
            idx += 1

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"SELECT * FROM core.agent_state {where} ORDER BY recorded_at ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        rows = await conn.fetch(query, *params)
        return [_row_to_record(dict(r)) for r in rows]
    finally:
        await conn.close()


def extract_governance_history(
    pg_url: str,
    agent_id: Optional[int] = None,
    since: Optional[float] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read EISV state history from the governance PostgreSQL database.

    Parameters
    ----------
    pg_url:
        PostgreSQL connection URL, e.g.
        ``postgresql://postgres:postgres@localhost:5432/governance``.
    agent_id:
        If provided, only extract states for this ``identity_id``.
    since:
        If provided, only extract states recorded after this epoch timestamp.
    limit:
        Maximum number of rows to return.

    Returns
    -------
    list[dict]
        Records ordered by ``recorded_at`` ascending.  Each record has
        ``timestamp`` (float epoch), ``identity_id``, ``regime``,
        ``coherence``, and ``eisv`` (dict with E, I, S, V keys) — the same
        shape as ``extract_state_history(compute_eisv=True)`` output.
    """
    return asyncio.run(_extract_async(pg_url, agent_id, since, limit))
