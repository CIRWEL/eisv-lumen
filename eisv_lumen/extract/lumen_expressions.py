"""Extract primitive expression history from Lumen's anima.db."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def extract_primitive_history(
    db_path: Union[str, Path],
    *,
    min_score: Optional[float] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read primitive_history rows from *db_path* and return parsed dicts.

    Each returned dict contains:
        id              - row id
        timestamp       - ISO-8601 timestamp string
        tokens          - list of token strings (split from CSV)
        category_pattern - list of category strings (split from CSV)
        state_at_generation - dict with warmth / brightness / stability / presence
        score           - float score value
        feedback_signals - parsed JSON dict

    Parameters
    ----------
    db_path:
        Path to the SQLite database file (anima.db).
    min_score:
        If provided, only rows with ``score >= min_score`` are returned.
    limit:
        If provided, cap the number of returned rows.
    """
    query_parts: list[str] = [
        "SELECT id, timestamp, tokens, category_pattern,"
        " warmth, brightness, stability, presence,"
        " score, feedback_signals"
        " FROM primitive_history",
    ]
    params: list[Any] = []

    if min_score is not None:
        query_parts.append("WHERE score >= ?")
        params.append(min_score)

    query_parts.append("ORDER BY id")

    if limit is not None:
        query_parts.append("LIMIT ?")
        params.append(limit)

    query = " ".join(query_parts)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
    finally:
        conn.close()

    records: List[Dict[str, Any]] = []
    for row in rows:
        tokens_raw: str = row["tokens"]
        cat_raw: Optional[str] = row["category_pattern"]
        feedback_raw: Optional[str] = row["feedback_signals"]

        records.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "tokens": [t.strip() for t in tokens_raw.split(",")] if tokens_raw else [],
                "category_pattern": (
                    [c.strip() for c in cat_raw.split(",")] if cat_raw else []
                ),
                "state_at_generation": {
                    "warmth": row["warmth"],
                    "brightness": row["brightness"],
                    "stability": row["stability"],
                    "presence": row["presence"],
                },
                "score": row["score"],
                "feedback_signals": (
                    json.loads(feedback_raw) if feedback_raw else {}
                ),
            }
        )

    return records
