"""Extract Lumen state_history from anima.db and map to EISV coordinates."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Union


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def anima_to_eisv(
    warmth: float,
    clarity: float,
    stability: float,
    presence: float,
) -> Dict[str, float]:
    """Map Anima state scalars to the EISV coordinate system.

    Mapping (from anima-mcp eisv_mapper.py):
        E (Energy)    = warmth
        I (Integrity) = clarity
        S (Entropy)   = 1.0 - stability
        V (Void)      = (1.0 - presence) * 0.3

    All values are clamped to [0.0, 1.0].
    """
    return {
        "E": _clamp(warmth),
        "I": _clamp(clarity),
        "S": _clamp(1.0 - stability),
        "V": _clamp((1.0 - presence) * 0.3),
    }


def extract_state_history(
    db_path: Union[str, Path],
    *,
    compute_eisv: bool = False,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Read state_history rows from *db_path* and return them as dicts.

    Parameters
    ----------
    db_path:
        Path to the anima.db SQLite file.
    compute_eisv:
        If ``True``, each returned dict includes an ``"eisv"`` key whose
        value is the EISV mapping computed via :func:`anima_to_eisv`.
    limit:
        Maximum number of rows to return.  ``None`` means all rows.

    Returns
    -------
    list[dict]
        Records ordered by ``timestamp`` ascending.  The ``sensors``
        column is parsed from its JSON text representation into a Python
        dict.
    """
    db_path = str(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM state_history ORDER BY timestamp ASC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    records: List[Dict] = []
    for row in rows:
        rec = dict(row)
        # Parse the sensors JSON text into a Python dict.
        raw_sensors = rec.get("sensors") or "{}"
        rec["sensors"] = json.loads(raw_sensors)

        if compute_eisv:
            rec["eisv"] = anima_to_eisv(
                warmth=rec["warmth"],
                clarity=rec["clarity"],
                stability=rec["stability"],
                presence=rec["presence"],
            )

        records.append(rec)

    return records
