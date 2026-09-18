"""
Data layer: loads the raw CSV into a local SQLite database once, and
exposes a small, safe query surface for the agent's SQL tool.

Why SQLite (and not "just pandas in memory")?
- Lets the agent write real SQL (aggregations, filters, group-bys) which
  LLMs are extremely well trained on -> much more reliable than asking an
  LLM to write pandas code and eval() it (which is also a code-injection
  risk we specifically want to avoid).
- Zero external services, zero cost, single file, trivial to inspect.
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

from app import config
from app.utils.cache import TTLCache

_df_cache = TTLCache()

_READ_ONLY_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|PRAGMA|REPLACE|CREATE)\b",
    re.IGNORECASE,
)


def ensure_db() -> None:
    """Idempotently (re)builds the SQLite DB from the CSV if missing or stale."""
    db_file = Path(config.DB_PATH)
    csv_file = Path(config.CSV_PATH)

    if not csv_file.exists():
        raise FileNotFoundError(f"Ticket CSV not found at {csv_file}")

    needs_rebuild = (
        not db_file.exists()
        or db_file.stat().st_mtime < csv_file.stat().st_mtime
    )
    if not needs_rebuild:
        return

    df = pd.read_csv(csv_file)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        df.to_sql(config.TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_status ON {config.TABLE_NAME}(status)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_priority ON {config.TABLE_NAME}(priority)"
        )
        conn.commit()
    finally:
        conn.close()


def get_schema_description() -> str:
    """Human-readable schema string injected into the agent's system prompt."""
    return (
        f"Table `{config.TABLE_NAME}` columns:\n"
        "- ticket_id (TEXT, e.g. 'TKT-001')\n"
        "- created_at (DATETIME, 'YYYY-MM-DD HH:MM:SS')\n"
        "- category (TEXT: Billing | Technical | General)\n"
        "- priority (TEXT: Low | Medium | High | Critical)\n"
        "- status (TEXT: Open | Resolved | Escalated)\n"
        "- response_time_hrs (FLOAT, hours to first response)\n"
        "- resolution_time_hrs (FLOAT, hours to resolution; NULL if unresolved)\n"
        "- agent_id (TEXT, e.g. 'AGT-04')\n"
        "- customer_rating (INTEGER 1-5; NULL if unresolved)\n"
        "- issue_summary (TEXT, free-text description)\n"
        "SQLite dialect. Use julianday('now') for 'now' in date math."
    )


def run_read_only_query(sql: str, row_limit: int = 200):
    """
    Executes a single read-only SELECT statement. Raises ValueError for
    anything that isn't a plain SELECT, so a misbehaving/prompt-injected
    LLM call can never mutate or exfiltrate beyond the ticket table.
    """
    sql_stripped = sql.strip().rstrip(";")
    if not _READ_ONLY_PATTERN.match(sql_stripped):
        raise ValueError("Only SELECT statements are permitted.")
    if _FORBIDDEN_KEYWORDS.search(sql_stripped):
        raise ValueError("Query contains a forbidden keyword.")
    if ";" in sql_stripped:
        raise ValueError("Multiple statements are not permitted.")

    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql_stripped)
        rows = cur.fetchmany(row_limit)
        columns = [d[0] for d in cur.description] if cur.description else []
        return {
            "columns": columns,
            "rows": [dict(r) for r in rows],
            "row_count": len(rows),
            "truncated": len(rows) == row_limit,
        }
    finally:
        conn.close()


def load_dataframe() -> pd.DataFrame:
    """
    Used by the anomaly-detection tool, which needs vectorized stats.
    Cached: keyed on the DB file's mtime, so a fresh ingest (new CSV)
    naturally busts the cache with no manual invalidation needed, while
    repeated calls between ingests (e.g. several tool calls in one agent
    turn, or back-to-back anomaly scans) skip the SQLite round trip.
    """
    mtime = Path(config.DB_PATH).stat().st_mtime

    def _load() -> pd.DataFrame:
        conn = sqlite3.connect(config.DB_PATH)
        try:
            return pd.read_sql(f"SELECT * FROM {config.TABLE_NAME}", conn, parse_dates=["created_at"])
        finally:
            conn.close()

    return _df_cache.get_or_set(("dataframe", mtime), ttl_seconds=3600, factory=_load)
