"""
The agent's tool belt. Every tool below is exposed to the LLM with a
description; the LLM (not any if/else in this codebase) decides which
tool(s) to call, in what order, and with what arguments, based on the
user's question. This is what makes the system "agentic" rather than a
router with hardcoded intents.

Tools:
  sql_query          -> structured / aggregate questions ("how many", "average", "which agent")
  scan_anomalies      -> anomaly detection, called only when the agent judges it relevant
  semantic_search     -> fuzzy / free-text questions ("tickets like X", "complaints about Y")
"""
from langchain_core.tools import tool

from app.data_layer import db
from app.rag import embed_store
from app.agent import anomaly_engine
from app.utils.cache import TTLCache

# Short-lived, tool-scoped caches (see app/utils/cache.py for rationale).
# TTLs are deliberately short: fresh enough that a live ticket stream would
# never feel stale, long enough to absorb the redundant calls that a single
# multi-step agent turn (or two tabs in the UI) commonly produces.
_sql_cache = TTLCache()
_anomaly_cache = TTLCache()
_search_cache = TTLCache()


@tool
def sql_query(sql: str) -> dict:
    """
    Run a read-only SQL SELECT query against the `tickets` table to answer
    questions involving counts, averages, filters, group-bys, or ranking
    (e.g. "how many critical tickets are unresolved", "which agent has the
    lowest average rating", "tickets older than 24h"). Only SELECT is
    allowed. Always reference actual column names from the schema you were
    given. If the query errors, read the error and try a corrected query.
    """
    def _run() -> dict:
        try:
            return db.run_read_only_query(sql)
        except Exception as exc:  # noqa: BLE001 - surfaced back to the LLM to self-correct
            return {"error": str(exc)}

    # Cache identical SQL text for a few seconds — cheap dedupe for the
    # common case of the agent re-checking its own prior result.
    return _sql_cache.get_or_set(sql.strip(), ttl_seconds=10, factory=_run)


def get_anomaly_candidates() -> dict:
    """
    Plain (non-tool) entry point for the statistical anomaly scan, shared
    by the `scan_anomalies` tool below AND the `/anomalies` REST endpoint
    (see api/main.py). Previously the endpoint recomputed this
    independently of whatever the agent's tool call had already produced —
    same statistics, run twice, on every request. Routing both call sites
    through this one cached function removes that duplicate work and
    guarantees the endpoint's "raw candidates" field is exactly what the
    agent actually reasoned over, not a second, potentially time-shifted
    computation of it.
    """
    def _scan() -> dict:
        df = db.load_dataframe()
        return anomaly_engine.find_candidates(df)

    # 15s TTL: cheap enough to hit on every request, short enough that the
    # SLA-breach "age" figures never feel meaningfully out of date.
    return _anomaly_cache.get_or_set("candidates", ttl_seconds=15, factory=_scan)


@tool
def scan_anomalies() -> dict:
    """
    Run statistical anomaly detection over the full ticket dataset and
    return candidate anomalies in three categories: SLA breaches (urgent
    tickets open too long), resolution-time outliers (abnormally slow
    resolutions relative to their category's normal range), and agents
    with clusters of low customer ratings. Call this when the user asks
    about anomalies, unusual tickets, outliers, or system health — then
    review the candidates yourself and decide which are genuinely worth
    flagging and why, rather than repeating the raw numbers verbatim.
    """
    return get_anomaly_candidates()


@tool
def semantic_search(query: str, k: int = 5) -> list:
    """
    Semantic (meaning-based) search over ticket issue descriptions. Use
    this for fuzzy questions that don't map to a clean SQL filter, e.g.
    "find tickets about login problems", "anything similar to a refund
    complaint", or when the user references an issue by description
    rather than by exact category/keyword.
    """
    # The corpus is static between ingests, so identical queries can be
    # cached generously (5 min) without any risk of staleness.
    key = (query.strip().lower(), k)
    return _search_cache.get_or_set(key, ttl_seconds=300, factory=lambda: embed_store.search(query, k=k))


ALL_TOOLS = [sql_query, scan_anomalies, semantic_search]
