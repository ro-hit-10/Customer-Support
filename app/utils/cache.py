"""
A deliberately tiny in-process TTL cache — no Redis, no extra service,
because the deployment target is "one command, zero cost, single
evaluator machine". It's used in three places, each for a different
reason:

1. `data_layer.db.load_dataframe()` — the 500-row CSV doesn't change
   between requests; re-reading + re-parsing it from SQLite on every
   tool call is pure waste.
2. `agent.tools.scan_anomalies` — dedupes bursts of anomaly scans (e.g.
   the UI's "Ask" tab and "Anomaly report" tab both triggering a scan
   within the same few seconds) without going stale for more than a
   few seconds.
3. `api.main` — caches the *final* agent answer per (question, session)
   for a short window. This is the highest-value cache: it can save a
   full multi-step LLM round trip (real latency and, on a rate-limited
   free tier, real quota) when a user re-asks or the UI re-renders the
   same question.

Not a correctness mechanism — every cached value is cheap to
recompute and short-lived, so a cache miss is always safe.
"""
import time
from typing import Any, Awaitable, Callable, Hashable


class TTLCache:
    def __init__(self) -> None:
        self._store: dict[Hashable, tuple[float, Any]] = {}

    def get_or_set(self, key: Hashable, ttl_seconds: float, factory: Callable[[], Any]) -> Any:
        now = time.time()
        cached = self._store.get(key)
        if cached is not None and (now - cached[0]) < ttl_seconds:
            return cached[1]
        value = factory()
        self._store[key] = (now, value)
        return value

    async def aget_or_set(
        self, key: Hashable, ttl_seconds: float, factory: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Async counterpart for factories that must be awaited (LLM calls)."""
        now = time.time()
        cached = self._store.get(key)
        if cached is not None and (now - cached[0]) < ttl_seconds:
            return cached[1]
        value = await factory()
        self._store[key] = (now, value)
        return value

    def invalidate(self, key: Hashable) -> None:
        self._store.pop(key, None)
