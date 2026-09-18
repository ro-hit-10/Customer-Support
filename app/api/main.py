"""
FastAPI app. Serves both the REST API (as required) and the minimal UI
(also required) from a single process, so the whole system starts with a
single command: `uvicorn app.api.main:app`.

On startup it idempotently ingests the CSV into SQLite and builds/loads
the local embedding index — the evaluator does not need a separate
"ingest" step.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.data_layer import db
from app.rag import embed_store
from app.agent import graph as agent_graph
from app.agent.tools import get_anomaly_candidates
from app.api.schemas import QueryRequest, QueryResponse, AnomalyResponse, HealthResponse
from app.utils.cache import TTLCache

# Caches the final agent answer per (question, session) for a short window.
# This is the highest-leverage cache in the system: a hit skips an entire
# multi-step LLM tool-calling loop, which is both the slowest part of a
# request and the part that consumes free-tier quota.
_answer_cache = TTLCache()
_ANSWER_CACHE_TTL_SECONDS = 120

app = FastAPI(title=config.API_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.on_event("startup")
def startup() -> None:
    db.ensure_db()
    df = db.load_dataframe()
    # warm=True forces the embedding model into memory now, even on a cache
    # hit, so the first real /query with a semantic_search call doesn't
    # eat a one-off model-load latency spike (previously only paid when
    # the cache actually missed).
    embed_store.build_or_load_index(df, warm=True)


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        df = db.load_dataframe()
        row_count = len(df)
    except Exception:
        row_count = 0
    return HealthResponse(
        status="ok" if row_count else "degraded",
        rows_loaded=row_count,
        llm_configured=bool(config.GROQ_API_KEY),
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Natural-language Q&A over the ticket dataset, handled entirely by the agent."""
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    session_id = req.session_id or "default"
    cache_key = (question.lower(), session_id)

    async def _run() -> dict:
        try:
            return await agent_graph.ask(question, session_id=session_id)
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    # NB: a raised HTTPException inside the cached factory is not itself
    # cached (get_or_set only stores the return value), so error responses
    # are always retried fresh rather than getting stuck in the cache.
    result = await _answer_cache.aget_or_set(
        cache_key, ttl_seconds=_ANSWER_CACHE_TTL_SECONDS, factory=_run
    )
    return QueryResponse(**result)


@app.post("/anomalies", response_model=AnomalyResponse)
async def anomalies():
    """
    Dedicated anomaly-detection endpoint. Runs the same agent-driven flagging
    pipeline as asking the agent "are there any anomalies?" via /query, but
    as a direct call for dashboards/automation that don't need free-text NL.

    The raw `candidates` field is read from the same cached statistical
    layer the agent's scan_anomalies tool call just populated (see
    agent/tools.py::get_anomaly_candidates) — it is not recomputed
    independently, so this endpoint never does the IQR/SLA pass twice.
    """
    try:
        result = await agent_graph.ask(
            "Scan for anomalies right now and give me a concise flagged report "
            "with your reasoning for each item you consider significant.",
            session_id="anomaly-endpoint",
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    raw_candidates = get_anomaly_candidates()
    return AnomalyResponse(summary=result["answer"], candidates=raw_candidates)
