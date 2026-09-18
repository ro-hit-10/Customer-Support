# Support Ticket Intelligence — Agentic AI System

An agentic AI system for a customer support ticket dataset. The **LLM agent
decides which tool to use** for every request (SQL query, anomaly scan, or
semantic search) — there is no hardcoded router mapping question types to
functions. Built for zero-cost local evaluation (Groq free tier + local
embeddings), single-command startup, REST API + minimal UI.

## 1. Architecture

```
                         ┌─────────────────────────┐
   User (UI or REST) ───▶│   FastAPI  (app/api)    │
                         │  /query  /anomalies      │
                         │  /health   "/" (UI)      │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │     LangGraph ReAct Agent          │
                    │  (Groq llama-3.3-70b, free tier)   │
                    │  reason → act → observe → reason   │
                    └───────────┬─────────┬──────────────┘
                                │ decides │ which tool(s)
              ┌─────────────────┼─────────┼─────────────────┐
              ▼                 ▼                            ▼
      ┌───────────────┐ ┌──────────────────┐        ┌──────────────────┐
      │  sql_query     │ │  scan_anomalies   │        │ semantic_search   │
      │  (SQLite,      │ │  (pandas stats:   │        │ (local embeddings │
      │  read-only)    │ │  IQR + SLA rules, │        │  cosine similarity│
      │                │ │  LLM judges/      │        │  over issue text) │
      │                │ │  explains flags)  │        │                   │
      └───────┬────────┘ └─────────┬─────────┘        └─────────┬─────────┘
              ▼                    ▼                            ▼
        SQLite (tickets.db, built from CSV on first run)   embeddings.npz (cached)
```

**The agent, not application code, decides tool selection.** `app/agent/graph.py`
gives the LLM a system prompt describing the three tools and general guidance
on when each tends to help; the LLM then chooses, calls, observes results, and
can call another tool before answering (e.g. `semantic_search` to find matching
tickets, then `sql_query` to compute a stat about them). This is a LangGraph
`create_react_agent` — the standard reason/act/observe loop pattern.

### Why these components

| Choice | Reasoning |
|---|---|
| **LangGraph ReAct agent** | Real multi-step, LLM-driven tool selection with observation/retry, not a keyword router. Industry-standard, well-supported. |
| **Groq (`llama-3.3-70b-versatile`)** | Free tier, no credit card, and very fast — matters because an agentic loop can take 2-4 LLM round trips per question. |
| **SQLite for structured queries** | LLMs are excellent at writing SQL and bad at doing arithmetic themselves. SQL also composes (filters + aggregations + group-by) far better than hand-rolled pandas-from-NL. The tool only permits `SELECT` (regex-enforced, single statement) so a miswritten or adversarial query can't mutate data. |
| **Local sentence-transformers embeddings + numpy cosine search for RAG** | 500 short text rows don't need a vector DB service. A cached in-memory matrix gives genuine semantic retrieval with zero cost and zero extra services; swapping in Chroma/FAISS later is a one-file change (`app/rag/embed_store.py`). |
| **Statistics-first anomaly detection, LLM-second** | A pure LLM scan of 500 rows is unreliable and can't do rigorous math. A pure fixed-threshold script can't explain itself or use judgement. So `anomaly_engine.py` computes **candidates** (per-category IQR outliers on resolution time, SLA breaches for urgent+open tickets past 24h, low-rating agent clusters) and hands them to the agent, which reviews them, decides which are genuinely worth flagging, and writes the explanation — the "AI checks and flags it" behaviour the brief asks for, without asking the LLM to compute quartiles. |
| **FastAPI serving both REST + UI** | One process, one command (`uvicorn app.api.main:app`), satisfies "REST API AND minimal UI" without extra orchestration. |

## 2. Setup

```bash
git clone <this-repo>
cd ai-support-agent
cp .env.example .env
# edit .env and paste a free Groq key from https://console.groq.com

./run.sh
# or manually:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

Open **http://localhost:8000** for the UI, or **http://localhost:8000/docs**
for interactive API docs. On first startup the app automatically:
1. Loads `data/support_tickets.csv` into `data/tickets.db` (SQLite).
2. Builds sentence embeddings for `issue_summary` and caches them to
   `data/embeddings.npz` (subsequent restarts load the cache instantly).

No separate ingestion step is required — this is the "single command" entry point.

## 3. REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + config check (row count loaded, LLM key present) |
| `/query` | POST `{"question": "...", "session_id": "optional"}` | Natural-language Q&A; agent picks tools autonomously |
| `/anomalies` | POST | Runs the agent-reviewed anomaly scan and returns both the narrative and raw statistical candidates |

## 4. Example queries & (representative) outputs

```
POST /query {"question": "How many critical tickets are unresolved?"}
→ tool_calls: [sql_query]
→ "There are 19 Critical-priority tickets that are still Open or Escalated."

POST /query {"question": "Which agent has the lowest average customer rating?"}
→ tool_calls: [sql_query]
→ "AGT-07 has the lowest average customer rating at 2.8 across 34 rated tickets."

POST /anomalies
→ tool_calls: [scan_anomalies]
→ "3 items worth flagging: TKT-233 (Billing, High priority) has been open
   ~988 days past SLA — looks like a stale/orphaned ticket rather than a
   genuine active breach, worth auditing. TKT-108 took 119.7 hrs to resolve
   vs a ~12 hr median for General tickets and got a 2-star rating — likely
   a real service failure. No agent currently shows a low-rating cluster."

POST /query {"question": "Find tickets similar to a login failure"}
→ tool_calls: [semantic_search]
→ Returns top-5 tickets by semantic similarity to "login failure",
  even ones that don't contain the literal words "login" or "failure".
```

Exact numbers/wording will vary run to run since the LLM writes the final
prose — the tool calls and underlying data are deterministic.

## 5. Optimizations

A few deliberate performance/cost/architecture choices worth knowing about
for the walkthrough:

| Optimization | Why |
|---|---|
| **In-process TTL cache on `load_dataframe()`** (`app/utils/cache.py`) | The 500-row table doesn't change between requests; repeated SQLite round-trips were pure waste. Keyed on the DB file's mtime, so a re-ingest busts it automatically — no stale data risk. |
| **15s cache on anomaly candidates** | Dedupes bursts (e.g. the UI's chat tab and anomaly tab both scanning within seconds) without the SLA "age" figures ever going meaningfully stale. |
| **Fixed a duplicate-compute bug in `/anomalies`** | The endpoint used to run the full IQR/SLA statistics pass *twice per request* — once inside the agent's `scan_anomalies` tool call, once again directly for the "raw candidates" field. Both call sites now go through one shared, cached function (`tools.get_anomaly_candidates`), so the endpoint's raw data is guaranteed to be exactly what the agent reasoned over, computed once. |
| **120s cache on final agent answers**, keyed by `(question, session_id)` | The highest-leverage cache in the system — a hit skips an entire multi-step LLM tool-calling loop. Saves both latency and free-tier quota on repeated/duplicate questions (a real risk with a chat UI re-rendering or a user double-clicking Ask). |
| **Async end-to-end (`agent.ainvoke`, `async def` routes)** | The dominant cost per request is network I/O waiting on Groq, not local CPU — async lets FastAPI serve other requests while one is in flight, a real concurrency win under multiple simultaneous users. |
| **Retry with jittered exponential backoff on the LLM call** | Free-tier APIs rate-limit. A transient 429/503 now gets retried (1s → ~2s → ~4s) instead of surfacing as a hard failure to the user. Non-transient errors (bad request, auth) still fail fast. |
| **Embedding model warm-loaded at startup**, not lazily on first query | Previously the ~1-2s sentence-transformers model load only happened lazily on cache miss; now it's forced at server startup regardless, so the first real `semantic_search` call never pays that latency spike. |
| **SQL tool restricted to single read-only `SELECT`, 200-row cap** | Keeps LLM-generated queries safe and keeps tool output small enough to stay cheap in the LLM's context window. |

## 6. Known limitations

- **Statistical anomaly thresholds are heuristics** (IQR ×1.5, 24h SLA), not
  domain-calibrated; tune via `.env` (`IQR_MULTIPLIER`, `SLA_BREACH_HOURS`).
- **No auth** on the API — fine for local evaluation, not production-ready.
- **In-memory conversation checkpointing** (LangGraph `MemorySaver`) is lost
  on restart; swap for a persistent checkpointer for multi-session use.
- **Groq free tier has rate limits**; heavy concurrent load will 429.
- **Embedding cache invalidation** is by exact ticket-ID-list match; a full
  CSV replace triggers re-embedding (a few seconds for 500 rows, one-time cost).
- The dataset's `created_at` values are all in early 2024, so "SLA breach age"
  in the demo output looks unrealistically large (it's computed against
  *today's* date) — the logic itself is correct and would look normal on a
  live ticket stream.

## 7. What I'd improve with more time

- Replace the numpy cosine-search RAG with a proper vector index (Chroma/FAISS)
  once corpus size grows past a few thousand rows.
- Add a lightweight eval harness (golden Q&A pairs) to catch SQL-generation
  regressions when swapping LLM models.
- Persist the LangGraph checkpointer to SQLite for durable multi-turn sessions.
- Add authentication + per-tenant data isolation for a real multi-team deployment.

## 8. Project structure

```
app/
  config.py                 # env-driven settings
  data_layer/db.py          # CSV → SQLite ingestion + safe read-only SQL tool
  rag/embed_store.py        # local embeddings + cosine-similarity search
  agent/anomaly_engine.py   # statistical candidate generation (IQR/SLA)
  agent/tools.py            # LangChain tool wrappers the agent chooses between
  agent/graph.py            # LangGraph ReAct agent (the orchestration brain)
  api/main.py                # FastAPI app: 3 endpoints + serves the UI
  api/schemas.py            # request/response models
  static/index.html         # minimal single-file chat + anomaly-report UI
data/support_tickets.csv    # provided dataset
requirements.txt
run.sh                       # single-command startup
```
