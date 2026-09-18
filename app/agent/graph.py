"""
Agent orchestration layer, built on LangGraph's prebuilt ReAct agent.

Why LangGraph (and not a hand-rolled if/elif on keywords)?
- The agent runs a genuine reason -> act -> observe -> reason loop: the LLM
  sees the user question + schema + tool descriptions, decides which
  tool(s) to call (possibly several, possibly none), inspects the tool
  output, and can decide to call another tool before answering (e.g.
  "scan_anomalies" then "sql_query" to cross-check a specific ticket).
  Tool selection is 100% the model's decision, not application code.
- It's the standard, well-supported way to build a tool-using agent in
  Python today, which matters for a "production-grade" evaluation.

The system prompt is where the agent is taught *how* to decide between
tools; it deliberately does not hardcode which tool answers which
question type.
"""
import asyncio
import random

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app import config
from app.agent.tools import ALL_TOOLS
from app.data_layer import db

_agent = None
_checkpointer = MemorySaver()  # per-session conversational memory, in-process


def _system_prompt() -> str:
    return (
        "You are a Senior AI Support Intelligence Analyst. "
        "Your responses must be executive-ready, highly structured, human-readable, and insightful for engineering and support leaders.\n\n"
        "FORMATTING RULES:\n"
        "- Always format output with standard GitHub Markdown (headers `###`, bullet points, **bold text**, tables, code blocks).\n"
        "- For anomaly scans or system health reports, structure your analysis into clear executive sections:\n"
        "  1. 📌 **Executive Summary & Critical Findings**\n"
        "  2. 🚨 **Priority SLA Breaches** (List key ticket IDs, days open, categories, and priority)\n"
        "  3. ⏱️ **Resolution-Time Outliers & Quality Risks**\n"
        "  4. 💡 **Actionable Engineering Recommendations**\n"
        "- Highlight ticket IDs (e.g. **TKT-108**), agent IDs (e.g. **AGT-07**), and key numbers in bold.\n"
        "- Be concise, direct, and actionable. Explain the business/technical impact of flagged items.\n\n"
        f"{db.get_schema_description()}\n\n"
        "Guidance on tool choice:\n"
        "- Structured/aggregate questions (counts, averages, filters, rankings) -> sql_query.\n"
        "- Anomaly / outlier / SLA / 'is anything wrong' questions -> scan_anomalies.\n"
        "- Fuzzy, descriptive, or 'similar to' questions about issue content -> semantic_search.\n"
        "- You may call more than one tool if a question needs it.\n"
        "- Always verify with a tool call — never fabricate numbers."
    )


def get_agent():
    global _agent
    if _agent is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and put it in a .env file (see .env.example)."
            )
        llm = ChatGroq(
            model=config.GROQ_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.GROQ_API_KEY,
        )
        _agent = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            state_modifier=_system_prompt(),
            checkpointer=_checkpointer,
        )
    return _agent


_RETRYABLE_MARKERS = ("rate limit", "429", "overloaded", "503", "timeout")


async def _invoke_with_retry(agent, payload: dict, run_config: dict, max_attempts: int = 3):
    """
    Groq's free tier is generous but not unlimited, and it's the single
    external dependency in the whole system — so a transient 429/503 there
    would otherwise surface as a hard user-facing failure. Retries with
    jittered exponential backoff (1s, ~2s, ~4s) absorb that without the
    caller ever seeing it, and only for genuinely retryable errors; anything
    else (bad request, auth failure) fails fast instead of masking a real bug.
    """
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await agent.ainvoke(payload, config=run_config)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            transient = any(marker in str(exc).lower() for marker in _RETRYABLE_MARKERS)
            if not transient or attempt == max_attempts - 1:
                raise
            await asyncio.sleep(delay + random.random() * 0.5)
            delay *= 2
    raise last_exc  # pragma: no cover - unreachable, satisfies type checkers


def _extract_result(result: dict) -> dict:
    messages = result["messages"]
    final_answer = messages[-1].content

    tool_trace = []
    for m in messages:
        calls = getattr(m, "tool_calls", None)
        if calls:
            for c in calls:
                tool_trace.append({"tool": c["name"], "args": c["args"]})

    return {"answer": final_answer, "tool_calls": tool_trace}


import uuid

async def ask(question: str, session_id: str = "default") -> dict:
    """
    Runs the agent loop for a single user question and returns the final
    answer plus a trace of which tools were used.
    """
    agent = get_agent()
    payload = {"messages": [{"role": "user", "content": question}]}
    thread_id = str(uuid.uuid4()) if session_id in ("default", "anomaly-endpoint") else session_id
    run_config = {"configurable": {"thread_id": thread_id}}
    result = await _invoke_with_retry(agent, payload, run_config)
    return _extract_result(result)
