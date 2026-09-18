from typing import Any, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    answer: str
    tool_calls: list


class AnomalyResponse(BaseModel):
    summary: str
    candidates: dict


class HealthResponse(BaseModel):
    status: str
    rows_loaded: int
    llm_configured: bool
