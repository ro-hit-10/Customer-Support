"""
Centralized configuration. All tunables are read from environment variables
(loaded from a .env file in local dev) so the system can be reconfigured
without touching code — required for the "run at zero cost, one command"
constraint (swap models/providers via env only).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- LLM provider (free-tier) ---
# Groq gives a generous free tier and very low latency tool-calling models,
# which matters a lot for an *agentic* system that may take several
# LLM round trips (route -> call tool -> reflect -> answer) per question.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# --- Data ---
CSV_PATH = os.getenv("CSV_PATH", str(BASE_DIR / "data" / "support_tickets.csv"))
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "tickets.db"))
TABLE_NAME = "tickets"

# --- RAG (local, free, no external API) ---
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", str(BASE_DIR / "data" / "embeddings.npz"))

# --- Anomaly detection thresholds (used as statistical *candidates*;
# the agent/LLM still decides what is actually reported, see agent/tools.py) ---
SLA_BREACH_HOURS = float(os.getenv("SLA_BREACH_HOURS", "24"))
IQR_MULTIPLIER = float(os.getenv("IQR_MULTIPLIER", "1.5"))

# --- API ---
API_TITLE = "AI Support Ticket Intelligence — Agentic System"
CORS_ORIGINS = ["*"]
