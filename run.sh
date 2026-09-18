#!/usr/bin/env bash
# Single-command start: pip install (first run only) + launch the API+UI.
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "No .env found — copy .env.example to .env and add your free GROQ_API_KEY first."
  exit 1
fi

python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt

echo "Starting server at http://localhost:8000  (UI at / , docs at /docs)"
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
