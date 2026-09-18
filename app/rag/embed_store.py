"""
Lightweight local RAG layer.

Why not a vector DB service (Pinecone/Weaviate/etc.)? The constraint is
"zero cost, single command, evaluator runs locally" and the corpus is only
500 short free-text rows. A cached in-memory numpy matrix + cosine
similarity gives real semantic retrieval with zero extra services, zero
API cost, and sub-millisecond search at this scale — the right tool for
this data size. It is isolated behind `search()` so swapping in
Chroma/FAISS later is a one-file change.

Embeddings are computed once with a small local sentence-transformers
model (all-MiniLM-L6-v2, ~80MB, runs on CPU) and cached to disk next to
the CSV, so restarts don't re-embed.
"""
import numpy as np
import pandas as pd

from app import config

_model = None
_embeddings: np.ndarray | None = None
_ticket_ids: list[str] | None = None
_summaries: list[str] | None = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL_NAME)
    return _model


def build_or_load_index(df: pd.DataFrame, warm: bool = False) -> None:
    """
    Idempotently builds (or loads a cached) embedding matrix for issue_summary.

    warm: if True, forces the sentence-transformers model into memory even
    on a cache hit. Called with warm=True from the FastAPI startup event so
    the (one-off, ~1-2s) model load happens while the server is starting up
    rather than stalling the first user's semantic_search call.
    """
    global _embeddings, _ticket_ids, _summaries
    from pathlib import Path

    cache_file = Path(config.EMBED_CACHE_PATH)
    _ticket_ids = df["ticket_id"].tolist()
    _summaries = df["issue_summary"].fillna("").tolist()

    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        if list(cached["ticket_ids"]) == _ticket_ids:
            _embeddings = cached["embeddings"]
            if warm:
                _get_model()  # pay the model-load cost now, not on first query
            return

    model = _get_model()
    _embeddings = model.encode(
        _summaries, normalize_embeddings=True, show_progress_bar=False
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file, embeddings=_embeddings, ticket_ids=np.array(_ticket_ids, dtype=object)
    )


def search(query: str, k: int = 5) -> list[dict]:
    """Cosine-similarity top-k search over issue_summary embeddings."""
    if _embeddings is None:
        raise RuntimeError("RAG index not built yet — call build_or_load_index() first.")

    model = _get_model()
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = _embeddings @ q_vec  # cosine similarity (vectors are normalized)
    top_idx = np.argsort(-scores)[:k]

    return [
        {
            "ticket_id": _ticket_ids[i],
            "issue_summary": _summaries[i],
            "similarity": float(scores[i]),
        }
        for i in top_idx
    ]
