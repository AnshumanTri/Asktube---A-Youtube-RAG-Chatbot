"""
In-memory FAISS store keyed by video_id.
Render ephemeral disk = we don't persist. User re-indexes on server restart.
One store per video_id, cached in this module-level dict for the session.
"""

from typing import Dict, List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Module-level cache: {video_id: FAISS}
_store_cache: Dict[str, FAISS] = {}

# Singleton embeddings (expensive to load, load once)
_embeddings: HuggingFaceEmbeddings = None


def _get_embeddings(embeddings: HuggingFaceEmbeddings) -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = embeddings
    return _embeddings


def build_vector_store(video_id: str, chunks: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Build FAISS index from chunks and cache it."""
    emb = _get_embeddings(embeddings)
    store = FAISS.from_documents(chunks, emb)
    _store_cache[video_id] = store
    return store


def get_vector_store(video_id: str) -> FAISS | None:
    """Return cached store or None if not indexed yet."""
    return _store_cache.get(video_id)


def is_indexed(video_id: str) -> bool:
    return video_id in _store_cache


def list_indexed() -> list:
    return list(_store_cache.keys())