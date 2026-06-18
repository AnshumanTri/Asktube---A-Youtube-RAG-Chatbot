"""
FastAPI backend.

Routes:
  POST /index        - ingest a YouTube video (URL or ID)
  POST /chat         - ask a question about an indexed video
  GET  /status/{id}  - check if a video is indexed
  GET  /health       - health check (UptimeRobot ping target)

In-memory: video chunks stored per video_id for the session lifetime.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from config import ALLOWED_ORIGINS, GROQ_API_KEY
from ingestion import ingest_video, get_embeddings
from vector_store import build_vector_store, get_vector_store, is_indexed, list_indexed
from agent import run_agent
from generation import get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session-level cache: video_id → chunks (for BM25 + reindexing)
_chunks_cache: dict = {}

# Load embeddings once at startup (takes ~5s, cached after)
_embeddings = None


@app.on_event("startup")
async def startup():
    global _embeddings
    logger.info("Loading embedding model...")
    _embeddings = get_embeddings()
    logger.info("Embedding model loaded.")

    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set! /chat will fail.")


# --- Request/Response models ---

class IndexRequest(BaseModel):
    url: str   # YouTube URL or video ID


class IndexResponse(BaseModel):
    video_id: str
    chunk_count: int
    message: str


class ChatRequest(BaseModel):
    video_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list
    iterations: int
    query_used: str


class StatusResponse(BaseModel):
    video_id: str
    indexed: bool


# --- Routes ---

@app.get("/health")
def health():
    """UptimeRobot pings this to keep Render awake."""
    return {"status": "ok"}


@app.get("/status/{video_id}", response_model=StatusResponse)
def status(video_id: str):
    return {"video_id": video_id, "indexed": is_indexed(video_id)}


@app.post("/index", response_model=IndexResponse)
def index_video(req: IndexRequest):
    """
    Ingest a YouTube video:
    1. Fetch transcript
    2. Chunk it
    3. Embed + store in FAISS
    """
    try:
        chunks, video_id = ingest_video(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=422, detail="Transcript was empty after chunking.")

    build_vector_store(video_id, chunks, _embeddings)
    _chunks_cache[video_id] = chunks

    logger.info(f"Indexed video {video_id}: {len(chunks)} chunks")

    return IndexResponse(
        video_id=video_id,
        chunk_count=len(chunks),
        message=f"Video indexed successfully with {len(chunks)} chunks.",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Answer a question about an indexed video using the agentic RAG loop.
    """
    if not is_indexed(req.video_id):
        raise HTTPException(
            status_code=404,
            detail=f"Video '{req.video_id}' is not indexed. Call /index first.",
        )

    if not req.question or len(req.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question is too short.")

    vector_store = get_vector_store(req.video_id)
    chunks = _chunks_cache.get(req.video_id, [])
    llm = get_llm()

    try:
        result = run_agent(
            question=req.question,
            vector_store=vector_store,
            chunks=chunks,
            llm=llm,
            video_id=req.video_id,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Error generating answer. Please try again.")

    return ChatResponse(**result)


@app.get("/indexed-videos")
def indexed_videos():
    """Dev helper: list all currently indexed video IDs."""
    return {"videos": list_indexed()}