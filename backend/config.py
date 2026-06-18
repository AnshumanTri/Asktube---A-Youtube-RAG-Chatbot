import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"   # free tier, best quality
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 2048

# --- Embeddings (local, no API key needed) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 800       # smaller than demo's 1000 → better precision
CHUNK_OVERLAP = 150

# --- Retrieval ---
# NOTE: contextual compression in retrieval.py runs ONE LLM call PER candidate
# document BEFORE slicing to COMPRESSED_TOP_K. Keep TOP_K/BM25_TOP_K close to
# COMPRESSED_TOP_K to avoid paying for compression calls on documents that get
# discarded immediately after. This was previously 6+6=12 candidates compressed
# down to 3 kept - i.e. ~9 wasted LLM calls per question.
TOP_K = 4              # fetch a bit more than we keep, compress down later
MMR_LAMBDA = 0.6       # diversity vs relevance balance (0=diverse, 1=relevant)
BM25_TOP_K = 4
COMPRESSED_TOP_K = 3   # after contextual compression

# --- Agent ---
MAX_AGENT_ITERATIONS = 3

# --- CORS (Chrome extension origin) ---
ALLOWED_ORIGINS = ["*"]   # tighten in prod if needed