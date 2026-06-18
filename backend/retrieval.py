"""
Retrieval pipeline:
Pre-retrieval  : LLM query rewriting (expand ambiguous queries)
Retrieval      : Hybrid = BM25 (keyword) + FAISS MMR (semantic + diversity)
                 Results merged via Reciprocal Rank Fusion
Post-retrieval : Contextual compression (LLM extracts only relevant passages)
"""

from typing import List, Tuple
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    TOP_K, MMR_LAMBDA, BM25_TOP_K, COMPRESSED_TOP_K,
    GROQ_TEMPERATURE
)


def rewrite_query(question: str, llm: ChatGroq) -> str:
    """
    Pre-retrieval: rewrite the user query to be more retrieval-friendly.
    Expands pronouns, makes implicit topics explicit.
    Kept tight to save tokens.
    """
    prompt = (
        "Rewrite the following question to improve document retrieval from a YouTube transcript. "
        "Make it specific and self-contained. Output ONLY the rewritten question, nothing else.\n\n"
        f"Question: {question}"
    )
    response = llm.invoke(prompt)
    rewritten = response.content.strip()
    # Safety: if LLM returns something weird/too long, fall back to original
    if len(rewritten) > 300 or not rewritten:
        return question
    return rewritten


def build_hybrid_retriever(
    vector_store: FAISS,
    chunks: List[Document],
    llm: ChatGroq,
) -> ContextualCompressionRetriever:
    """
    Build the full retrieval stack:
    BM25 + FAISS-MMR → EnsembleRetriever → ContextualCompression
    """
    # BM25: keyword-based sparse retrieval
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = BM25_TOP_K

    # FAISS MMR: semantic + diversity (avoids redundant chunks)
    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "lambda_mult": MMR_LAMBDA,   # 0.6 = slight bias toward relevance
            "fetch_k": TOP_K * 3,        # MMR considers fetch_k candidates
        }
    )

    # EnsembleRetriever: merges BM25 + MMR via Reciprocal Rank Fusion
    # weights must sum to 1.0; equal weight works well empirically
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, mmr_retriever],
        weights=[0.4, 0.6],   # slightly favor semantic over keyword
    )

    # Contextual Compression: LLM reads each chunk, extracts only relevant passage
    # This reduces context tokens significantly before sending to generation LLM
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    return compression_retriever


def retrieve(
    question: str,
    vector_store: FAISS,
    chunks: List[Document],
    llm: ChatGroq,
    skip_rewrite: bool = False,
) -> Tuple[List[Document], str]:
    """
    Full retrieval for one question.
    Returns (relevant_docs, rewritten_query)
    skip_rewrite=True when agent already rewrote the query.
    """
    rewritten = question if skip_rewrite else rewrite_query(question, llm)

    retriever = build_hybrid_retriever(vector_store, chunks, llm)
    docs = retriever.invoke(rewritten)

    # Cap to COMPRESSED_TOP_K to keep context window tight
    docs = docs[:COMPRESSED_TOP_K]

    return docs, rewritten