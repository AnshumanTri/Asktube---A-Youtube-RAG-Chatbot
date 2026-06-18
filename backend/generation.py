"""
Generation:
- Structured prompt with context window management
- Guardrails: refuse off-topic, hallucination-reducing instructions
- Citations: answer references chunk timestamps
"""

from typing import List
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS


def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
    )


# System prompt: guardrails baked in
SYSTEM_PROMPT = """You are a knowledgeable tutor helping a student understand a YouTube video they are studying from. \
Your job is to answer questions based ONLY on the provided transcript excerpts.

STRICT RULES:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. If the context does not contain enough information, respond EXACTLY: "I couldn't find that in the video transcript."
3. Do NOT make up facts, quotes, or details not present in the context.
4. Do NOT include timestamps, citation markers, or source references in your answer text — sources are handled separately.

STYLE:
- Default to a thorough, explanatory answer, like a tutor walking a student through a concept. Explain the "why", not just the "what".
- Synthesize and explain in your own words rather than repeating the excerpts verbatim.
- If the student explicitly asks for a shorter, quicker, or one-line answer, give a brief answer instead.
- If the student asks for more depth or "explain more", expand further than your default."""

USER_PROMPT = """TRANSCRIPT EXCERPTS:
{context}

QUESTION: {question}

Answer based strictly on the excerpts above. Do not mention timestamps in your answer text:"""


def format_context(docs: List[Document]) -> str:
    """Format retrieved docs into context string with timestamps (for the LLM's eyes only)."""
    parts = []
    for i, doc in enumerate(docs, 1):
        ts = doc.metadata.get("start_time", 0)
        minutes = int(ts) // 60
        seconds = int(ts) % 60
        timestamp_str = f"{minutes}m{seconds}s" if minutes > 0 else f"{seconds}s"
        parts.append(f"[Excerpt {i} | ~{timestamp_str} into video]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_sources(docs: List[Document], video_id: str) -> List[dict]:
    """
    Build structured, deduplicated timestamp sources for the frontend to render
    as clickable links. Kept separate from answer text so the UI controls formatting.
    """
    sources = []
    seen_seconds = set()

    for doc in docs:
        ts = doc.metadata.get("start_time", 0)
        seconds = int(ts)
        if seconds in seen_seconds:
            continue
        seen_seconds.add(seconds)

        minutes = seconds // 60
        secs = seconds % 60
        label = f"{minutes}m {secs}s" if minutes > 0 else f"{secs}s"

        sources.append({
            "label": label,
            "seconds": seconds,
            "url": f"https://www.youtube.com/watch?v={video_id}&t={seconds}s",
        })

    sources.sort(key=lambda s: s["seconds"])
    return sources


def generate_answer(question: str, docs: List[Document], llm: ChatGroq, video_id: str = "") -> dict:
    """
    Generate final answer. Returns {"answer": str, "sources": list[dict]}.
    Sources are structured separately from answer text for clean, clickable rendering.
    """
    if not docs:
        return {"answer": "I couldn't find that in the video transcript.", "sources": []}

    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    sources = build_sources(docs, video_id) if video_id else []

    return {"answer": answer, "sources": sources}


def check_guardrails(question: str) -> str | None:
    """
    Fast pre-check before hitting LLM.
    Returns a rejection message if the question is clearly off-topic/harmful.
    Returns None if question is fine to proceed.
    """
    blocked_patterns = [
        "ignore previous", "forget your instructions", "jailbreak",
        "pretend you are", "act as", "dan mode",
    ]
    q_lower = question.lower()
    for pattern in blocked_patterns:
        if pattern in q_lower:
            return "I'm a YouTube video assistant and can only answer questions about the video content."

    if len(question.strip()) < 3:
        return "Please ask a valid question about the video."

    return None  # all clear