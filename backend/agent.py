"""
Agentic RAG loop using LangGraph.

State machine:
  START → rewrite_query → retrieve → evaluate_context
             ↑                              |
             |        (context poor)        |
             └──────── rewrite_query ←──────┘
                                            |
                              (context good / max iters)
                                            ↓
                                        generate → END

The agent decides if retrieved context is sufficient.
If not, it rewrites the query and tries again (max MAX_AGENT_ITERATIONS).
"""

from typing import TypedDict, List, Optional
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

from retrieval import retrieve, rewrite_query
from generation import generate_answer, check_guardrails
from config import MAX_AGENT_ITERATIONS


# --- State definition ---

class AgentState(TypedDict):
    question: str               # original user question
    current_query: str          # may be rewritten
    docs: List[Document]        # retrieved docs current iteration
    answer: Optional[dict]      # final {"answer": str, "sources": list} when done
    iterations: int             # how many retrieval attempts so far
    vector_store: FAISS         # passed in, not modified
    chunks: List[Document]      # passed in, for BM25
    llm: ChatGroq               # passed in
    video_id: str                # passed in, for building timestamp source URLs


# --- Node functions ---

def node_rewrite(state: AgentState) -> AgentState:
    """Rewrite the current query for better retrieval."""
    llm = state["llm"]
    question = state["question"]
    current_query = state["current_query"]

    # First iteration: rewrite original question
    # Subsequent: rewrite with hint that previous attempt was insufficient
    if state["iterations"] == 0:
        new_query = rewrite_query(current_query, llm)
    else:
        hint_prompt = (
            f"The previous search query '{current_query}' did not return sufficient results "
            f"for the question: '{question}'. "
            "Generate a different, more specific search query. Output ONLY the query."
        )
        response = llm.invoke(hint_prompt)
        new_query = response.content.strip()[:300] or current_query

    return {**state, "current_query": new_query}


def node_retrieve(state: AgentState) -> AgentState:
    """Retrieve docs using current query."""
    docs, _ = retrieve(
        question=state["current_query"],
        vector_store=state["vector_store"],
        chunks=state["chunks"],
        llm=state["llm"],
        skip_rewrite=True,   # agent already rewrote
    )
    return {**state, "docs": docs, "iterations": state["iterations"] + 1}


def node_evaluate(state: AgentState) -> AgentState:
    """
    Evaluate if retrieved context is sufficient.
    Simple heuristic: if we have ≥1 doc with meaningful content, proceed to generate.
    Could be upgraded to LLM-based relevance scoring later.
    """
    docs = state["docs"]
    sufficient = (
        len(docs) >= 1
        and any(len(doc.page_content.strip()) > 50 for doc in docs)
    )

    if sufficient or state["iterations"] >= MAX_AGENT_ITERATIONS:
        # Generate answer (even if insufficient, generation handles "I don't know")
        result = generate_answer(state["question"], docs, state["llm"], state["video_id"])
        return {**state, "answer": result}

    # Not sufficient yet, will route back to rewrite
    return {**state, "answer": None}


# --- Routing function ---

def route_after_evaluate(state: AgentState) -> str:
    """Route: if answer is set → END, else → rewrite."""
    if state["answer"] is not None:
        return "end"
    return "rewrite"


# --- Build graph ---

def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("rewrite", node_rewrite)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("evaluate", node_evaluate)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"end": END, "rewrite": "rewrite"},
    )

    return graph.compile()


# Module-level compiled agent (build once, reuse)
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def run_agent(
    question: str,
    vector_store: FAISS,
    chunks: List[Document],
    llm: ChatGroq,
    video_id: str,
) -> dict:
    """
    Entry point. Returns {"answer": str, "sources": list, "iterations": int, "query_used": str}
    """
    # Guardrail check before agent runs
    rejection = check_guardrails(question)
    if rejection:
        return {"answer": rejection, "sources": [], "iterations": 0, "query_used": question}

    initial_state: AgentState = {
        "question": question,
        "current_query": question,
        "docs": [],
        "answer": None,
        "iterations": 0,
        "vector_store": vector_store,
        "chunks": chunks,
        "llm": llm,
        "video_id": video_id,
    }

    agent = get_agent()
    final_state = agent.invoke(initial_state)
    result = final_state["answer"]

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "iterations": final_state["iterations"],
        "query_used": final_state["current_query"],
    }