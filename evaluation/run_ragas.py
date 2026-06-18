"""
RAGAS Evaluation Script.

Runs the actual RAG pipeline (retrieval + generation, NOT the agent wrapper)
against a fixed Q&A dataset, then scores it with RAGAS metrics:
  - context_precision : are the retrieved chunks actually relevant?
  - context_recall     : did retrieval find everything needed to answer?
  - faithfulness        : is the answer grounded in the retrieved context?
  - answer_relevancy    : does the answer actually address the question?

Usage:
    python run_ragas.py

Requires eval_dataset.json filled in with real ground_truth answers.
Reuses backend/ modules directly - no logic duplicated.
"""

import sys
import json
import os
import time
from pathlib import Path

# Allow importing from ../backend without restructuring the project
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq

from ingestion import ingest_video, get_embeddings
from vector_store import build_vector_store
from retrieval import retrieve
from generation import generate_answer
from config import GROQ_API_KEY


def load_eval_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "_instructions" in data:
        raise ValueError(
            "eval_dataset.json still has the _instructions placeholder. "
            "Fill in real ground_truth answers and remove _instructions before running."
        )

    placeholder_markers = ["REPLACE_WITH", "PASTE_"]
    for pair in data["qa_pairs"]:
        for marker in placeholder_markers:
            if marker in pair["question"] or marker in pair["ground_truth"]:
                raise ValueError(
                    f"Found unfilled placeholder in: {pair}. "
                    "Replace all REPLACE_WITH_* / PASTE_* fields with real content."
                )

    return data


def run_pipeline_on_dataset(video_id: str, qa_pairs: list, pipeline_llm: ChatGroq) -> dict:
    """
    Index the video once, then run retrieval+generation for each question.
    Returns dict ready for RAGAS Dataset construction.

    pipeline_llm is injected (not generation.get_llm()) so evaluation can use
    a lower max_tokens cap than the live chatbot - detailed tutor-style answers
    are great for real users but expensive for a 10-question batch eval on a
    100K-token daily free tier. The live chatbot's behavior is untouched.

    A short delay between questions avoids bursting past Groq's
    tokens-per-minute limit.
    """
    print(f"Indexing video: {video_id}")
    chunks, _ = ingest_video(video_id)
    embeddings = get_embeddings()
    vector_store = build_vector_store(video_id, chunks, embeddings)

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, pair in enumerate(qa_pairs, 1):
        question = pair["question"]
        print(f"[{i}/{len(qa_pairs)}] {question}")

        docs, _ = retrieve(question, vector_store, chunks, pipeline_llm)
        result = generate_answer(question, docs, pipeline_llm, video_id)
        answer = result["answer"]
        context_texts = [doc.page_content for doc in docs]

        questions.append(question)
        answers.append(answer)
        contexts_list.append(context_texts)
        ground_truths.append(pair["ground_truth"])

        if i < len(qa_pairs):
            time.sleep(5)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def main():
    dataset_path = Path(__file__).parent / "eval_dataset.json"
    data = load_eval_dataset(str(dataset_path))

    # Capped-output LLM for the pipeline itself (retrieval rewrite, compression,
    # generation). Lower max_tokens than the live chatbot's 2048 - this keeps
    # answers shorter ONLY during evaluation, to fit Groq's 100K-token daily
    # free-tier limit across 10 questions x ~7 calls each. The live chatbot
    # (generation.get_llm() / config.GROQ_MAX_TOKENS) is untouched.
    pipeline_llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=400,
    )

    pipeline_results = run_pipeline_on_dataset(data["video_id"], data["qa_pairs"], pipeline_llm)
    hf_dataset = Dataset.from_dict(pipeline_results)

    print("\nRunning RAGAS evaluation (this calls the LLM multiple times per row)...")
    print("Using a lighter/separate Groq model as judge, throttled to avoid rate limits.")
    print("This will be noticeably slower than a single pipeline run - that's expected.\n")

    # Separate, smaller judge model - has its own rate-limit pool from the
    # main llama-3.3-70b chatbot model, and uses fewer tokens per judge call.
    judge_llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    evaluator_llm = LangchainLLMWrapper(judge_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    # Lower concurrency (max_workers) so we don't burst past Groq's
    # tokens-per-minute limit, with generous retry/backoff for any
    # transient 429s that still occur.
    throttled_config = RunConfig(
        max_workers=2,
        timeout=120,
        max_retries=8,
        max_wait=30,
    )

    result = evaluate(
        dataset=hf_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=throttled_config,
    )

    print("\n=== RAGAS Results ===")
    print(result)

    df = result.to_pandas()
    output_path = Path(__file__).parent / "ragas_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFull per-question results saved to: {output_path}")


if __name__ == "__main__":
    main()