# AskTube — Agentic RAG Chatbot for YouTube

Ask questions about any YouTube video, grounded strictly in its transcript. A Chrome extension backed by an agentic Retrieval-Augmented Generation pipeline , built end to end on free tier infrastructure.

**[Landing Page](https://anshumantri.github.io/Asktube---A-Youtube-RAG-Chatbot)** · **[Install Extension](#installation)** · **[Live Backend](https://asktube-a-youtube-rag-chatbot-2.onrender.com)**

---

## What This Is

AskTube lets you open any YouTube video, index its transcript with one click, and ask questions about it in a persistent Chrome side panel. Answers are generated strictly from the video's content , never from the LLM's general knowledge , with clickable timestamp citations that jump back to the relevant moment in the video.

It is basically a Chrome Plugin Extension . It was built as a study/notes tool: the side panel stays open across tab switches, and each video keeps its own separate chat history, so you can flip between videos without losing your place.

This is not a thin wrapper around a single LLM call. It's a full agentic RAG system: a LangGraph state-machine agent that rewrites queries, retrieves via hybrid search, evaluates whether the retrieved context is sufficient, and retries if not , before generating a guarded, cited answer.

---

## Why It's Different From a Basic RAG Tutorial

| | Typical RAG tutorial | AskTube |
|---|---|---|
| Retrieval | Single similarity search | Hybrid BM25 (keyword) + FAISS MMR (semantic), merged via Reciprocal Rank Fusion |
| Query handling | Raw user question sent directly | LLM rewrites the question for better retrieval before searching |
| Context | All retrieved chunks sent as-is | Contextual compression extracts only the relevant passage per chunk |
| Control flow | One-shot chain | LangGraph agent loop: retrieve → evaluate sufficiency → retry (up to 3x) → generate |
| Generation | Plain answer | Guardrailed (prompt-injection resistant), strictly grounded, structured timestamp citations |
| Evaluation | None | RAGAS: context precision, context recall, faithfulness, answer relevancy |
| Interface | None / CLI | Chrome Manifest V3 side panel with persistent per-video chat history |
| Cost | Often assumes OpenAI | 100% free tier: Groq for LLM, local sentence-transformers for embeddings |

---

## Architecture

```
Chrome Side Panel (Manifest V3)
        │  REST (fetch)
        ▼
FastAPI Backend  ──────────────────────────────────────────
        │
        ├─ Ingestion
        │    youtube-transcript-api (v1.x) → yt-dlp fallback
        │    Sentence-aware chunking (RecursiveCharacterTextSplitter)
        │    Timestamp metadata preserved per chunk
        │
        ├─ Embedding + Storage
        │    sentence-transformers/all-MiniLM-L6-v2 (local, free)
        │    FAISS in-memory vector store, keyed by video ID
        │
        ├─ Agentic Loop (LangGraph)
        │    rewrite → retrieve → evaluate → (retry or) generate
        │    max 3 iterations
        │
        ├─ Retrieval (per iteration)
        │    Pre:  LLM query rewriting
        │    Core: BM25 + FAISS-MMR → Reciprocal Rank Fusion (EnsembleRetriever)
        │    Post: LLMChainExtractor contextual compression
        │
        ├─ Generation
        │    Groq Llama 3.3 70B, tutor-style detailed prompt
        │    Guardrails: prompt-injection detection, strict grounding
        │    Structured timestamp sources (not inline text)
        │
        └─ Evaluation
             RAGAS: context_precision, context_recall, faithfulness, answer_relevancy
             Separate lightweight judge model (Llama 3.1 8B) to avoid quota contention
```

---

## Tech Stack

**Backend:** FastAPI · LangChain · LangGraph · Groq (Llama 3.3 70B) · FAISS · BM25 (rank-bm25) · sentence-transformers · youtube-transcript-api · yt-dlp

**Extension:** Chrome Manifest V3 · Side Panel API · chrome.storage.local · vanilla JS

**Evaluation:** RAGAS · HuggingFace `datasets`

**Deployment:** Render (free tier) · UptimeRobot (keep-alive)

**Cost to run:** $0. No OpenAI, no Pinecone, no paid infrastructure.

---

## Key Engineering Decisions

**Why Groq instead of OpenAI.** Groq's free tier provides fast inference on Llama 3.3 70B with no cost. The tradeoff: a hard 100K-tokens-per-day ceiling per model, which directly shaped pipeline design (see Evaluation Constraints below).

**Why local embeddings instead of an embeddings API.** `sentence-transformers/all-MiniLM-L6-v2` runs on CPU, costs nothing, and removes a network dependency from the hot path.

**Why hybrid retrieval instead of pure semantic search.** BM25 catches exact keyword matches (names, technical terms) that dense embeddings sometimes miss; MMR adds semantic understanding and penalizes redundant chunks. Reciprocal Rank Fusion combines both rankings rather than naively concatenating results.

**Why contextual compression runs on a capped candidate pool.** Compression invokes one LLM call *per retrieved candidate document* — an early version retrieved 12 candidates per question and discarded most after compression, burning ~9 wasted LLM calls per question. `TOP_K` and `BM25_TOP_K` were tuned down to keep the candidate pool close to what's actually kept, cutting compression cost roughly in half without changing retrieval quality.

**Why a side panel instead of a popup.** Manifest V3 popups close on any loss of focus — incompatible with a study tool where users alt-tab between the video and other resources. The side panel persists across tab switches; chat history is keyed per-video in `chrome.storage.local` so switching videos doesn't erase context.

**Why `tabs` permission instead of `activeTab`.** `activeTab` only grants tab URL visibility immediately after a user clicks the extension icon — it does not extend to passive background events like tab-switch listeners. Auto-detecting the active video on every tab change required the broader `tabs` permission instead.

---

## Evaluation Results & Constraints

Evaluated with RAGAS on a 10-question test set (9 content questions + 1 deliberately off-topic question to test hallucination resistance) against a real YouTube video transcript.

| Metric | Score |
|---|---|
| Context Precision | 0.80 |
| Context Recall | 0.62 |
| Faithfulness | 0.70 |
| Answer Relevancy | 0.63 |

**Honest limitation, disclosed rather than hidden:** running a full agentic pipeline (query rewrite + hybrid retrieval + contextual compression + generation, several LLM calls each) across 10 questions, followed by RAGAS's own multi-call judging step, repeatedly exceeded Groq's free-tier 100K-tokens-per-day ceiling for the 70B model during evaluation runs. The results above come from a completed run; subsequent attempts to re-run the full evaluation hit daily quota exhaustion before completing , a genuine free-tier constraint, not a flaw in the pipeline itself. Mitigations applied: a separate lightweight judge model (Llama 3.1 8B) isolated from the main model's quota, reduced retrieval candidate pool, and throttled concurrency , useful but insufficient to guarantee a full clean run every time on a free account. In production with a paid tier, this constraint disappears entirely.

The hallucination-resistance test passed cleanly: the off-topic question correctly received "I couldn't find that in the video transcript" rather than a fabricated answer.

---

## Project Structure

```
youtube-rag-chatbot/
├── backend/              FastAPI server
│   ├── main.py            Routes: /index, /chat, /status, /health
│   ├── agent.py           LangGraph agentic loop
│   ├── retrieval.py       Hybrid BM25 + MMR retrieval, query rewriting, compression
│   ├── generation.py      Prompt, guardrails, citation structuring
│   ├── ingestion.py       Transcript fetch (+ yt-dlp fallback), chunking
│   ├── vector_store.py    FAISS index management
│   ├── config.py          All tunable constants
│   └── requirements.txt
│
├── extension/             Chrome Manifest V3 side panel
│   ├── manifest.json
│   ├── background.js      Service worker: tab-change detection, panel behavior
│   ├── sidepanel.html/css/js
│   └── icons/
│
├── evaluation/            RAGAS evaluation harness
│   ├── eval_dataset.json   Test questions + ground truth answers
│   └── run_ragas.py
│
└── landing-page/
    └── index.html
```

---

## Installation

The backend is already deployed — you only need the extension.

1. **Download** this repository (Code → Download ZIP, or `git clone`).
2. Open Chrome and go to `chrome://extensions`.
3. Enable **Developer mode** (top-right toggle).
4. Click **Load unpacked** and select the `extension/` folder.
5. Open any YouTube video with captions, click the AskTube icon in your toolbar.
6. Click **Index this video**, then start asking questions.

### Running the backend locally (optional, for development)

```bash
cd backend
python -m venv venv
venv\Scripts\activate  
pip install -r requirements.txt
cp .env.example .env     
uvicorn main:app --reload --port 8000
```

Update `BACKEND_URL` in `extension/sidepanel.js` and `host_permissions` in `extension/manifest.json` to `http://localhost:8000` if testing locally instead of against the deployed backend.

### Running the evaluation

```bash
cd evaluation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Fill in eval_dataset.json with real ground-truth answers for your chosen video
python run_ragas.py
```

---

## Known Limitations

- **Free-tier rate limits.** Groq's daily token quota can be exhausted by heavy use or repeated evaluation runs (see Evaluation Constraints above).
- **No persistent vector storage in production.** Render's free tier has ephemeral disk; the FAISS index is rebuilt in memory each time a video is indexed and is lost on server restart. Re-indexing takes ~10-30 seconds.
- **Cold starts.** Render's free tier sleeps after 15 minutes of inactivity; the first request after sleep takes 30-60 seconds. Mitigated with an UptimeRobot keep-alive ping.
- **Transcript availability.** Videos without captions (manual or auto-generated) cannot be indexed.

---

