# Trip Planner Agent — Agentic AI Learning Project

A hands-on, from-scratch agentic AI project built to learn the core mechanics of tool-calling agents, RAG, and agent guardrails — without relying on a framework (LangChain, LangGraph, etc.). Every layer (the agentic loop, tool contracts, RAG pipeline, evals) was built manually first, specifically to understand what frameworks abstract away before using one.

The assistant answers questions about **Boston, New York, and San Francisco** using three tools:
- **Live weather** (Open-Meteo API)
- **Live currency conversion** (Frankfurter API, 200+ currencies)
- **Places to explore** (RAG over a PDF trip guide, using a local vector database)

Runs entirely on a local machine using **Ollama** (`llama3` + `nomic-embed-text`) — no API keys, no cloud cost.

---

## Why build this without a framework?

The goal was to understand the actual mechanics before trusting a framework's abstractions:
- How a model *requests* a tool call vs. how code *executes* it
- Why a `while` loop — not a fixed number of steps — is what makes an agent "agentic"
- How RAG retrieval can silently fail even when the code runs without errors
- What guardrails actually look like in code, not just in theory

---

## Project Journey (Day 1 → Day 9)

| Day | What was built | Key concept |
|---|---|---|
| **Day 1** | Single-tool agent, manual round-trip (no loop) | The model only *requests* tool calls — it never executes them. Code does. |
| **Day 2** | Two tools (weather + currency), manual round-trips | Handling multiple tool calls in one turn; a real bug where the model sent a number as a string (`"500"` vs `500.0`) |
| **Day 3** | Real `while` loop — the actual agentic loop | Fixed-round logic breaks on dependent, sequential questions (e.g. "check weather, THEN decide currency"); a proper loop lets the model reason incrementally |
| **Day 4** | Swapped fake data for real APIs (Open-Meteo, Frankfurter) | Tool *contracts* — the model never knows or cares whether data comes from a dictionary or a live API |
| **Day 5** | Interactive multi-turn chat (`input()` loop) | Nested while-loops: an outer loop for the conversation, an inner loop for one turn's tool-calling |
| **Day 6** | Guardrails: grounding fixes, TTL-based caching, history trimming | LLMs will confidently hallucinate (fake APIs, fake budget numbers) unless explicitly told to stay grounded in tool results |
| **Day 7** | RAG from a real PDF (`pdfplumber`) | Chunking, embeddings, cosine similarity — and the real debugging lesson: **3 separate silent bugs** stacked on top of each other (see below) |
| **Day 8** | Automated evals (`step12_evals.py`) | Replaces manual conversation testing with a fixed test suite; introduced CI/CD-style exit codes (`sys.exit(1)` on failure) |
| **Day 9** | Real vector database (Chroma) + persistent memory (JSON) | Separated **ingestion** (build the knowledge base once) from **serving** (query it on every chat) — and fixed the RAG bugs found in Day 7 |

---

## Architecture

```
User input
   │
   ▼
Agentic loop (while True)
   │
   ├── Model decides: answer directly, OR request a tool
   │
   ├── get_weather(city)              → Open-Meteo API (live)
   ├── convert_currency(amt, from, to) → Frankfurter API (live)
   └── search_things_to_explore(query) → Chroma vector DB (RAG over trip_guide.pdf)
   │
   ▼
Tool result fed back to model → loop continues until model has a final answer
```

**Ingestion (separate, one-time step — `ingest.py`):**
```
trip_guide.pdf → extract text → split into sentence-based chunks
   → embed each chunk (nomic-embed-text) → store in Chroma (persisted to disk)
```

---

## Setup

```bash
# 1. Install Ollama and pull the required models
ollama pull llama3
ollama pull nomic-embed-text

# 2. Install Python dependencies
pip install requests pdfplumber chromadb

# 3. Build the knowledge base (ONLY needs to run once, or when trip_guide.pdf changes)
python3 ingest.py

# 4. Chat
python3 step15_chroma_and_memory.py

# 5. (Optional) Run automated evals instead of manual testing
python3 step12_evals.py
```

---

## Known Limitations (honest, not overstated)

This is a learning project, not a production system. Specifically:

- **Single local user only.** Conversation memory is a single JSON file — there's no concept of multiple users or sessions.
- **`llama3` (not `llama3.1`) has weaker tool-calling reliability** than larger hosted models (Claude, GPT-4). It occasionally narrates its own reasoning in the final answer instead of answering directly, and earlier in development it "hedged" by calling multiple tools speculatively instead of reasoning step-by-step — a system prompt fix reduced but didn't fully eliminate this.
- **No authentication, rate limiting, or input sanitization.** If this were exposed on a real website, it would need protection against prompt injection and abuse before going live.
- **RAG quality depends on 3 non-obvious fixes** that are easy to silently get wrong (documented below) — even now, retrieval quality is "good enough for a 3-city demo," not rigorously benchmarked.
- **No deployment yet.** Currently runs locally only. AWS Bedrock deployment is a planned next step, not yet implemented.
- **Currency/weather data has no offline fallback.** If Open-Meteo or Frankfurter APIs are down, the tool returns an error rather than degrading gracefully.

## Real bugs found during development (documented on purpose)

Three separate, *silent* RAG bugs were found and fixed — none of them threw an error, all of them just returned wrong data:

1. **Missing embedding prefix** — `nomic-embed-text` is an asymmetric embedding model; it requires `"search_document: "` / `"search_query: "` prefixes to align embeddings correctly. Without them, retrieval was consistently wrong.
2. **Wrong distance metric** — Chroma defaults to L2 (Euclidean) distance, not cosine similarity. Since `nomic-embed-text` embeddings aren't normalized, this gave incorrect rankings even after fixing #1.
3. **Chunking cut sentences mid-way** — fixed-word-count chunking (40 words/chunk) frequently split sentences in half, weakening each chunk's topical signal. Switched to sentence-based chunking (group whole sentences, not words).

This is left in the README deliberately — the debugging process (isolating each bug with a standalone diagnostic script, rather than guessing) is arguably the most useful part of this project to talk about.

## Next steps

- Human-in-the-loop / approval flow for a risky tool (e.g. a mock `book_hotel`)
- Deploy to AWS Bedrock (replacing local Ollama with a hosted model)
- Basic prompt-injection testing
- CI/CD pipeline running `step12_evals.py` before deploy
