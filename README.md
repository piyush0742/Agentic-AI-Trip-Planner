# Trip Planner — Raw Version

A travel-assistant agent built **from scratch, with no agent framework** —
plain Python, manual tool-calling loop, and a locally-run LLM via Ollama.

## Why this version exists

Before reaching for LangChain, this branch was built to understand what an
agent framework actually abstracts away: tool-calling, memory, retrieval,
and the orchestration loop between them. Every piece here — the tool
dispatch, the RAG pipeline, the conversation memory — is hand-written, so
the trade-offs of using a framework (see the `main` branch) are grounded in
first-hand experience rather than assumption.

## What it does

A terminal-based assistant that can:
- Check current weather for a city (Open-Meteo API, with TTL-based caching)
- Convert an amount between currencies (Frankfurter API)
- Answer questions about things to explore in a city, grounded in a local
  PDF knowledge base (RAG over a city-guide PDF, via Chroma)

The agent decides which tool to call (or none) based on the user's message,
calls it, and reasons over the result before responding — a manual
implementation of the "single-agent, multiple tools" pattern.

## Stack

- **LLM**: Ollama, running `llama3.1` locally
- **Embeddings**: Ollama, `nomic-embed-text`
- **Vector store**: Chroma (persistent, local)
- **Memory**: a JSON file on disk, loaded/saved per session
- **No LangChain, no framework** — the agent loop, tool schemas, and
  message history are all managed directly in Python

## Project layout

```
trip-planner/
├── chat.py              # main entry point — the agent loop
├── ingest.py            # one-time/PDF-change script: chunk + embed + store
├── chunk-diagnose.py    # diagnostic tool used while debugging retrieval
├── evals.py             # automated test cases, non-zero exit on failure
├── trip_guide.pdf        # the knowledge base the RAG tool searches
├── conversation.json    # persisted chat history (generated, gitignored)
└── journey/             # Day-1 … Day-7 — incremental build log
```

## Real bugs and fixes

Retrieval looked "fine" early on but was quietly wrong. Diagnosing it
surfaced four distinct problems:

1. **Embedding prefix mismatch** — `nomic-embed-text` expects
   `search_document:` / `search_query:` prefixes to align document and
   query embeddings correctly. LangChain-style wrappers don't add these
   automatically; a custom `PrefixedOllamaEmbeddings` class was written to
   apply them consistently at ingest and query time.
2. **Wrong distance metric** — Chroma's default didn't match what
   `nomic-embed-text` was trained for; corrected to cosine similarity.
3. **Mid-sentence chunking** — naive fixed-length splitting cut sentences
   mid-way, degrading both embedding quality and retrieved context.
   Fixed by chunking on sentence boundaries.
4. **Weak differentiation at small scale** — with only ~16 chunks across
   10 cities, pure top-1 similarity search sometimes returned the wrong
   city's chunk. Fixed with top-3 retrieval, a 0.5 relevance threshold,
   and multi-candidate labeling so the model can disambiguate.

**A verified finding, not just a claim**: with retrieval deliberately
degraded under identical grounding instructions, `llama3` confidently
hallucinated an answer, while Gemini (tested later, in the `main` branch)
correctly said it didn't have the information. Same prompt, same broken
context, different failure mode.

## Running it

```bash
# Requires Ollama running locally with llama3.1 and nomic-embed-text pulled
pip install -r requirements.txt
python3 ingest.py      # builds the Chroma store from the PDF
python3 chat.py        # starts the interactive terminal chat
```

Run the evals:
```bash
python3 evals.py       # exits non-zero if any test case fails
```

## Known limitations 

- **Local only** — no deployment, no HTTP interface, no multi-user support
- **Single-session memory** — one JSON file, not per-user
- **No observability** — no structured logging, no tracing
- **Requires Ollama running locally** — not portable to a server without it

These aren't oversights — this branch is deliberately scoped to the
"understand the primitives" stage. The production-hardened, deployed
version — LangChain, Gemini, Cloud Run, CI/CD, observability — lives on
the `main` branch.
