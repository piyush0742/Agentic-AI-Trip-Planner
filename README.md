# Trip Planner — Production Version

A live, deployed travel-assistant agent: LangChain, Gemini (via Vertex AI),
FastAPI, Docker, and Cloud Run — with CI/CD, observability, and a browser
chat UI. This is the `dev` branch's raw implementation, rebuilt on a
framework and taken from a local script to a running service.

## What it does

A web-based chat assistant that can:
- Check current weather for a city (Open-Meteo API)
- Convert an amount between currencies (Frankfurter API)
- Answer questions about places to explore, grounded in a 10-city PDF
  knowledge base (RAG via Chroma)

Unlike the `dev` branch, this version supports **multiple concurrent
users** (session-based memory), is **rate-limited**, and is **actually
reachable over the internet**.

## Stack

| Concern | Tool |
|---|---|
| Agent framework | LangChain (`create_agent`, `@tool` decorators) |
| LLM | Gemini 2.5 Flash, via Vertex AI (`vertexai=True`) |
| Embeddings | `gemini-embedding-001`, via `VertexAIEmbeddings` |
| Vector store | Chroma (persistent, baked into the image) |
| API | FastAPI |
| Container | Docker |
| Hosting | Google Cloud Run |
| CI/CD | GitHub Actions |
| Auth (CI → GCP) | Workload Identity Federation (no long-lived keys) |
| Logging | Structured JSON → Cloud Logging |
| Tracing | OpenTelemetry → Cloud Trace |

## Why Gemini + Vertex AI instead of Ollama

The `dev` branch self-hosts an LLM locally. This version uses a managed
API instead — Cloud Run only runs lightweight CPU orchestration logic;
the actual model inference happens on Google's infrastructure. This also
removes the embedding-model dependency problem the project hit early on
(see "Real bugs" below) — Ollama has no equivalent on a server that
doesn't run it.

## Project layout

```
langchain/
├── chat.py                 # FastAPI app + agent definition
├── ingest.py                # PDF → chunks → embeddings → Chroma
├── evals.py                 # automated tool-call tests (CI gate)
├── Dockerfile
├── requirements.txt
├── trip_chat_client.html    # browser chat UI, served at "/"
└── trip_guide_expanded.pdf  # 10-city knowledge base

.github/workflows/ci-cd.yml  # test + build + deploy pipeline
```

## Observability

- **Structured logging**: every tool call and chat request logs JSON with
  `session_id`, `latency_seconds`, and `status` — Cloud Logging parses
  this natively, so logs are filterable/queryable by field, not just
  plain text.
- **Distributed tracing**: each request gets a trace (`chat_request` →
  `agent_invoke` → per-tool spans), visible in Cloud Trace. This showed,
  concretely, that LLM reasoning — not the tool/API calls — is the
  dominant cost in request latency.
- **Metrics**: not pushed via a separate API call; derived from the
  structured logs via Cloud Logging's log-based metrics (a console
  config step, not code).

## CI/CD pipeline

On every push: installs dependencies (cached), re-ingests the PDF into
Chroma **only if the PDF's content hash changed** (via GitHub Actions
cache), and runs `evals.py` as a gate. On a successful push to `main`
specifically, it also builds the Docker image, pushes it to Artifact
Registry, and deploys to Cloud Run — no manual `docker build/push/deploy`
required.

Authentication to GCP uses **Workload Identity Federation**, not a
service-account JSON key — GitHub requests a short-lived token at
runtime instead of a long-lived secret sitting in GitHub Secrets.

## Real deployment bugs hit and fixed

1. **Mac ARM vs Cloud Run amd64** — local Docker builds on Apple Silicon
   needed `--platform linux/amd64` for both build and push.
2. **Blocking `input()` breaking uvicorn** — the original interactive
   loop had to be guarded behind `if __name__ == "__main__":` so FastAPI
   could import the module without hanging on startup.
3. **Embedding dimension mismatch** — switching embedding models
   (768 → 3072 dimensions) required fully deleting and rebuilding the
   Chroma store; dimensions aren't interchangeable.
4. **`BatchSpanProcessor` silently dropping traces** — Cloud Run freezes
   a container's CPU immediately after sending the HTTP response. The
   async batch-export background thread never got a chance to run, so
   custom trace spans vanished. Fixed by switching to
   `SimpleSpanProcessor` (synchronous export, before the response
   returns).
5. **Wrong GCP project picked up in CI** — Workload Identity Federation
   defaulted `gcloud` to an unrelated project inferred from the service
   account's name. Fixed by explicitly setting `project_id` in the
   `setup-gcloud` step and `--project` on the deploy command.
6. **`GoogleGenerativeAIEmbeddings` vs `VertexAIEmbeddings`** — the
   former routes through the API-key-based "Gemini Developer API," not
   Vertex AI's ADC-based auth, even when told `vertexai=True`. The actual
   fix was switching to `VertexAIEmbeddings` from
   `langchain-google-vertexai`, which matches the project's existing
   WIF-based authentication.

## Running it locally

```bash
pip install -r requirements.txt
python3 ingest.py    # if chroma_db/ doesn't exist yet
python3 chat.py      # or: uvicorn chat:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` for the chat UI.
