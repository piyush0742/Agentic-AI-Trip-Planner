"""
CHAT SCRIPT (LangChain version) — full feature parity with the raw project.

Same 3 tools (weather, currency, RAG search), same guardrails (grounding,
no-budget-data, no-redundant-calls), same persistent memory idea - built
using LangChain's create_agent instead of a hand-written while-loop.

Run with:
    pip install -r requirements.txt
    python3 ingest.py     (run once first, if not already done)
    python3 chat.py
"""

import requests
import time
import json
import os
import sys
import logging

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# ---------------------------------------------------------------------
# CONFIG - fill in your own GCP project id
# ---------------------------------------------------------------------
GCP_PROJECT = "agentic-ai-510016"
GCP_LOCATION = "us-central1"
CACHE_DURATION_SECONDS = 300
MAX_HISTORY_TURNS = 10  # each "turn" = one user message + one assistant reply
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "trip_guide"
CONVERSATION_FILE = "conversation.json"


# ---------------------------------------------------------------------
# OBSERVABILITY SETUP
#
# 1. LOGGING: Cloud Run automatically captures anything written to
#    stdout/stderr and sends it to Cloud Logging. If what we write is
#    valid JSON with a "severity" key, Cloud Logging parses it as a
#    STRUCTURED log entry (filterable by severity, queryable by any
#    field) instead of one long plain-text blob. That's the entire
#    trick - no separate logging agent/service to run, just formatting.
#
# 2. METRICS: we are NOT calling the Cloud Monitoring API to push
#    custom metrics here - that would mean a separate client and an
#    extra network call on every request. Instead we log fields like
#    "latency_seconds" and "status" as structured JSON, and Cloud
#    Logging's "log-based metrics" feature (created in the GCP Console/
#    gcloud, not in this code) turns those fields into real
#    dashboards/alerts. This is the standard low-effort pattern.
#
# 3. TRACING: OpenTelemetry + the Cloud Trace exporter. Each incoming
#    request gets a trace automatically (FastAPIInstrumentor), and we
#    add child spans around the agent call and each tool call - so in
#    Cloud Trace you can see exactly how long the LLM took vs. each
#    tool, for any single request.
# ---------------------------------------------------------------------
class CloudLoggingJSONFormatter(logging.Formatter):
    """One-line JSON per log record, with a 'severity' field - this is
    specifically what Cloud Logging looks for to treat stdout as
    structured (queryable) logs instead of plain text."""

    def format(self, record):
        entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("session_id", "latency_seconds", "tool", "status", "error"):
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        return json.dumps(entry)


logger = logging.getLogger("trip_planner")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(CloudLoggingJSONFormatter())
logger.addHandler(_handler)
logger.propagate = False  # don't also emit via the root logger's plain format

# Tracing - exports spans to Cloud Trace. We use SimpleSpanProcessor
# (synchronous, exports each span immediately) instead of
# BatchSpanProcessor (async, exports in batches on a background
# thread) - Cloud Run freezes the container's CPU right after the
# HTTP response is sent, so a background batch never gets a chance to
# run and spans silently get lost. Synchronous export is slightly
# slower per-request, but guarantees spans are sent before the
# response returns. If Cloud Trace isn't reachable (e.g. running
# locally with no GCP credentials), the exporter just fails silently
# instead of crashing the app.
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    SimpleSpanProcessor(CloudTraceSpanExporter())
)
tracer = trace.get_tracer("trip_planner")


# ---------------------------------------------------------------------
# TOOL 1: Weather (real API, TTL cache - same as raw project)
# ---------------------------------------------------------------------
weather_cache = {}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    with tracer.start_as_current_span("tool.get_weather") as span:
        span.set_attribute("city", city)
        start = time.time()

        key = city.lower()
        if key in weather_cache:
            cached_result, fetched_at = weather_cache[key]
            if time.time() - fetched_at < CACHE_DURATION_SECONDS:
                logger.info("get_weather cache hit", extra={"tool": "get_weather", "status": "cache_hit"})
                return cached_result

        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=15
        )
        geo_data = geo_response.json()
        if "results" not in geo_data:
            logger.warning("get_weather: city not found", extra={"tool": "get_weather", "status": "not_found"})
            return f"Could not find location: {city}"

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=15
        )
        weather_data = weather_response.json()
        temp = weather_data["current_weather"]["temperature"]
        result = f"Current temperature in {city}: {temp}°C"
        weather_cache[key] = (result, time.time())

        logger.info(
            "get_weather completed",
            extra={"tool": "get_weather", "status": "ok", "latency_seconds": round(time.time() - start, 3)}
        )
        return result


# ---------------------------------------------------------------------
# TOOL 2: Currency conversion (real API - Frankfurter v2)
# ---------------------------------------------------------------------
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money from one currency to another."""
    with tracer.start_as_current_span("tool.convert_currency") as span:
        span.set_attribute("from_currency", from_currency)
        span.set_attribute("to_currency", to_currency)
        start = time.time()

        amount = float(amount)
        response = requests.get(
            f"https://api.frankfurter.dev/v2/rate/{from_currency.lower()}/{to_currency.lower()}",
            timeout=15
        )
        if response.status_code != 200:
            logger.warning(
                "convert_currency: unsupported currency pair",
                extra={"tool": "convert_currency", "status": "unsupported"}
            )
            return f"Could not convert {from_currency} to {to_currency} - currency may not be supported"
        data = response.json()
        rate = data["rate"]
        converted = amount * rate

        logger.info(
            "convert_currency completed",
            extra={"tool": "convert_currency", "status": "ok", "latency_seconds": round(time.time() - start, 3)}
        )
        return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"


# ---------------------------------------------------------------------
# TOOL 3: RAG search over the PDF (Chroma vector store)
# ---------------------------------------------------------------------
# Must match ingest.py's embedding model + task_type exactly, or the
# query vectors and the stored document vectors won't align - this is
# Google's direct equivalent of the old manual "search_query:" prefix
# trick nomic-embed-text needed. Also - unlike Ollama - this has no
# dependency on a locally-running model process, so it works
# identically on Cloud Run and in CI, not just on a dev machine.
embeddings = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
    project=GCP_PROJECT,
    location=GCP_LOCATION,
)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
)


@tool
def search_things_to_explore(query: str) -> str:
    """Search for places to explore/visit in major East Coast and West
    Coast US cities (Boston, New York, Philadelphia, Washington DC, Miami,
    Los Angeles, San Francisco, Seattle, Portland, San Diego) based on
    what the user is interested in (history, food, nature, museums,
    sports). Include the city name in your search query if the user
    mentioned one."""
    with tracer.start_as_current_span("tool.search_things_to_explore") as span:
        span.set_attribute("query", query)
        start = time.time()

        # Get top 3 candidates instead of just 1 - a single embedding match
        # isn't always reliable for short/vague queries (confirmed via
        # diagnose.py testing). Giving the model multiple candidates lets
        # it reason over them, instead of blindly trusting one result.
        results = vectorstore.similarity_search_with_relevance_scores(query, k=3)

        RELEVANCE_THRESHOLD = 0.5
        relevant_results = [
            (doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD
        ]

        if not relevant_results:
            logger.info(
                "search_things_to_explore: no relevant match",
                extra={
                    "tool": "search_things_to_explore",
                    "status": "no_match",
                    "latency_seconds": round(time.time() - start, 3),
                }
            )
            return (
                "No sufficiently relevant information found in the knowledge "
                "base for this query."
            )

        # Combine the surviving candidates - clearly separated, so the model
        # can tell them apart and pick/synthesize the actually relevant one(s).
        combined = "\n\n".join(
            f"[Match {i+1}, relevance={score:.2f}]: {doc.page_content}"
            for i, (doc, score) in enumerate(relevant_results)
        )

        logger.info(
            "search_things_to_explore completed",
            extra={
                "tool": "search_things_to_explore",
                "status": "ok",
                "latency_seconds": round(time.time() - start, 3),
            }
        )
        return combined


AVAILABLE_TOOLS = [get_weather, convert_currency, search_things_to_explore]


# ---------------------------------------------------------------------
# SYSTEM PROMPT - same guardrails as the raw project
# ---------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a trip planning assistant for major East Coast and West "
    "Coast US cities: Boston, New York, Philadelphia, Washington DC, "
    "Miami, Los Angeles, San Francisco, Seattle, Portland, and San Diego. "
    "You can check current weather, convert currency, and search for "
    "places to explore in these cities. You must call only ONE tool at "
    "a time. After receiving a tool's result, think about what to do "
    "next based on that result before calling another tool. Never call "
    "multiple tools speculatively just in case they might be needed. "
    "If information has ALREADY been given earlier in this conversation, "
    "reuse it instead of calling the tool again. When you use "
    "search_things_to_explore, the result may contain MULTIPLE candidate "
    "matches labeled [Match 1], [Match 2], etc. with a relevance score - "
    "pick the one(s) that genuinely answer the question, and ignore "
    "matches from the wrong city if the question specified one. ONLY use "
    "information from the tool's result. If none of the matches seem "
    "relevant to the question, say you don't have specific information "
    "on that topic - do NOT use your own general knowledge to fill the "
    "gap. You have NO budget, "
    "cost, or pricing data available - if asked about trip costs or "
    "budgets, say this information isn't available rather than "
    "estimating numbers. If a question cannot be answered with your "
    "available tools, do NOT call any tool at all - just say you don't "
    "have that capability, instead of calling an unrelated tool. If "
    "asked about earlier parts of THIS conversation, answer directly "
    "from history you already have - do NOT call search_things_to_explore "
    "for that. Answer directly without narrating which tool you called "
    "or why."
)


# ---------------------------------------------------------------------
# THE AGENT - Vertex AI (Gemini) + all 3 tools
# ---------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,       # routes through Vertex AI, not the public Gemini API key path
    project=GCP_PROJECT,
    location=GCP_LOCATION,
)

agent = create_agent(
    model=llm,
    tools=AVAILABLE_TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------
# HELPER - Gemini sometimes returns .content as a plain string, and
# sometimes as a list of content blocks (e.g. [{'type': 'text',
# 'text': '...', 'extras': {...}}]). This handles both cases so we
# always print just the clean text.
# ---------------------------------------------------------------------
def get_text_content(content):
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return content


# ---------------------------------------------------------------------
# PERSISTENT MEMORY - simplified vs. the raw project on purpose.
# We store just {"role": ..., "content": ...} pairs (plain text turns),
# NOT the full LangChain message objects with tool-call details. This
# is enough to give the agent conversational context across sessions,
# even though it doesn't replay exact past tool calls.
# ---------------------------------------------------------------------
def load_conversation():
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as f:
            return json.load(f)
    return []


def save_conversation(turns):
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(turns, f, indent=2)


# ---------------------------------------------------------------------
# FASTAPI - multi-user endpoint. Each request carries a session_id,
# so multiple people can chat independently without their conversation
# histories getting mixed up.
# ---------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

# The chat UI is served same-origin (FileResponse from this same app,
# see the "/" route below), so the browser doesn't even need CORS to
# call /chat - this only matters if some other site's frontend calls
# this API directly. Restricted to the service's own Cloud Run URL
# instead of allow_origins=["*"], which accepted requests from any site.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://trip-planner-599550004216.us-central1.run.app"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Automatically creates a trace span for every incoming HTTP request -
# gives you request-level traces in Cloud Trace with zero manual work,
# on top of the custom spans we add inside /chat below.
FastAPIInstrumentor.instrument_app(app)


@app.get("/")
async def serve_chat_page():
    return FileResponse("trip_chat_client.html")


# Per-session conversation memory. NOTE: this lives in process memory -
# it resets whenever the Cloud Run container restarts (see README
# "Known limitations" - Firestore would fix this, not done yet).
all_conversations = {}


# ---------------------------------------------------------------------
# RATE LIMITING - same pattern as weather_cache: a dictionary tracking
# recent request timestamps PER SESSION. If a session has made
# MAX_REQUESTS or more within the last WINDOW_SECONDS, reject the
# request with HTTP 429 instead of calling the (costly) LLM.
# ---------------------------------------------------------------------
rate_limit_tracker = {}   # {"session_id": [timestamp1, timestamp2, ...]}
MAX_REQUESTS = 10
WINDOW_SECONDS = 60


def check_rate_limit(session_id):
    now = time.time()
    timestamps = rate_limit_tracker.get(session_id, [])

    # Drop any timestamps older than the window - they no longer count.
    recent_timestamps = [t for t in timestamps if now - t < WINDOW_SECONDS]

    if len(recent_timestamps) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded - max {MAX_REQUESTS} requests per {WINDOW_SECONDS} seconds."
        )

    recent_timestamps.append(now)
    rate_limit_tracker[session_id] = recent_timestamps


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
async def chat(request: ChatRequest):
    start = time.time()

    with tracer.start_as_current_span("chat_request") as span:
        span.set_attribute("session_id", request.session_id)

        try:
            check_rate_limit(request.session_id)   # raises HTTPException if over limit
        except HTTPException:
            logger.warning(
                "rate limit exceeded",
                extra={"session_id": request.session_id, "status": "rate_limited"}
            )
            raise

        try:
            conversation = all_conversations.get(request.session_id, [])
            conversation.append({"role": "user", "content": request.message})

            with tracer.start_as_current_span("agent_invoke"):
                result = await agent.ainvoke({"messages": conversation})

            final_message = result["messages"][-1]
            answer_text = get_text_content(final_message.content)

            conversation.append({"role": "assistant", "content": answer_text})
            all_conversations[request.session_id] = conversation

            logger.info(
                "chat request completed",
                extra={
                    "session_id": request.session_id,
                    "status": "ok",
                    "latency_seconds": round(time.time() - start, 3),
                }
            )
            return {"answer": answer_text}

        except Exception as e:
            logger.error(
                "chat request failed",
                extra={
                    "session_id": request.session_id,
                    "status": "error",
                    "error": str(e),
                    "latency_seconds": round(time.time() - start, 3),
                }
            )
            raise HTTPException(status_code=500, detail="Internal error processing chat request")


# ---------------------------------------------------------------------
# INTERACTIVE TERMINAL MODE - only runs when this file is executed
# directly (`python3 chat.py`), NOT when uvicorn imports it to find
# `app`. Without this guard, uvicorn would hang forever on input(),
# since there's no interactive terminal for it to read from.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    conversation = load_conversation()

    print("=" * 60)
    print("Trip Planner Assistant (LangChain + Vertex AI version)")
    print("East Coast & West Coast US cities")
    print("Ask about weather, currency conversion, or places to explore.")
    if conversation:
        print(f"(Resumed previous conversation - {len(conversation)} turns loaded)")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ("quit", "exit"):
            save_conversation(conversation)
            print("Goodbye! (conversation saved)")
            break

        if not user_input.strip():
            print("(please type something before pressing Enter)")
            continue

        conversation.append({"role": "user", "content": user_input})

        if len(conversation) > MAX_HISTORY_TURNS * 2:
            conversation = conversation[-(MAX_HISTORY_TURNS * 2):]

        result = agent.invoke({"messages": conversation})

        final_message = result["messages"][-1]
        answer_text = get_text_content(final_message.content)

        print(f"\nAssistant: {answer_text}")

        conversation.append({"role": "assistant", "content": answer_text})
        save_conversation(conversation)