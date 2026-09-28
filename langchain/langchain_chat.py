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

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse



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
# TOOL 1: Weather (real API, TTL cache - same as raw project)
# ---------------------------------------------------------------------
weather_cache = {}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    key = city.lower()
    if key in weather_cache:
        cached_result, fetched_at = weather_cache[key]
        if time.time() - fetched_at < CACHE_DURATION_SECONDS:
            return cached_result

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=15
    )
    geo_data = geo_response.json()
    if "results" not in geo_data:
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
    return result


# ---------------------------------------------------------------------
# TOOL 2: Currency conversion (real API - Frankfurter v2)
# ---------------------------------------------------------------------
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money from one currency to another."""
    amount = float(amount)
    response = requests.get(
        f"https://api.frankfurter.dev/v2/rate/{from_currency.lower()}/{to_currency.lower()}",
        timeout=15
    )
    if response.status_code != 200:
        return f"Could not convert {from_currency} to {to_currency} - currency may not be supported"
    data = response.json()
    rate = data["rate"]
    converted = amount * rate
    return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"


# ---------------------------------------------------------------------
# TOOL 3: RAG search over the PDF (Chroma vector store)
# ---------------------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
      model="gemini-embedding-001",
      project="agentic-ai-510016",   # jo variable pehle se define hai
      vertexai=True,
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
# FASTAPI - single-turn endpoint (no session/memory yet - that's the
# next step). Every request is independent: one question in, one
# answer out, no conversation history carried between requests.
# ---------------------------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def serve_chat_page():
    return FileResponse("trip_chat_client.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

all_conversations = {}
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
def chat(request: ChatRequest):
    conversation = all_conversations.get(request.session_id, [])
    conversation.append({"role": "user", "content": request.message})
    result = agent.invoke({"messages": conversation})    
    final_message = result["messages"][-1]
    answer_text = get_text_content(final_message.content)
    conversation.append({"role": "assistant", "content": answer_text})
    all_conversations[request.session_id] = conversation 
    return {"answer": answer_text}


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