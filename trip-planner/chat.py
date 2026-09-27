"""
STEP 15 — REAL VECTOR DATABASE + PERSISTENT MEMORY

TWO upgrades from before:

1. VECTOR DATABASE (Chroma) instead of our own manual for-loop +
   cosine_similarity. Chroma stores embeddings on DISK (in ./chroma_db,
   built by ingest.py) and searches them internally - no re-reading the
   PDF, no re-computing embeddings every time you start this script.

2. PERSISTENT MEMORY - conversation history is saved to a JSON file.
   Close this script, run it again tomorrow - it remembers what you
   talked about, instead of starting blank every time.

BEFORE RUNNING THIS:
  1. Run ingest.py FIRST (only needs to happen once, or when the PDF changes)
  2. pip install chromadb requests
"""

import requests
import time
import math
import json
import os
import chromadb

OLLAMA_URL = "http://localhost:11434/api/chat"
EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "llama3.1"
EMBED_MODEL_NAME = "nomic-embed-text"

MAX_ITERATIONS = 6
CACHE_DURATION_SECONDS = 300
MAX_HISTORY_MESSAGES = 20

CHROMA_DB_PATH = "./chroma_db"          # built by ingest.py
CONVERSATION_FILE = "conversation.json"  # where we save chat history


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount of money from one currency to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_things_to_explore",
            "description": (
                "Search for places to explore/visit in Boston, New York, or "
                "San Francisco based on what the user is interested in "
                "(e.g. history, food, nature, museums, sports). Include the "
                "city name in your search query if the user mentioned one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "e.g. 'historic sites in Boston' or 'parks in New York'"}
                },
                "required": ["query"]
            }
        }
    }
]


weather_cache = {}


def get_weather(city):
    key = city.lower()
    if key in weather_cache:
        cached_result, fetched_at = weather_cache[key]
        if time.time() - fetched_at < CACHE_DURATION_SECONDS:
            return cached_result

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
    )
    geo_data = geo_response.json()
    if "results" not in geo_data:
        return f"Could not find location: {city}"

    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True}
    )
    weather_data = weather_response.json()
    temp = weather_data["current_weather"]["temperature"]
    result = f"Current temperature in {city}: {temp}°C"
    weather_cache[key] = (result, time.time())
    return result


def convert_currency(amount, from_currency, to_currency):
    amount = float(amount)
    response = requests.get(
        f"https://api.frankfurter.dev/v2/rate/{from_currency.lower()}/{to_currency.lower()}"
    )
    if response.status_code != 200:
        return f"Could not convert {from_currency} to {to_currency} - currency may not be supported"
    data = response.json()
    rate = data["rate"]
    converted = amount * rate
    return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"


def get_embedding(text):
    response = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL_NAME, "prompt": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["embedding"]


# ---------------------------------------------------------------------
# UPGRADE 1: connect to the EXISTING Chroma database (built by
# ingest.py). We do NOT read the PDF or compute embeddings here at
# all anymore - that already happened, once, in ingest.py.
# ---------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="trip_guide")


def search_things_to_explore(query):
    # Same asymmetric-model requirement as ingest.py - queries need the
    # "search_query: " prefix, documents needed "search_document: ".
    # Mismatching these is exactly what caused wrong search results.
    query_embedding = get_embedding("search_query: " + query)

    # Chroma does the similarity search internally - no more manual
    # for-loop + cosine_similarity(). n_results=1 means "give me the
    # single best match".
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    # results["documents"] is a list-of-lists (one list per query we
    # sent) - we only sent 1 query, so we grab [0][0]: first query's
    # first (best) result.
    if not results["documents"] or not results["documents"][0]:
        return "No information found."

    return results["documents"][0][0]


available_tools = {
    "get_weather": get_weather,
    "convert_currency": convert_currency,
    "search_things_to_explore": search_things_to_explore,
}


def call_ollama(messages):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "messages": messages, "tools": tools, "stream": False},
        timeout=60
    )
    response.raise_for_status()
    return response.json()


SYSTEM_PROMPT = (
    "You are a trip planning assistant for Boston, New York, and "
    "San Francisco. You can check current weather, convert "
    "currency, and search for places to explore in these 3 "
    "cities. You must call only ONE tool at a time. After "
    "receiving a tool's result, think about what to do next based "
    "on that result before calling another tool. Never call "
    "multiple tools speculatively just in case they might be needed. "
    "If information has ALREADY been given earlier in this "
    "conversation, reuse it instead of calling the tool again. "
    "When you use search_things_to_explore, ONLY use information "
    "from the tool's result. If the result doesn't seem relevant to "
    "the question, say you don't have specific information on that "
    "topic - do NOT use your own general knowledge to fill the gap. "
    "You have NO budget, cost, or pricing data available - if asked "
    "about trip costs or budgets, say this information isn't "
    "available rather than estimating numbers. "
    "If a question cannot be answered with your available tools "
    "(get_weather, convert_currency, search_things_to_explore), do "
    "NOT call any tool at all - just say you don't have that "
    "capability, instead of calling an unrelated tool."
    "If asked about earlier parts of THIS conversation (e.g. 'what did we "
    "talk about'), answer directly from the conversation history you "      
    "already have - do NOT call search_things_to_explore for that, since "
    "that tool is only for searching city information, not conversation memory."
)


# ---------------------------------------------------------------------
# UPGRADE 2: PERSISTENT MEMORY - load conversation from a JSON file
# if one already exists, instead of always starting fresh.
#
# json.dump() / json.load() are just file-handling (which you already
# know from Day 4) with ONE difference: they automatically convert
# Python lists/dictionaries into text and back, so we don't have to
# do that conversion ourselves.
# ---------------------------------------------------------------------
def load_conversation():
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as f:
            return json.load(f)  # reads the file, turns it back into a Python list
    else:
        return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_conversation(messages):
    with open(CONVERSATION_FILE, "w") as f:
        json.dump(messages, f, indent=2)  # turns the Python list into text, writes it


messages = load_conversation()

print("=" * 60)
print("Trip Planner Assistant - Boston, NY, San Francisco")
print("Ask about weather, currency conversion, or places to explore.")
if len(messages) > 1:
    print(f"(Resumed previous conversation - {len(messages) - 1} messages loaded)")
print("Type 'quit' to exit.")
print("=" * 60)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ("quit", "exit"):
        save_conversation(messages)  # save one last time before closing
        print("Goodbye! (conversation saved)")
        break

    messages.append({"role": "user", "content": user_input})

    if len(messages) > MAX_HISTORY_MESSAGES:
        system_message = messages[0]
        recent_messages = messages[-(MAX_HISTORY_MESSAGES - 1):]
        messages = [system_message] + recent_messages

    iteration = 0

    while True:
        iteration += 1

        if iteration > MAX_ITERATIONS:
            print("Stopping - hit MAX_ITERATIONS safety limit.")
            break

        try:
            data = call_ollama(messages)
        except Exception as e:
            print(f"Something went wrong calling Ollama: {e}")
            break

        message = data["message"]

        if "tool_calls" in message and message["tool_calls"]:
            messages.append(message)

            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]

                print(f"  -> calling '{tool_name}' with {tool_args}")

                function_to_call = available_tools[tool_name]
                try:
                    result = function_to_call(**tool_args)
                except Exception as e:
                    result = f"Error running {tool_name}: {e}"

                print(f"     result: {result}")
                messages.append({"role": "tool", "content": result})

            continue

        else:
            print(f"\nAssistant: {message['content']}")
            messages.append({"role": "assistant", "content": message["content"]})
            save_conversation(messages)  # save after every turn, not just on quit
            break