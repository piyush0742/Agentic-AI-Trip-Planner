"""
STEP 4 — THE REAL AGENTIC LOOP (while loop)

Same tools, same functions as before. The ONLY thing that changes is
HOW we talk to the model: instead of "exactly 2 rounds, no matter what",
we now say "keep going back and forth until the model is actually done."

This fixes the "hedging" problem you just saw — the model can now
see the weather result FIRST, then decide which ONE currency call
it actually needs, instead of guessing every possibility up front.

New Python idea here: `while True` with a `break` inside it — you've
already learned while-loops on Day 3, this is the same thing, just
with no fixed number of repeats decided in advance.
"""

import requests
import time
import math

OLLAMA_URL = "http://localhost:11434/api/chat"
EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "llama3.1"
EMBED_MODEL_NAME = "nomic-embed-text"

MAX_ITERATIONS = 6  # simple safety net - stop after this many rounds no matter what
CACHE_DURATION_SECONDS = 300  # cache weather for 5 minutes, then re-fetch


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


# ---------------------------------------------------------------------
# FIX 1: Simple cache - a plain dictionary, same idea as fake_data
# dictionaries from earlier. If we've already fetched this city's
# weather in THIS conversation, reuse it instead of calling the API again.
# ---------------------------------------------------------------------
weather_cache = {}


def get_weather(city):
    key = city.lower()

    # Check cache, but only trust it if it's still "fresh" (not expired)
    if key in weather_cache:
        cached_result, fetched_at = weather_cache[key]
        age_in_seconds = time.time() - fetched_at

        if age_in_seconds < CACHE_DURATION_SECONDS:
            print(f"     (using cached result for {city}, {int(age_in_seconds)}s old)")
            return cached_result
        else:
            print(f"     (cached result for {city} expired, fetching fresh data)")

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

    # Store BOTH the result AND when we fetched it
    weather_cache[key] = (result, time.time())
    return result


def convert_currency(amount, from_currency, to_currency):
    amount = float(amount)  # defensive conversion - model may send it as text

    # v2 supports 201 currencies (v1 only supported ~31 major ones,
    # which is why AED failed earlier). Endpoint returns just the RATE,
    # not the converted amount - so we multiply ourselves, same as
    # normal math you already know.
    response = requests.get(
        f"https://api.frankfurter.dev/v2/rate/{from_currency.lower()}/{to_currency.lower()}"
    )

    if response.status_code != 200:
        return f"Could not convert {from_currency} to {to_currency} - currency may not be supported"

    data = response.json()
    rate = data["rate"]
    converted = amount * rate
    return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"


# =======================================================================
# RAG SETUP — this time, reading from a REAL PDF file instead of
# hardcoded Python text. Everything downstream (embeddings, similarity
# search) stays exactly the same as before - only WHERE the text comes
# from has changed.
# =======================================================================
import pdfplumber

PDF_PATH = "trip_guide.pdf"  # put this file in the same folder as this script
WORDS_PER_CHUNK = 40         # how many words go in each chunk


def extract_text_from_pdf(filepath):
    """Reads every page of the PDF and joins it all into one big string."""
    full_text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text


def split_into_chunks(text, words_per_chunk):
    """Splits one big block of text into smaller chunks, a fixed number
    of words at a time. We do this instead of splitting on blank lines,
    because PDF text extraction often loses paragraph breaks - real
    PDFs are messier than clean Python strings."""
    words = text.split()  # splits on whitespace, gives a list of words
    chunks = []

    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

    return chunks


def get_embedding(text):
    """Turns text into a list of numbers (a 'vector') that captures its
    meaning. Two pieces of text about similar topics will have similar
    numbers. We call this once per chunk (at startup) and once per
    user query (when they ask a question)."""
    response = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL_NAME, "prompt": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["embedding"]  # just a list of numbers


def sum_of_squares(vector):
    total = 0
    for value in vector:
        total += value * value
    return total


def cosine_similarity(vector1, vector2):
    """Measures how 'similar' two embeddings are. Returns a number
    between -1 and 1 - closer to 1 means more similar. This is plain
    math (multiplication, addition, square root) - nothing new syntax-wise."""
    dot_product = 0
    for i in range(len(vector1)):
        dot_product += vector1[i] * vector2[i]

    magnitude1 = math.sqrt(sum_of_squares(vector1))
    magnitude2 = math.sqrt(sum_of_squares(vector2))

    return dot_product / (magnitude1 * magnitude2)


def search_things_to_explore(query):
    # NOTE: no more manual "city" filtering. We let semantic search do
    # ALL the work - it will naturally find the right chunk (whichever
    # city it's about) based purely on meaning, not exact name matching.
    query_embedding = get_embedding(query)

    best_chunk = None
    best_score = -1  # cosine similarity is always >= -1, so this is a safe starting point

    for chunk in document_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk["text"]


available_tools = {
    "get_weather": get_weather,
    "convert_currency": convert_currency,
    "search_things_to_explore": search_things_to_explore,
}



def call_ollama(messages):
    """Small helper - just wraps the repeated requests.post() call.
    Same code as before, just pulled into its own function since
    we now call it many times instead of exactly twice."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "messages": messages, "tools": tools, "stream": False},
        timeout=60
    )
    response.raise_for_status()
    return response.json()


# =======================================================================
# THE INTERACTIVE CHAT LOOP
#
# OUTER while loop  -> keeps asking the REAL USER for new questions
# INNER while loop  -> handles ONE question's agent tool-calling
#                      (this is the exact same logic as before, just
#                       moved so it repeats for every new user message)
# =======================================================================

MAX_HISTORY_MESSAGES = 20  # keep conversation from growing forever

print("=" * 60)
print("Trip Planner Assistant - Boston, NY, San Francisco")
print("Ask about weather, currency conversion, or places to explore.")
print("Type 'quit' to exit.")
print("=" * 60)

# ---------------------------------------------------------------------
# INGESTION - this whole block runs ONCE, before the chat starts.
# Read the PDF -> split into chunks -> embed each chunk.
# ---------------------------------------------------------------------
print("\nReading PDF and preparing knowledge base (this takes a few seconds)...")

pdf_text = extract_text_from_pdf(PDF_PATH)
chunks = split_into_chunks(pdf_text, WORDS_PER_CHUNK)

document_chunks = []
for chunk_text in chunks:
    document_chunks.append({
        "text": chunk_text,
        "embedding": get_embedding(chunk_text)
    })

print(f"Knowledge base ready - {len(document_chunks)} chunks loaded from {PDF_PATH}.\n")

messages = [
    {
        "role": "system",
        "content": (
            "You are a trip planning assistant for Boston, New York, and "
            "San Francisco. You can check current weather, convert "
            "currency, and search for places to explore in these 3 "
            "cities. You must call only ONE tool at a time. After "
            "receiving a tool's result, think about what to do next based "
            "on that result before calling another tool. Never call "
            "multiple tools speculatively just in case they might be needed. "
            "If information has ALREADY been given earlier in this "
            "conversation, reuse it instead of calling the tool again."
            "When you use search_things_to_explore, ONLY use information "
            "from the tool's result. If the result doesn't seem relevant to "
            "the question, say you don't have specific information on that "
            "topic - do NOT use your own general knowledge to fill the gap. "
            "You have NO budget/cost/pricing data available - if asked about "
            "trip costs or budgets, say this information isn't available "
            "rather than estimating numbers."
        )
    }
]

# OUTER LOOP - keeps the conversation going, exactly like a real chat
while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    # Add the user's new question to the ongoing conversation history.
    # Notice: we do NOT reset `messages` - old questions/answers stay,
    # so the model still remembers earlier parts of the conversation.
    messages.append({"role": "user", "content": user_input})

    # ------------------------------------------------------------------
    # FIX 2: Trim history if it's getting too long. We ALWAYS keep the
    # system message (messages[0]) - it's the instructions, never drop it.
    # For the rest, we only keep the most recent messages.
    # ------------------------------------------------------------------
    if len(messages) > MAX_HISTORY_MESSAGES:
        system_message = messages[0]
        recent_messages = messages[-(MAX_HISTORY_MESSAGES - 1):]
        messages = [system_message] + recent_messages
        print("  (older conversation history trimmed to save space)")

    iteration = 0

    # INNER LOOP - same tool-calling logic as before, just now it runs
    # once per user question instead of once for the whole script.
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

            messages.append(message)  # remember the model's own turn

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

            # No break - go back to the top of the INNER loop and ask
            # Ollama again, now with this round's tool results included.
            continue

        else:
            # Model is done answering THIS user question.
            print(f"\nAssistant: {message['content']}")
            messages.append({"role": "assistant", "content": message["content"]})
            break  # exits the INNER loop only - back to asking the user