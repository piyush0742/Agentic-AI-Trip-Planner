"""
EVALS — automated testing, no manual typing needed.

Same tools, same functions, same RAG setup as step10/step11.
The ONLY difference: instead of input() asking YOU for a question
every time, we have a fixed LIST of test questions, and the script
runs through all of them automatically, printing a report at the end.

This is the "evals" concept from earlier - since AI answers aren't
always word-for-word identical each run, we don't check for an exact
match. Instead we check things like "was the RIGHT tool called" and
print the answer so you can quickly eyeball whether it looks correct.
"""

import requests
import time
import math
import pdfplumber

OLLAMA_URL = "http://localhost:11434/api/chat"
EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "llama3.1"
EMBED_MODEL_NAME = "nomic-embed-text"

MAX_ITERATIONS = 6
CACHE_DURATION_SECONDS = 300
PDF_PATH = "trip_guide.pdf"
WORDS_PER_CHUNK = 40


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


def extract_text_from_pdf(filepath):
    full_text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text


def split_into_chunks(text, words_per_chunk):
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        chunks.append(" ".join(chunk_words))
    return chunks


def get_embedding(text):
    response = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL_NAME, "prompt": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["embedding"]


def sum_of_squares(vector):
    total = 0
    for value in vector:
        total += value * value
    return total


def cosine_similarity(vector1, vector2):
    dot_product = 0
    for i in range(len(vector1)):
        dot_product += vector1[i] * vector2[i]
    magnitude1 = math.sqrt(sum_of_squares(vector1))
    magnitude2 = math.sqrt(sum_of_squares(vector2))
    return dot_product / (magnitude1 * magnitude2)


def search_things_to_explore(query):
    query_embedding = get_embedding(query)
    best_chunk = None
    best_score = -1
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
    "available rather than estimating numbers."
    "If a question cannot be answered with your available tools "
    "(get_weather, convert_currency, search_things_to_explore), do NOT "
    "call any tool at all - just say you don't have that capability."
)


# =======================================================================
# NEW: run_agent() - the same agent loop as before, but wrapped in a
# FUNCTION that takes a question and RETURNS the answer + which tools
# were called, instead of printing to the screen and waiting for input().
# =======================================================================
def run_agent(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    tools_called = []  # keep track of every tool used, for the report
    iteration = 0

    while True:
        iteration += 1
        if iteration > MAX_ITERATIONS:
            return "(stopped - hit MAX_ITERATIONS)", tools_called

        data = call_ollama(messages)
        message = data["message"]

        if "tool_calls" in message and message["tool_calls"]:
            messages.append(message)

            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                tools_called.append(tool_name)

                function_to_call = available_tools[tool_name]
                try:
                    result = function_to_call(**tool_args)
                except Exception as e:
                    result = f"Error running {tool_name}: {e}"

                messages.append({"role": "tool", "content": result})

            continue

        else:
            return message["content"], tools_called


# =======================================================================
# THE TEST CASES — this is the part YOU edit to add more scenarios.
# "expected_tool" is None when the agent should NOT call any tool
# (e.g. a question it genuinely can't answer, like a budget estimate).
# =======================================================================
test_cases = [
    {
        "question": "What's the weather in Boston?",
        "expected_tool": "get_weather",
    },
    {
        "question": "How much is 100 USD in EUR?",
        "expected_tool": "convert_currency",
    },
    {
        "question": "What are some historic places to explore in Boston?",
        "expected_tool": "search_things_to_explore",
    },
    {
        "question": "What is the estimated budget for a 5 day trip to Boston?",
        "expected_tool": None,  # no tool covers this - agent should say so honestly
    },
    {
        "question": "Can you get me flight details from Mumbai to Boston?",
        "expected_tool": None,  # no flight tool exists - agent should say so honestly
    },
]


# =======================================================================
# INGESTION - runs once, before any test starts
# =======================================================================
print("Preparing knowledge base...")
pdf_text = extract_text_from_pdf(PDF_PATH)
chunks = split_into_chunks(pdf_text, WORDS_PER_CHUNK)
document_chunks = []
for chunk_text in chunks:
    document_chunks.append({"text": chunk_text, "embedding": get_embedding(chunk_text)})
print(f"Knowledge base ready - {len(document_chunks)} chunks.\n")


# =======================================================================
# RUN ALL TEST CASES AUTOMATICALLY - this is the actual eval loop
# =======================================================================
print("=" * 70)
print("RUNNING EVALS")
print("=" * 70)

passed = 0
failed = 0

for i, case in enumerate(test_cases, start=1):
    print(f"\n[Test {i}] {case['question']}")

    answer, tools_called = run_agent(case["question"])

    expected = case["expected_tool"]

    if expected is None:
        # We expect NO tool call at all for this one
        test_passed = (len(tools_called) == 0)
    else:
        # We expect this specific tool to appear somewhere in the list
        test_passed = (expected in tools_called)

    if test_passed:
        passed += 1
        print(f"  PASS - tools called: {tools_called}")
    else:
        failed += 1
        print(f"  FAIL - expected tool: {expected}, but got: {tools_called}")

    print(f"  Answer: {answer}")

print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed (out of {len(test_cases)})")
print("=" * 70)