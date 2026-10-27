"""
EVALS (LangChain version) - automated testing, no manual typing needed.

Same idea as the raw project's evals.py: run a fixed list of test
questions through the agent, check whether the RIGHT tool was called
(or correctly NOT called), and print a pass/fail report. Exits with a
non-zero code on failure - this is what the CI/CD pipeline gates on.

This file's LLM and embeddings setup must match chat.py exactly (same
model, same task_type) - otherwise evals would be testing a different
configuration than what's actually deployed.

Run with:
    pip install -r requirements.txt
    python3 ingest.py    (if not already done)
    python3 evals.py
"""

import sys
import requests
import time

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

GCP_PROJECT = "agentic-ai-510016"
GCP_LOCATION = "us-central1"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "trip_guide"


weather_cache = {}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    key = city.lower()
    if key in weather_cache:
        cached_result, fetched_at = weather_cache[key]
        if time.time() - fetched_at < 300:
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


# Must match ingest.py's embedding model + task_type exactly, or the
# query vectors and the stored document vectors won't align.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_query",
)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
)


@tool
def search_things_to_explore(query: str) -> str:
    """Search for places to explore/visit in major East Coast and West
    Coast US cities based on what the user is interested in."""
    # Same top-3 + relevance-threshold approach as production chat.py -
    # evals should exercise the actual retrieval logic, not a simplified
    # version of it.
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

    combined = "\n\n".join(
        f"[Match {i+1}, relevance={score:.2f}]: {doc.page_content}"
        for i, (doc, score) in enumerate(relevant_results)
    )
    return combined


SYSTEM_PROMPT = (
    "You are a trip planning assistant for major East Coast and West "
    "Coast US cities. You can check current weather, convert currency, "
    "and search for places to explore in these cities. You must call "
    "only ONE tool at a time. After receiving a tool's result, think "
    "about what to do next based on that result before calling another "
    "tool. Never call multiple tools speculatively just in case they "
    "might be needed. When you use search_things_to_explore, ONLY use "
    "information from the tool's result - do NOT use your own general "
    "knowledge to fill the gap. You have NO budget, cost, or pricing "
    "data available - if asked about trip costs or budgets, say this "
    "information isn't available rather than estimating numbers. If a "
    "question cannot be answered with your available tools, do NOT "
    "call any tool at all - just say you don't have that capability."
)

# Must match chat.py's LLM setup exactly - same model, same vertexai
# routing - evals should test the configuration that's actually deployed.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
)

agent = create_agent(
    model=llm,
    tools=[get_weather, convert_currency, search_things_to_explore],
    system_prompt=SYSTEM_PROMPT,
)


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


def run_agent(question):
    """Runs ONE question through a fresh conversation (no history) and
    returns (answer_text, tools_called)."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    tools_called = []
    for message in result["messages"]:
        # AIMessage objects have a .tool_calls attribute (a list, empty
        # if no tools were called in that turn)
        if hasattr(message, "tool_calls") and message.tool_calls:
            for call in message.tool_calls:
                tools_called.append(call["name"])

    final_message = result["messages"][-1]
    answer_text = get_text_content(final_message.content)

    return answer_text, tools_called


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
        "question": "What are some historic places to explore in Philadelphia?",
        "expected_tool": "search_things_to_explore",
    },
    {
        "question": "What is the estimated budget for a 5 day trip to Miami?",
        "expected_tool": None,  # no tool covers this - agent should say so honestly
    },
    {
        "question": "Can you get me flight details from Mumbai to Seattle?",
        "expected_tool": None,  # no flight tool exists
    },
]


print("=" * 70)
print("RUNNING EVALS (LangChain version)")
print("=" * 70)

passed = 0
failed = 0

for i, case in enumerate(test_cases, start=1):
    print(f"\n[Test {i}] {case['question']}")

    answer, tools_called = run_agent(case["question"])
    expected = case["expected_tool"]

    if expected is None:
        test_passed = (len(tools_called) == 0)
    else:
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

sys.exit(1 if failed > 0 else 0)