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

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"

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


available_tools = {
    "get_weather": get_weather,
    "convert_currency": convert_currency,
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
print("Ask about weather or currency conversion. Type 'quit' to exit.")
print("=" * 60)

messages = [
    {
        "role": "system",
        "content": (
            "You are a trip planning assistant for Boston, New York, and "
            "San Francisco. You must call only ONE tool at a time. After "
            "receiving a tool's result, think about what to do next based "
            "on that result before calling another tool. Never call "
            "multiple tools speculatively just in case they might be needed. "
            "If information has ALREADY been given earlier in this "
            "conversation, reuse it instead of calling the tool again."
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