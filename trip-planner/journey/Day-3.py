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

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"

MAX_ITERATIONS = 6  # simple safety net - stop after this many rounds no matter what


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


def get_weather(city):
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
    return f"Current temperature in {city}: {temp}°C"


def convert_currency(amount, from_currency, to_currency):
    amount = float(amount)  # defensive conversion - model may send it as text
    fake_rates = {
        ("usd", "inr"): 83.2,
        ("usd", "jpy"): 149.5,
        ("usd", "eur"): 0.92,
    }
    key = (from_currency.lower(), to_currency.lower())
    rate = fake_rates.get(key)

    if rate is None:
        return f"No conversion rate available for {from_currency} to {to_currency}"

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
# THE ACTUAL AGENTIC LOOP
# =======================================================================
messages = [{
    "role": "user",
    "content": (
        "Check the weather in Boston. If it's below 15°C, tell me how much "
        "500 USD is in EUR (I'll need warm clothes from Europe). "
        "Otherwise, tell me how much it is in INR."
    )
}]

iteration = 0

while True:
    iteration += 1
    print(f"\n===== ROUND {iteration} =====")

    if iteration > MAX_ITERATIONS:
        print("Stopping - hit MAX_ITERATIONS safety limit.")
        break

    try:
        data = call_ollama(messages)
    except Exception as e:
        print(f"Something went wrong calling Ollama: {e}")
        break

    message = data["message"]

    # ------------------------------------------------------------------
    # THE ONE DECISION THAT MATTERS: is the model asking for a tool,
    # or is it actually done?
    # ------------------------------------------------------------------
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

        # Notice: NO break here. We loop back to the top and ask
        # Ollama again, now WITH this round's results included.
        continue

    else:
        # No more tool calls -> model is genuinely done. Print and stop.
        print("\nFINAL ANSWER:")
        print(f"  {message['content']}")
        break