"""
STEP 2 — TWO TOOLS, STILL NO WHILE-LOOP.

New idea vs step 1: since there are now 2 tools, the model might ask
for ONE of them, or BOTH at once, or NEITHER. So instead of assuming
"there's exactly 1 tool_call", we now loop over the whole list with
a normal for-loop (something you already know).

Everything else — dictionaries, functions, if/else, try/except —
is exactly the same as step 1.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"


# ---------------------------------------------------------------------
# Now TWO tool descriptions in the list, instead of one.
# ---------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
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
# get_weather — same real API call from before (Open-Meteo, free, no key)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# convert_currency — fake data, same pattern as get_weather in step 1.
# Normal function, dictionary lookup, nothing new.
# ---------------------------------------------------------------------
def convert_currency(amount, from_currency, to_currency):
    amount = float(amount)  # <-- defensively convert, in case it arrives as a string
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


# ---------------------------------------------------------------------
# A dictionary mapping tool NAME (as a string) -> the actual function.
# This is what lets us call "whichever tool the model asked for"
# without writing a big if/elif chain for every tool.
# ---------------------------------------------------------------------
available_tools = {
    "get_weather": get_weather,
    "convert_currency": convert_currency,
}


# =======================================================================
# STEP 1: Send the question
# =======================================================================
print("STEP 1: Sending question to Ollama...")

#messages = [{
#   "role": "user",
#    "content": "What's the weather in Boston, and how much is 500 USD in INR?"
#}]
messages = [{
    "role": "user",
    "content": (
        "Check the weather in Boston. If it's below 15°C, tell me how much "
        "500 USD is in EUR (I'll need warm clothes from Europe). "
        "Otherwise, tell me how much it is in INR."
    )
}]
try:
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "messages": messages, "tools": tools, "stream": False},
        timeout=60
    )
    response.raise_for_status()
    data = response.json()
except Exception as e:
    print(f"Something went wrong calling Ollama: {e}")
    raise SystemExit(1)

print("-" * 60)

message = data["message"]

# =======================================================================
# STEP 2: Loop over EVERY tool call the model asked for (could be 0, 1, or 2)
# =======================================================================
if "tool_calls" in message and message["tool_calls"]:

    print(f"STEP 2: Model asked for {len(message['tool_calls'])} tool call(s)")

    messages.append(message)  # remember the model's own turn

    # A normal for-loop — you already know this from Day 3
    for tool_call in message["tool_calls"]:
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]

        print(f"  -> calling '{tool_name}' with {tool_args}")

        # Look up the right function using our dictionary, then call it.
        # **tool_args unpacks the dictionary into keyword arguments —
        # e.g. {"city": "Tokyo"} becomes get_weather(city="Tokyo")
        function_to_call = available_tools[tool_name]
        try:
            result = function_to_call(**tool_args)
        except Exception as e:
            result = f"Error running {tool_name}: {e}"

        print(f"     result: {result}")

        # Add this tool's result to the conversation
        messages.append({"role": "tool", "content": result})

    print("-" * 60)

    # ===================================================================
    # STEP 3: Send everything back, ONE more time, for the final answer
    # ===================================================================
    print("STEP 3: Sending all results back to Ollama...")

    try:
        final_response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "messages": messages, "tools": tools, "stream": False},
            timeout=60
        )
        final_response.raise_for_status()
        final_data = final_response.json()
    except Exception as e:
        print(f"Something went wrong on the second call: {e}")
        raise SystemExit(1)

    print("\nSTEP 4: Model's FINAL answer:")
    print(f"  {final_data['message']['content']}")

else:
    print("Model answered directly, no tool needed:")
    print(f"  {message['content']}")