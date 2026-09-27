"""
STEP 1 — ONE TOOL, NO LOOP — using OLLAMA (runs on YOUR machine, free)
instead of Claude's API.

Same Python concepts as before, nothing new required:
  - variables, print()          -> Day 1
  - dictionaries                -> Day 2
  - if/else                     -> Day 3
  - functions (def, return)     -> Day 3
  - try/except                  -> Intermediate (done)

ONE genuinely new thing: the `requests` library. Ollama runs a small
web server on your own machine (at http://localhost:11434). Instead of
using a library object like `anthropic.Anthropic()`, we send it a
normal HTTP request — same idea as visiting a website, except our
Python code is doing the "visiting" instead of a browser.

BEFORE RUNNING THIS:
  1. Make sure Ollama is installed and running (check with: ollama --version)
  2. Pull a tool-capable model:     ollama pull llama3.1
  3. Install requests:              pip install requests
  4. Run this file:                 python step1_ollama.py
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1"


# ---------------------------------------------------------------------
# Same idea as before — a dictionary describing one tool. The format
# Ollama expects looks slightly different from Claude's (it wraps
# things inside a "function" key), but it's still just a dictionary.
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
    }
]


# ---------------------------------------------------------------------
# Same normal function as before — nothing new here.
# ---------------------------------------------------------------------
def get_weather(city):
    # Step 1: convert city name to coordinates (lat/lon)
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
    )
    geo_data = geo_response.json()

    if "results" not in geo_data:
        return f"Could not find location: {city}"

    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]

    # Step 2: get the actual current weather for those coordinates
    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True}
    )
    weather_data = weather_response.json()

    temp = weather_data["current_weather"]["temperature"]
    return f"Current temperature in {city}: {temp}°C"


# =======================================================================
# STEP 1: Send the question to Ollama (running locally on your machine)
# =======================================================================
print("STEP 1: Sending question to Ollama...")

messages = [{"role": "user", "content": "What's the weather in Boston?"}]

try:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "tools": tools,
            "stream": False
        },
        timeout=60
    )
    response.raise_for_status()  # raises an error if something went wrong
    data = response.json()       # convert the response text into a dictionary
except Exception as e:
    print(f"Something went wrong calling Ollama: {e}")
    print("Is Ollama running? Try: ollama serve")
    raise SystemExit(1)

print("-" * 60)


# =======================================================================
# STEP 2: Check what the model decided to do.
#
# NEW THING (not Python, just Ollama's response shape):
# `data["message"]` is a dictionary. If the model wants to call a tool,
# it will have a key called "tool_calls" — a LIST of tool requests.
# This is the same idea as Claude's `.content` blocks, just shaped
# differently. Since `data` is a plain dictionary here (not a special
# object like Claude's), we access things with square brackets: data["message"]
# =======================================================================
message = data["message"]

if "tool_calls" in message and message["tool_calls"]:

    # Just grabbing the first tool call from the list — normal indexing
    tool_call = message["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    tool_args = tool_call["function"]["arguments"]  # this is a dictionary, e.g. {"city": "Tokyo"}

    print(f"STEP 2: Model wants to call '{tool_name}'")
    print(f"        with input: {tool_args}")
    print("-" * 60)

    # ===================================================================
    # STEP 3: WE run the real function ourselves.
    # ===================================================================
    print("STEP 3: Running our own function...")

    try:
        result = get_weather(tool_args["city"])
    except Exception as e:
        result = f"Error running the tool: {e}"

    print(f"        Result: '{result}'")
    print("-" * 60)

    # ===================================================================
    # STEP 4: Send that result back to Ollama so it can answer properly.
    # We add TWO new messages: the assistant's tool request, and our
    # tool's result (role "tool").
    # ===================================================================
    print("STEP 4: Sending result back to Ollama...")

    messages.append(message)  # the model's own turn (it asked for the tool)
    messages.append({
        "role": "tool",
        "content": result
    })

    try:
        final_response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": tools,
                "stream": False
            },
            timeout=60
        )
        final_response.raise_for_status()
        final_data = final_response.json()
    except Exception as e:
        print(f"Something went wrong on the second call: {e}")
        raise SystemExit(1)

    print("\nSTEP 5: Model's FINAL answer:")
    print(f"  {final_data['message']['content']}")

else:
    # This branch runs if the model decided it didn't need the tool at all
    print("Model answered directly, no tool needed:")
    print(f"  {message['content']}")