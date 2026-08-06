from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

def get_weather(city):
    fake_data = {
        "Mumbai": "32°C, humid",
        "Delhi": "38°C, sunny",
        "London": "18°C, rainy"
    }
    return fake_data.get(city, "No data available for this city")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name, e.g. Mumbai"}
                },
                "required": ["city"]
            }
        }
    }
]

# Map tool names (strings) to actual Python functions
available_functions = {
    "get_weather": get_weather
}

messages = [{"role": "user", "content": "What's the weather in London right now?"}]

# --- STEP 1: First call — model decides whether to use a tool ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

message = response.choices[0].message
messages.append(message)  # save the model's tool-call request into history

# --- STEP 2: Check if the model actually wants to call a tool ---
if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)  # arguments come back as a JSON string!

        print(f"Model wants to call: {function_name}({function_args})")

        # --- STEP 3: Actually run the real Python function ---
        function_to_call = available_functions[function_name]
        result = function_to_call(**function_args)

        # --- STEP 4: Send the result back, linked to the tool_call id ---
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

    # --- STEP 5: Call the model again — now it has the tool's result ---
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    print("Final answer:", final_response.choices[0].message.content)

else:
    # Model answered directly without needing a tool
    print("Direct answer:", message.content)