
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

available_functions = {"get_weather": get_weather}

def run_agent(user_input, max_steps=5):
    messages = [{"role": "user", "content": user_input}]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message
        messages.append(message)

        # If the model didn't ask for any tool, it's done — return the answer
        if not message.tool_calls:
            return message.content

        # Otherwise, run EVERY requested tool call this round
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"[Step {step+1}] Calling {function_name}({function_args})")

            function_to_call = available_functions[function_name]
            result = function_to_call(**function_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        # loop continues — model gets another turn, now with fresh tool results

    return "Max steps reached without a final answer."

answer = run_agent("What's the weather in Mumbai and Delhi? Which one is hotter?")
print("\nFinal answer:", answer)

# You've now built, from scratch, the exact mechanism that powers every "AI agent" product you've heard of. 
# Everything from here is refinement, not reinvention.