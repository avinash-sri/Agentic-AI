from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

# A real function that actually does something
def get_weather(city):
    # Fake data for now — in Module 2.2 we'll hit a real API
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
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. Mumbai"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

messages = [{"role": "user", "content": "What's the weather in Mumbai right now?"}]

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=messages,
    tools=tools  # <-- this is new
)

message = response.choices[0].message
print(message)