import json
import os

import httpx
from openai import OpenAI
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()

client = OpenAI()

OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5/weather"

class Agent:
    def __init__(self, system_prompt, model="gpt-4o-mini"):
        self.model = model
        self.messages = [{"role": "system", "content": system_prompt}]

    def ask(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(model=self.model, messages=self.messages)
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


def call_weather(city):
    key = os.environ.get("OPENWEATHER_API_KEY", "")
    params = {"q": city, "appid": key, "units": "metric"}
    response = httpx.get(OPENWEATHER_BASE, params=params, timeout=10.0)

    data = response.json()

    if response.status_code != 200:
        return f"Error fetching weather data: {data.get('message', 'Unknown error')}"

    temperature = data["main"]["temp"]
    description = data["weather"][0]["description"]
    return f"The current weather in {city} is {temperature}°C with {description}."

def call_budget(total_budget, days):
    daily_expense = total_budget / days
    agent = Agent("You are a travel assistant. Calculate the total budget for a trip given the number of days and daily expenses.")
    stay = daily_expense * 0.40
    food = daily_expense * 0.30
    activities = daily_expense * 0.30

    agent.messages.append({"role": "assistant", "content": f"Budget breakdown: Stay: ${stay:.2f}, Food: ${food:.2f}, Activities: ${activities:.2f}"})

    return agent.ask(f"Calculate the budget for a trip of {days} days with a daily expense of ${daily_expense}.")

def call_attractions(city):
    agent = Agent("You are a travel assistant. List popular attractions in the specified city.")
    query: str = f"popular attractions in {city}"
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    formatted = [f"- {r['title']}: {r['body'][:200]}..." for r in results]
    agent.messages.append({"role": "assistant", "content": f"Attractions data: {formatted}"})

    return agent.ask(f"Search for popular attractions in {city}.")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters":{
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to get the weather for"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_budget",
            "description": "Calculate the budget for a trip given the number of days and daily expenses",
            "parameters": {
                "type": "object",
                "properties": {
                    "total_budget": {"type": "number", "description": "Total trip budget"},
                    "days": {"type": "integer", "description": "Number of days"}
                },
                "required": ["days", "total_budget"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "Search for popular attractions in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to search for attractions"}
                },
                "required": ["city"]
            }
        }
    }
]

available_functions = {
    "get_weather": call_weather,
    "calculate_budget": call_budget,
    "search_attractions": call_attractions
}

def run_trip_planner(messages, max_steps=5):

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"[trip planner] {name}({str(args)[:80]}...")

            result = available_functions[name](**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Max steps reached without a final answer."

        

History = [{"role": "system", "content": "You are a travel planning assistant. You can provide weather information, calculate budgets, and suggest attractions for a trip. Use the available tools to assist the user with their travel planning."}]

def plan(user_request):
    History.append({"role": "user", "content": user_request})
    final_messages = run_trip_planner(History)
    return final_messages


# print(plan("Plan south-goa trip for 4 days with 20000 indian rupees budget. Also provide weather and attractions information for goa."))
# print(plan("Actually Plan it for 5 days"))

print(plan("Plan a trip to Goa for 4 days with a budget of 15000 Indian rupees. Also provide weather and attractions information for Goa."))
print(plan("We're planning transportation by rented scooty, meals at local restaurants, and we want to go on butterfly beach and arambol beach. Please provide a detailed itinerary for the trip including stay in South Goa and North Goa, including daily activities, estimated costs, and any tips for making the most of our time in Goa."))

