from openai import OpenAI
from dotenv import load_dotenv
import json
from ddgs import DDGS

load_dotenv()
client = OpenAI()


def calculator(expression):
    try:
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "Error: invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def get_weather(city):
    fake_data = {"Mumbai": "32°C, humid", "Delhi": "38°C, sunny", "London": "18°C, rainy"}
    return fake_data.get(city, "No data available for this city")

def web_search(query):                    
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    formatted = [f"- {r['title']}: {r['body'][:200]}..." for r in results]
    return "\n".join(formatted)


class Agent:
    def __init__(self, system_prompt="You are a helpful assistant.", model="gpt-4o-mini", max_steps=5):
        self.model = model
        self.max_steps = max_steps

        # Memory lives HERE, as an instance attribute — persists across .ask() calls
        self.messages = [{"role": "system", "content": system_prompt}]

        # Tool definitions the model sees
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a basic math expression, e.g. '23*7' or '(4+5)/3'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "A math expression to evaluate"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name, e.g. Mumbai"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for current, up-to-date, or real-time information not in your training data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # Map names -> real Python functions to execute
        self.available_functions = {
            "calculator": calculator,
            "get_weather": get_weather,
            "web_search": web_search
        }

    def ask(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        for step in range(self.max_steps):
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools
            )
            message = response.choices[0].message
            self.messages.append(message)

            if not message.tool_calls:
                return message.content

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"  [tool] {name}({args})")

                result = self.available_functions[name](**args)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        return "Max steps reached."


# --- Using the agent across multiple turns ---
agent = Agent(system_prompt=(
    "You are a helpful assistant. You do NOT know current events after your training cutoff. "
    "Always use the web_search tool for anything about recent results, current champions, "
    "prices, or events — never answer from memory for these."
    "The current year is 2026"
))

print(agent.ask("What's 234 * 88?"))
print(agent.ask("What's the weather in London?"))
print(agent.ask("Add 10 to the temperature you just told me, ignoring units."))
print(agent.ask("Solve 4^2 + 3^2 - 5*6"))
print(agent.ask("Who won the most recent F1 championship"))