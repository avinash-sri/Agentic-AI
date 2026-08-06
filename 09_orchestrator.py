from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

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


# --- Specialist agents (created fresh each call, so no leftover memory between tasks) ---

def call_planner(input):
    agent = Agent("You are a planning expert. Break the task into a short numbered list of steps. No code.")
    return agent.ask(input)

def call_coder(input):
    agent = Agent("You are an expert Python programmer. Write clean code per the plan. Output ONLY code.")
    return agent.ask(input)

def call_reviewer(input):
    agent = Agent("You are a strict senior code reviewer. Point out bugs and improvements concisely.")
    return agent.ask(f"Review this code:\n\n{input}")


# --- Describe these specialists as TOOLS the orchestrator can call ---

tools = [
    {
        "type": "function",
        "function": {
            "name": "call_planner",
            "description": "Delegate to a planning specialist to break a coding task into steps",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string", "description": "The coding task"}},
                "required": ["input"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_coder",
            "description": "Delegate to a coding specialist to write Python code given a task and/or plan",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string", "description": "The task and plan, combined into one string"}},
                "required": ["input"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_reviewer",
            "description": "Delegate to a review specialist to critique a piece of Python code",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string", "description": "The code to review"}},
                "required": ["input"]
            }
        }
    }
]

available_functions = {
    "call_planner": call_planner,
    "call_coder": call_coder,
    "call_reviewer": call_reviewer
}


def run_orchestrator(user_request, max_steps=8):
    messages = [
        {"role": "system", "content": (
            "You are an orchestrator managing a team of specialists: a planner, a coder, and a reviewer. "
            "For any coding request, delegate to them in a sensible order (usually plan -> code -> review), "
            "using their outputs to inform the next step. Once you have a final plan, code, and review, "
            "summarize everything into one final answer for the user."
        )},
        {"role": "user", "content": user_request}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(model="gpt-5.5", messages=messages, tools=tools)
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content  # orchestrator is done, has a final answer

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"[orchestrator delegates] {name}({str(args)[:80]}...)")

            result = available_functions[name](**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Max steps reached."


final = run_orchestrator("Write a Python function to check if a number is prime, and make sure it's well reviewed.")
# final = run_orchestrator("Write a syntax of intitializing a variable in python.")
# final = run_orchestrator("Just write me a quick one-liner to reverse a string.")
print("\n=== FINAL ANSWER ===\n", final)
