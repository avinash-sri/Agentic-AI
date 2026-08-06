
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

class Agent:
    def __init__(self, system_prompt, model="gpt-4o-mini", max_steps=3):
        self.model = model
        self.max_steps = max_steps
        self.messages = [{"role": "system", "content": system_prompt}]

    def ask(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# --- Define the three specialist agents ---

planner = Agent(system_prompt=(
    "You are a software planning expert. Given a coding task, break it into "
    "a short, clear numbered list of implementation steps. Do NOT write code — "
    "only the plan."
))

coder = Agent(system_prompt=(
    "You are an expert Python programmer. You will be given a task and a plan. "
    "Write clean, working Python code that follows the plan. "
    "Output ONLY the code, no explanations."
))

reviewer = Agent(system_prompt=(
    "You are a strict but constructive senior code reviewer. Given a piece of Python code, "
    "point out bugs, edge cases, and improvements. Be specific and concise. "
    "If the code is genuinely good, say so briefly."
))


def run_coding_pipeline(task):
    print("=" * 50)
    print("TASK:", task)
    print("=" * 50)

    # Step 1: Planner breaks down the task
    plan = planner.ask(task)
    print("\n--- PLAN ---\n", plan)

    # Step 2: Coder writes code, given task + plan
    code = coder.ask(f"Task: {task}\n\nPlan:\n{plan}")
    print("\n--- CODE ---\n", code)

    # Step 3: Reviewer critiques the code
    review = reviewer.ask(f"Review this code:\n\n{code}")
    print("\n--- REVIEW ---\n", review)

    return {"plan": plan, "code": code, "review": review}


result = run_coding_pipeline(
    # "Write a Python function for calculating the factorial of a number using recursion."
    "Write a one liner to add two variables"
)