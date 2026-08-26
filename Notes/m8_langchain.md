# Module 8 — LangChain Hands-On

## Setup

```bash
pip install langchain langchain-openai
```

## The provider-agnostic LLM wrapper

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Explain agentic AI in one sentence.")
print(response.content)
```

`llm.invoke(...)` / `.content` replace raw `client.chat.completions.create(...)` / `.choices[0].message.content`. Same underlying API call — swapping `ChatOpenAI` for `ChatAnthropic` etc. barely changes the rest of the code.

## Tools: `@tool` decorator vs hand-written JSON schema

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    fake_data = {"Mumbai": "32°C, humid", "Goa": "30°C, humid"}
    return fake_data.get(city, "No data available")

@tool
def calculate_budget(total_budget: float, days: int) -> str:
    """Calculate a daily budget breakdown given total budget and number of days."""
    daily = total_budget / days
    return f"Daily budget: {daily:.0f} (Stay: {daily*0.4:.0f}, Food: {daily*0.3:.0f}, Activities: {daily*0.3:.0f})"
```

Docstring → tool `description`. Type hints → parameter schema. Replaces the ~15-line hand-written JSON dict from Module 2 for the same tool — this is the real time-saving.

## The modern agent: `create_agent`

Replaces the entire hand-written agent loop from Module 3.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather, calculate_budget],
    system_prompt="You are a helpful travel planning assistant."
)

result = agent.invoke({"messages": [{"role": "user", "content": "Plan a 4-day Goa trip with 20000 budget."}]})
print(result["messages"][-1].content)
```

What's hidden inside `.invoke()`: the `for step in range(max_steps)` loop, `json.loads(tool_call.function.arguments)`, and manually appending `{"role": "tool", "tool_call_id": ...}` messages — all the mechanics built by hand in Module 3.

**Note**: `create_agent` runs on the LangGraph runtime under the hood (as of the LangChain/LangGraph 1.0 joint release, Oct 2025) — LangChain is the ergonomic layer, LangGraph is the execution engine beneath it.

## Memory across turns

```python
result1 = agent.invoke({"messages": [{"role": "user", "content": "Plan a 4-day Goa trip with 20000 budget."}]})
history = result1["messages"]

history.append({"role": "user", "content": "Actually make it 5 days."})
result2 = agent.invoke({"messages": history})
```

Same principle as the hand-rolled `History` list — **frameworks don't remove the "resend the growing list" responsibility**, they just give you a slightly more structured container (`result["messages"]`) around it.

## Key Lessons

1. `@tool` decorator = schema auto-generation from function signature + docstring. Saves real boilerplate vs. hand-written JSON schemas.
2. `create_agent` = the same decide→act→observe→repeat loop from Module 3, just hidden inside `.invoke()`.
3. Even with a framework, you still own the message history — memory isn't automatic just because you're using LangChain.
4. `AgentExecutor` (LangChain's original agent abstraction) is now **deprecated** — `create_agent` (running on the LangGraph runtime) is the current recommended path.

## When LangChain alone is enough

Good fit: straightforward pipelines, rapid prototyping, single-agent tool use with no branching/loop-back behavior needed (i.e., anything shaped like Module 3's agent, or Module 4's fixed pipeline).

Not a good fit: complex control flow, multiple agents handing off to each other conditionally, needing to pause/resume/replay mid-execution — that's what LangGraph (Module 9) is for.
