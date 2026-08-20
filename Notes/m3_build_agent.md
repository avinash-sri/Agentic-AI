# Module 3 — Building a Reusable Agent

## The general-purpose agent loop

One round of tool-calling isn't enough for real tasks — the model may need several rounds (call tool A, look at result, call tool B, etc.) before it can answer.

```python
def run_agent(user_input, max_steps=5):
    messages = [{"role": "user", "content": user_input}]

    for step in range(max_steps):
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content  # stopping condition: no more tools needed

        for tool_call in message.tool_calls:  # model may request multiple tools in ONE turn
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = available_functions[name](**args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})

    return "Max steps reached without a final answer."
```

- `max_steps` is a **safety cap** — without it, a confused agent could loop forever burning API credits. Every real framework has some version of this.
- Model can call multiple tools in a single turn (e.g., weather for two cities at once) — must loop over all of them before continuing.

## Wrapping it in a reusable `Agent` class

```python
class Agent:
    def __init__(self, system_prompt="You are a helpful assistant.", model="gpt-4o-mini", max_steps=5):
        self.model = model
        self.max_steps = max_steps
        self.messages = [{"role": "system", "content": system_prompt}]  # memory lives HERE, persists across .ask() calls
        self.tools = [...]
        self.available_functions = {...}

    def ask(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        for step in range(self.max_steps):
            response = client.chat.completions.create(model=self.model, messages=self.messages, tools=self.tools)
            message = response.choices[0].message
            self.messages.append(message)
            if not message.tool_calls:
                return message.content
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = self.available_functions[name](**args)
                self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
        return "Max steps reached."
```

`self.messages` persisting across calls is what lets a later `.ask()` reference earlier results (e.g., "add 10 to the temperature you just told me").

## Real tools: calculator + web search

- **Calculator**: LLMs are unreliable at arithmetic — always delegate real math to a tool, never trust the model to compute it in free text.
  ```python
  def calculator(expression):
      allowed_chars = "0123456789+-*/(). "
      if not all(c in allowed_chars for c in expression):
          return "Error: invalid characters"
      return str(eval(expression))
  ```
  ⚠️ Never `eval()` raw model output without a whitelist guard — prompt injection risk. Production: use `numexpr`/`asteval` instead of `eval`.

- **Web search** (Tavily or DuckDuckGo/`ddgs`) — lets the agent answer things after its training cutoff.
  ```python
  from ddgs import DDGS
  def web_search(query):
      with DDGS() as ddgs:
          results = list(ddgs.text(query, max_results=3))
      return "\n".join(f"- {r['title']}: {r['body'][:200]}..." for r in results)
  ```

## Grounding the agent in "now"

Models don't know the current date on their own. For anything time-sensitive:
```python
from datetime import date
today = date.today().strftime("%B %d, %Y")
system_prompt = (
    f"Today's date is {today}. You do NOT know events after your training cutoff. "
    f"Always use web_search for current events/results, and trust search results over your own memory."
)
```

## Key Lessons

1. Adding a new tool = 3 steps only: (1) write the function, (2) describe it in `tools`, (3) register it in `available_functions`. The loop logic itself never changes.
2. `self.messages` as an instance attribute is what turns a script into something with an ongoing "session."
3. Package APIs change — hit a `duckduckgo_search` → `ddgs` rename mid-project; a `RuntimeWarning` (not an Error) means "works now, fix before it breaks."

## Bugs I Hit

- `duckduckgo_search` deprecated in favor of `ddgs` — fixed by swapping the import.
- Agent answered "2023 F1 champion" despite using web search for "most recent champion" — root cause: no anchor for "now," and vague search query. Fixed by injecting today's date + explicit instruction to trust search over memory.
