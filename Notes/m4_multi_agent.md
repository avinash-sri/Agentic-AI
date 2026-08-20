# Module 4 — Multi-Agent Systems

## Why use multiple agents

1. **Separation of concerns** — one agent juggling too many jobs gets a bloated, self-conflicting prompt
2. **Different models per task** — cheap/fast model for simple steps, expensive reasoning model for hard ones
3. **Parallel work** — independent sub-agents can work simultaneously

**Honest tradeoff**: more agents = more cost (each hop is a full API call), more latency, harder debugging. **Start single-agent; split only when you've actually hit a real limitation.**

## Common patterns

- **Orchestrator → Workers**: one manager agent delegates to specialists, combines results
- **Pipeline**: fixed sequence, each agent's output feeds the next
- **Debate / Critique**: agents critique/iterate on each other's output

## Pattern 1: Fixed Pipeline (Planner → Coder → Reviewer)

Each specialist is just another `Agent` instance. **You** pass text between them explicitly — there's no hidden channel between agents.

```python
planner = Agent("You are a planning expert. Break the task into steps. No code.")
coder = Agent("You are an expert Python programmer. Output ONLY code.")
reviewer = Agent("You are a strict code reviewer. Point out bugs concisely.")

def run_pipeline(task):
    plan = planner.ask(task)
    code = coder.ask(f"Task: {task}\n\nPlan:\n{plan}")
    review = reviewer.ask(f"Review this code:\n\n{code}")
    return {"plan": plan, "code": code, "review": review}
```

Narrow, single-purpose system prompts (e.g., "output ONLY code, no explanation") are what make chaining reliable.

## Pattern 2: Dynamic Orchestrator (agents-as-tools)

**Key realization: "agent calling agent" = tool calling, where the tool happens to be another LLM.** Same loop mechanism as Module 3, nested.

```python
def call_planner(input):  # note: standardized param name across all tools, see lessons below
    agent = Agent("You are a planning expert...")
    return agent.ask(input)

def call_coder(input):
    agent = Agent("You are an expert Python programmer...")
    return agent.ask(input)

tools = [
    {"type": "function", "function": {"name": "call_planner", "parameters": {
        "properties": {"input": {"type": "string", "description": "The coding task"}}, "required": ["input"]}}},
    # ... call_coder, call_reviewer similarly
]

def run_orchestrator(user_request, max_steps=8):
    messages = [
        {"role": "system", "content": "You manage a planner, coder, and reviewer. "
                                       "Delegate in a sensible order (usually plan -> code -> review), "
                                       "but you may skip/reorder/repeat steps based on the specific task."},
        {"role": "user", "content": user_request}
    ]
    for step in range(max_steps):
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            return message.content
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            try:
                result = available_functions[name](**args)
            except TypeError as e:
                result = f"Error calling {name}: {e}. Please retry with correct argument name."
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    return "Max steps reached."
```

The system prompt gives a **suggested default order**, but the model has real flexibility to deviate (skip planner for trivial tasks, call reviewer anyway as a habit, etc.) — that flexibility is the actual value over a hardcoded pipeline, not "the model invents order from nothing."

## Key Lessons

1. **Standardize tool parameter names** (e.g., always `input`) across all specialist tools. Smaller models can hallucinate/confuse parameter names when different tools have differently-named slots for conceptually similar content (`task` vs `task_and_plan` vs `code`).
2. **Never fully trust the model calls a tool with exact schema.** Wrap tool execution in `try/except`; feed the error back to the model so it can self-correct, rather than crashing.
3. Order guidance in the system prompt (`"usually plan -> code -> review"`) controls sequencing; parameter naming controls argument correctness. These are separate concerns solved by separate fixes.
4. **Memory bug pattern**: if a function mutates a shared list in place (e.g. `messages.append(...)` inside a loop), the caller must NOT also append the same content afterward — pick exactly one place that "owns" writing to history.
5. **Tool-result ordering**: a `role: "tool"` message must always immediately follow the assistant message that requested it. If you skip appending the assistant's tool-call message, the API will reject the tool-result message that follows.

## Bugs I Hit

- `TypeError: call_coder() got an unexpected keyword argument 'task'` — model sent the wrong param name because different tools had inconsistent parameter names. Fixed by standardizing all tools to use `input`, plus `try/except` around execution.
- Double-appended the final assistant message (once inside the loop via `messages.append(message)`, once again in the caller) — bloated/duplicated history. Fixed by having exactly one place own the append.
- Removed the `messages.append(message)` line entirely while fixing the above, which broke tool-calling completely (`role: "tool"` message with no matching preceding tool-call request → API error). Fixed by keeping the append **unconditional**, right after getting the response, before checking `tool_calls`.
