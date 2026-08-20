# Module 5 — Frameworks & Production Concerns

## When to use a framework

Frameworks (LangGraph, CrewAI, OpenAI Agents SDK) automate the exact loop built by hand in Modules 2-4: schema generation, the `while` loop, message history, multi-agent handoffs.

| Situation | Recommendation |
|---|---|
| Learning / small personal project | Raw code — full understanding, direct debugging |
| Team project, standard patterns | A framework saves real time |
| Complex branching/looping multi-agent workflows | LangGraph (models agents as a graph/state machine) |
| OpenAI-specific, need tracing/evals built in | OpenAI Agents SDK |
| Fast prototyping, opinionated team-of-agents structure | CrewAI |

**Frameworks don't do anything you can't do yourself** — they save time and add production features, at the cost of an abstraction layer. Understanding the raw mechanics first means you can debug *through* the abstraction when it leaks.

## What OpenAI Agents SDK looks like

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    ...
    return f"{city}: {temp}°C, {desc}"

agent = Agent(name="Weather Assistant", instructions="You help with weather.", tools=[get_weather])
result = Runner.run_sync(agent, "What's the weather in Goa?")
print(result.final_output)
```

`@function_tool` auto-generates the schema from type hints/docstring. `Runner.run_sync` runs the entire agent loop internally — the same loop written by hand in Module 3.

## Guardrails

1. **Input guardrails** — filter/validate requests before they reach the agent (off-topic, prompt injection, PII)
2. **Output guardrails** — check the response before showing the user (policy violations, hallucination checks, format validation)
3. **Tool execution guardrails** — `max_steps` cap, `try/except` around execution, restricted `eval()` inputs, rate limiting, cost caps, human-in-the-loop approval for high-stakes actions (sending money/emails)

## Evaluation

- **Golden test sets** — fixed inputs + expected behavior, re-run whenever prompts/tools change
- **LLM-as-judge** — a separate LLM call grades output against criteria (imperfect but scalable)
- **Human review sampling** — periodic manual review of real conversations

Treat prompt changes like code changes: re-run the test set, don't just eyeball one example.

## Cost & Latency

- Every tool call = one round trip; model choice (`gpt-4o-mini` vs `gpt-4o`) can be a 10-20x cost difference
- Multi-agent multiplies cost (see Module 4)
- Streaming improves perceived latency (see Module 7)
- Prompt caching cuts cost/latency for repeated prompt prefixes (e.g., fixed system prompts)

## Key Lessons

1. A framework's `Runner.run_sync` (or equivalent) is the same loop you already wrote by hand — knowing the mechanics makes framework debugging much easier.
2. Multi-agent systems need real justification (distinct expertise, different models, parallelism) — not used by default.
