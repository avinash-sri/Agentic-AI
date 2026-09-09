# Module 11 — Agent Design Patterns

These are named patterns for loop shapes already built in earlier modules — this module is about recognizing/naming them, not new mechanics.

## ReAct (Reason + Act)

Interleave reasoning with action, one step at a time: think → act → observe → think again.

```
Thought: I need the weather in Goa before planning.
Action: get_weather(city="Goa")
Observation: 30°C, humid
Thought: Now I need the budget.
Action: calculate_budget(total_budget=20000, days=5)
Observation: Daily: 4000...
Thought: I have enough info.
Final Answer: ...
```

**This is the Module 3 `run_agent` loop.** The original 2022 ReAct paper had the model write "Thought:"/"Action:" as parsed plain text; native function-calling (what you've been using since Module 2) is the modern, structured version of the same idea.

Can make the reasoning explicit even with tool calling:
```python
system_prompt = "Before calling any tool, briefly state your reasoning (one line), then call the tool."
```

## Reflection (self-critique before finalizing)

Model critiques its own output and revises before finalizing — either same agent in sequence, or a separate reviewer (Module 4.3's Reviewer, Module 9's LangGraph loop-back).

Simplest version, one agent "talking to itself":
```python
def reflect_and_improve(task):
    draft = llm.invoke(f"Complete this task: {task}").content
    critique = llm.invoke(f"Critique this response for accuracy/completeness. Task: {task}\nResponse: {draft}").content
    improved = llm.invoke(f"Task: {task}\nOriginal: {draft}\nCritique: {critique}\nWrite an improved response.").content
    return improved
```

Models are often better at spotting flaws in an *already-written* answer than avoiding them while writing the first draft.

**Tradeoff**: triples cost/latency for often-modest gains. Worth it for high-stakes outputs (code, important docs); often not worth it for casual queries.

## Plan-and-Execute

Write the full plan upfront, then execute step by step — replan only if something goes wrong. Separates "what to do" from "how to do each part."

```python
def plan_and_execute(task):
    plan_response = llm.invoke(f"Break this task into a numbered list of concrete steps: {task}").content
    steps = parse_numbered_list(plan_response)

    results = [run_agent(step) for step in steps]  # reuse Module 3's agent loop per step

    final = llm.invoke(f"Task: {task}\nStep results: {results}\nWrite a final combined answer.").content
    return final
```

## ReAct vs Plan-and-Execute

| | ReAct | Plan-and-Execute |
|---|---|---|
| Planning | Implicit, one step at a time | Explicit, upfront |
| Adapts to surprises | Naturally | Needs explicit replanning logic |
| Cost | One call per step | One planning call + one per step |
| Predictability | Lower | Higher — plan can be inspected/approved before execution |
| Best for | Exploratory, uncertain next steps | Clear-structure tasks, or wanting human approval before spending on execution |

## Where these already appeared in this course

| Pattern | Built in... |
|---|---|
| ReAct | Module 3 (`run_agent`), Module 8 (`create_agent`) |
| Reflection | Module 4.3 (Reviewer), Module 9 (LangGraph loop-back) |
| Plan-and-Execute | Module 4.3 (Planner→Coder→Reviewer), Module 4.4 (orchestrator) |

## Key Lessons

1. These patterns are vocabulary for shapes already discovered by solving real problems — not new things to build from scratch.
2. Naming them helps with reading papers/frameworks/job postings, and with deliberately choosing a pattern up front rather than backing into one.
3. The real design choice is usually **implicit vs explicit control flow** (ReAct/orchestrator = model decides step-by-step; Plan-and-Execute/LangGraph = structure decided upfront or encoded explicitly) — same tradeoff seen in Module 9's orchestrator-vs-graph comparison.
