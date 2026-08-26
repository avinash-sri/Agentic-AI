# Module 9 — LangGraph Hands-On

## Setup

```bash
pip install langgraph langchain-openai
```

## Core idea: agent as a graph, not a loop

Instead of `for step in range(max_steps)` + `if/else` branching, define:
- **Nodes** — functions (LLM call, tool call, or plain logic)
- **Edges** — rules for what runs next, including **conditional edges**
- **State** — shared object flowing through every node; each node reads/updates it

```
planner → coder → reviewer ─(approved)→ END
             ▲___________________(needs revision)__|
```

## Defining state

```python
from typing import TypedDict

class TripCodeState(TypedDict):
    task: str
    plan: str
    code: str
    review: str
    approved: bool
```

## Defining nodes

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

def planner_node(state: TripCodeState):
    response = llm.invoke(f"Break this into a short numbered plan, no code: {state['task']}")
    return {"plan": response.content}

def coder_node(state: TripCodeState):
    prompt = f"Task: {state['task']}\nPlan: {state['plan']}\nWrite clean Python code. Output ONLY code."
    response = llm.invoke(prompt)
    return {"code": response.content}

def reviewer_node(state: TripCodeState):
    response = llm.invoke(f"Review this code, end with APPROVED or NEEDS_REVISION:\n\n{state['code']}")
    review_text = response.content
    return {"review": review_text, "approved": "APPROVED" in review_text}
```

Each node reads what it needs from `state`, returns only the piece it updates — LangGraph merges it automatically. No manual message-passing between functions like in Module 4's `call_planner`/`call_coder`/`call_reviewer`.

## Defining the graph, with a real loop-back

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(TripCodeState)
graph.add_node("planner", planner_node)
graph.add_node("coder", coder_node)
graph.add_node("reviewer", reviewer_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "coder")
graph.add_edge("coder", "reviewer")

def route_after_review(state: TripCodeState):
    return END if state["approved"] else "coder"  # loop back to coder if not approved

graph.add_conditional_edges("reviewer", route_after_review)

app = graph.compile()
```

## Running

```python
result = app.invoke({"task": "...", "plan": "", "code": "", "review": "", "approved": False})
```

## The honest comparison vs. Module 4's orchestrator

| | Module 4 orchestrator | LangGraph |
|---|---|---|
| Who decides control flow | The orchestrator LLM, via tool-calling reasoning | You, via explicit edges / conditional functions |
| Loop-back / retry | Possible but implicit, harder to guarantee | Explicit, visible in the graph definition |
| Debugging | Read through message history/prints | Inspect state at each node; graph is visualizable |
| Best for | Flexible, LLM-driven ordering | Guaranteed structure/control flow |

Neither is strictly better — the orchestrator pattern lets the **model** steer dynamically; LangGraph lets **you** encode flow explicitly and predictably.

## Key Lessons

1. Graphs replace implicit LLM-driven delegation with explicit, inspectable control flow — the trade is flexibility for predictability.
2. State (`TypedDict`) replaces manually threading text between functions — each node just reads/writes the piece it needs.
3. Conditional edges are how you express "if X, loop back; else, proceed" cleanly — the exact capability a flat `for` loop with tool-calling struggles to express reliably.
4. **Don't parse free-text LLM output for control-flow decisions via naive substring checks** (e.g., `"APPROVED" in review_text` — a response like "Not approved" would incorrectly match, since it contains "APPROVED" as a substring). Use structured/JSON output for anything a conditional edge depends on.

## Bugs I Hit / Notes

- Attempted to force a loop-back with a stricter reviewer prompt — didn't reliably trigger, likely due to the substring-matching fragility described above, or the model's review response not phrasing rejection the expected way. Understood the conditional-edge mechanism conceptually even without observing the loop directly — worth revisiting with structured output (`response_format={"type": "json_object"}`) for a more reliable test.
