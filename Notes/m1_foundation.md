# Module 1 — Foundations

## Concept

- A plain LLM call answers once from memory. An **agent** = LLM + tools + a loop that keeps going until a goal is done.
- The core API call every agent is built on:

```python
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain agentic AI in one sentence."}]
)
print(response.choices[0].message.content)
```

## The `messages` list — 3 roles

| Role | Purpose |
|---|---|
| `system` | Sets behavior/personality, sent once at the start |
| `user` | What the human says |
| `assistant` | What the LLM said — must be re-added to history so the model "remembers" its own replies |

## Multi-turn memory (manual)

**The LLM has no memory of its own.** Every API call is independent. "Memory" = you keep a growing list of messages and resend all of it every time.

```python
conversation = [{"role": "system", "content": "You are a friendly assistant."}]

def chat(user_input):
    conversation.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=conversation)
    reply = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})  # easy to forget!
    return reply
```

## Chat Completions vs Responses API

- **Chat Completions** (`client.chat.completions.create`) — the "classic" API, you manage the `messages` array and history yourself. Still fully supported.
- **Responses API** (`client.responses.create`) — newer, can be stateful server-side (`previous_response_id`), has built-in tools (web search, file search, code execution), OpenAI's recommended default for new projects.
- **Why learn Chat Completions first**: the `messages`/roles concept is universal across providers, and manually managing the agent loop builds the mental model that frameworks/Responses API hide from you later.

## Key Lessons

1. Memory is just a list you resend — nothing magic.
2. Forgetting to append the assistant's own reply breaks multi-turn memory silently.
3. Every provider uses some version of role-based messages, even if wire format differs.

## Bugs I Hit

- Ran two separate calls expecting the model to remember a name from the first — it didn't, because I never sent the first exchange back in the second call.