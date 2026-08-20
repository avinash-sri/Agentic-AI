# Module 7 — Streaming Responses

## Core idea

The model generates text one token at a time regardless. Streaming sends each token to you *as it's produced*, instead of buffering the whole response — improves *perceived* speed, not total time.

## Basic streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a short paragraph about Sri Lanka."}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:  # some chunks (first/last) carry None
        print(delta, end="", flush=True)
print()
```

- `stream=True` → response becomes an **iterator** of small chunks instead of one object
- `chunk.choices[0].delta.content` → each chunk is a "diff" (only the new text), not the accumulated message so far
- Must guard `if delta:` — `None` chunks will crash concatenation/printing if not skipped

## Accumulating full text (needed to save into history)

```python
full_response = ""
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
        full_response += delta
```

`full_response` is equivalent to non-streamed `response.choices[0].message.content`.

## Streaming with tool calls (more complex)

Tool call `arguments` also stream as JSON fragments — must reassemble before `json.loads()`:

```python
collected_tool_calls = {}
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        for tc_chunk in delta.tool_calls:
            idx = tc_chunk.index  # distinguishes multiple simultaneous tool calls
            if idx not in collected_tool_calls:
                collected_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
            if tc_chunk.id:
                collected_tool_calls[idx]["id"] = tc_chunk.id
            if tc_chunk.function.name:
                collected_tool_calls[idx]["name"] += tc_chunk.function.name
            if tc_chunk.function.arguments:
                collected_tool_calls[idx]["arguments"] += tc_chunk.function.arguments
```

## Pragmatic pattern: only stream the final answer

Tool-calling turns aren't user-facing anyway, so streaming them adds complexity for no real UX benefit. **Common pattern**: run tool-calling turns normally (non-streamed); only stream once `message.tool_calls` is empty and it's time for the final answer.

```python
def run_trip_planner(messages, max_steps=5):
    for step in range(max_steps):
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            return stream_final_answer(messages[:-1])
        for tool_call in message.tool_calls:
            ...  # unchanged
    return "Max steps reached."

def stream_final_answer(messages):
    stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=True)
    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response += delta
    print()
    messages.append({"role": "assistant", "content": full_response})
    return full_response
```

Costs one extra API call (re-requesting the final turn with streaming) — acceptable simplicity tradeoff while learning.

## Key Lessons

1. Streaming is a display/UX concern, not a correctness concern — the final accumulated text is identical to the non-streamed version.
2. Tool-call streaming needs fragment reassembly keyed by `index` when multiple tool calls happen in one turn.
3. Sensible default: don't stream internal tool-calling steps, only the user-facing final answer.
