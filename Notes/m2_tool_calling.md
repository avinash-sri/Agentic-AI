# Module 2 — Tool / Function Calling

## Concept

The LLM never executes anything — it only generates text. Tool calling works like this:

1. You describe available tools (name, description, JSON-schema args) to the model
2. Model outputs a structured "I want to call tool X with these args" message instead of a normal reply
3. **Your code** actually runs the function
4. Result is sent back to the model as a new message
5. Model uses the result to write the final answer

```
User asks something
   → LLM decides: "I need to call get_weather(city='Mumbai')"
   → Your code actually calls get_weather("Mumbai")
   → Result sent back to LLM
   → LLM writes final answer using that result
```

## Defining a tool

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name, e.g. Mumbai"}
            },
            "required": ["city"]
        }
    }
}]
```

**The `description` fields matter a lot** — the model decides *when* and *how* to call a tool based on these. Vague descriptions = bad tool usage (over- or under-triggering).

## Executing the tool call (full loop, one round)

```python
import json

available_functions = {"get_weather": get_weather}
messages = [{"role": "user", "content": "What's the weather in Mumbai?"}]

response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
message = response.choices[0].message
messages.append(message)  # save the model's tool-call request

if message.tool_calls:
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)  # always a JSON string — must parse
        result = available_functions[name](**args)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,  # links result to the specific request
            "content": result
        })

    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    print(final.choices[0].message.content)
else:
    print(message.content)  # model answered directly, no tool needed
```

## Key Lessons

1. `tool_call.function.arguments` is always a **JSON string** — must `json.loads()` it.
2. New 4th message role: `"tool"` — must include `tool_call_id` to link back to the request.
3. Must append the assistant's tool-call message to history *before* appending the tool result, or the model gets confused (tool result with no matching request).
4. Always check `if message.tool_calls` — not every input needs a tool.
5. **A tool call being made ≠ the model trusting/using the result correctly.** Verify the final answer actually reflects the tool's output, especially for time-sensitive facts (model's training-data bias can override tool results, e.g. answering "2023 F1 champion" despite searching for "most recent").

## Bugs I Hit

- Forgot `json.loads()` — got a string instead of usable arguments.
- Model called a tool but ignored the result because the query ("most recent") was ambiguous about time — fixed by injecting today's date into the system prompt and explicitly telling the model to trust search over memory for time-sensitive facts.
