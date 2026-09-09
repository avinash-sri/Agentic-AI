# Module 14 — Evaluation, Hands-On

## Why a test set matters

Manually eyeballing one output per prompt change catches obvious breaks but misses subtle regressions — a fix for one case can silently break another you didn't happen to test that day. A test set turns "try a few inputs and eyeball it" into "run N known cases automatically every time."

## Building a golden test set

```python
test_cases = [
    {
        "input": "Plan a 3-day trip to Goa with a budget of 15000 rupees.",
        "expected_tools_called": ["get_weather", "calculate_budget", "search_attractions"],
        "must_mention": ["Goa", "budget"],
    },
    {
        "input": "What's 2+2?",
        "expected_tools_called": [],  # should NOT call trip-planning tools for unrelated input
        "must_mention": ["4"],
    },
    {
        "input": "Plan a trip to Mars.",
        "expected_tools_called": ["get_weather"],
        "must_mention": [],  # just checking it doesn't crash
    },
]
```

Testing that the agent correctly does **nothing special** for unrelated input (case 2) is as important as testing tool use when expected — an easy case to forget manually since "nothing happens" doesn't feel like something to verify.

## Automated (code-only) checks — fast, free, deterministic

```python
def check_tools_called(actual_tool_calls, expected_tools):
    return set(actual_tool_calls) == set(expected_tools)

def check_keywords_present(response_text, must_mention):
    return all(k.lower() in response_text.lower() for k in must_mention)
```

Modify `run_trip_planner` to also return which tools were called:
```python
def run_trip_planner(messages, max_steps=5):
    called_tools = []
    for step in range(max_steps):
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            return message.content, called_tools
        for tool_call in message.tool_calls:
            called_tools.append(tool_call.function.name)
            ...
    return "Max steps reached.", called_tools
```

## LLM-as-judge — for qualities code can't check

```python
def llm_judge(user_input, agent_response, criteria):
    judge_prompt = f"""You are evaluating an AI travel assistant's response.
User asked: {user_input}
Assistant responded: {agent_response}
Evaluate on: {criteria}
Respond with ONLY JSON: {{"score": 1-5, "reasoning": "brief explanation"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"}  # forces valid, parseable JSON
    )
    return json.loads(response.choices[0].message.content)
```

`response_format={"type": "json_object"}` is the fix referenced back in Module 9's fragile `"APPROVED" in review_text` substring bug — structured output gives a reliable field to threshold on instead of parsing free text.

## Test runner

```python
def run_test_suite(test_cases):
    results = []
    for case in test_cases:
        messages = [{"role": "system", "content": "..."}, {"role": "user", "content": case["input"]}]
        response_text, called_tools = run_trip_planner(messages)
        tools_ok = check_tools_called(called_tools, case["expected_tools_called"])
        keywords_ok = check_keywords_present(response_text, case["must_mention"])
        judge_result = llm_judge(case["input"], response_text, "Is this helpful, accurate, and relevant?")
        results.append({"input": case["input"], "tools_ok": tools_ok, "keywords_ok": keywords_ok,
                         "judge_score": judge_result["score"], "judge_reasoning": judge_result["reasoning"]})
    return results
```

## The actual workflow this enables

```python
baseline_results = run_test_suite(test_cases)
# ... make a prompt/tool change ...
new_results = run_test_suite(test_cases)

for baseline, new in zip(baseline_results, new_results):
    if baseline["judge_score"] != new["judge_score"] or baseline["tools_ok"] != new["tools_ok"]:
        print(f"REGRESSION/CHANGE DETECTED: {baseline['input'][:50]}")
```

This is the concrete mechanism behind "treat prompt changes like code changes" (Module 5).

## Honest limitations

1. **LLM-as-judge has its own bias/inconsistency** — a scalable proxy, not ground truth; can be gamed by responses that "sound" good without being substantively correct.
2. **A test set only catches what you thought to include** — doesn't catch unanticipated failure modes. Production systems still need periodic human review sampling alongside automated eval.
3. Running the suite costs real API calls (one per case + one judge call per case) — worth sizing the suite deliberately, especially for frequent/CI usage.

## Key Lessons

1. Deterministic checks (tool names, keywords) and LLM-as-judge checks solve different problems — combine both rather than relying on either alone.
2. Testing "should NOT trigger X" cases is as important as testing "should trigger X" cases.
3. Structured output (`response_format={"type": "json_object"}`) is the general fix for any evaluation or control-flow logic that depends on parsing an LLM's free-text response reliably.
