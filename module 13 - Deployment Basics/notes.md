# Module 13 — Deployment Basics

## Wrapping an agent in an API (FastAPI)

```bash
pip install fastapi uvicorn
```

```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
app = FastAPI()

class PlanRequest(BaseModel):
    message: str

History = [{"role": "system", "content": "You are a travel planning assistant..."}]

@app.post("/plan")
def plan_trip(request: PlanRequest):
    History.append({"role": "user", "content": request.message})
    result = run_trip_planner(History)  # existing function, unchanged
    return {"response": result}

@app.get("/health")
def health_check():
    return {"status": "ok"}
```

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for an auto-generated interactive test UI. Core idea: the agent logic doesn't change — you just give the outside world a network door (`POST /plan`) to call it through.

## Per-user state (session isolation)

A single global `History` list is broken with multiple concurrent users — everyone reads/writes the same conversation. Fix: key history by session.

```python
import uuid

sessions = {}  # session_id -> message history list

class PlanRequest(BaseModel):
    message: str
    session_id: str | None = None

@app.post("/plan")
def plan_trip(request: PlanRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": "You are a travel planning assistant..."}]
    sessions[session_id].append({"role": "user", "content": request.message})
    result = run_trip_planner(sessions[session_id])
    return {"response": result, "session_id": session_id}
```

Client gets a `session_id` on first call, sends it back on follow-ups. Same "resend the growing list" principle from Module 1, now keyed per-user.

**Limitation**: `sessions = {}` is still in-memory — restarting the server loses all conversations. Production: store in Redis/a database (same persistence motivation as Module 12's move from list → Chroma).

## Secrets management

- `.env` + `load_dotenv()` is for **local development only** — never deploy a `.env` file (never commit it, never copy it to a server).
- Production: set environment variables directly on the hosting platform's dashboard/CLI. Code stays the same:
  ```python
  import os
  api_key = os.environ.get("OPENAI_API_KEY")
  ```
  `load_dotenv()` becomes a no-op in production (no `.env` file exists there) — the platform already set the env vars, so `os.environ.get(...)` finds them directly.
- **Never hardcode keys** (`OpenAI(api_key="sk-...")`) — bots scan public GitHub repos for exposed keys within minutes of a push.

## Logging

`print()` statements aren't visible once something runs unattended in production — need persistent, searchable logs.

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="agent.log")
logger = logging.getLogger(__name__)

@app.post("/plan")
def plan_trip(request: PlanRequest):
    logger.info(f"Received request: session={request.session_id}, message={request.message[:100]}")
    try:
        result = run_trip_planner(...)
        logger.info(f"Successfully responded to session={request.session_id}")
        return {"response": result}
    except Exception as e:
        logger.error(f"Failed for session={request.session_id}: {e}")
        return {"error": "Something went wrong. Please try again."}
```

Log only what's needed to debug (truncated messages, session IDs, error types) — never full conversations or secrets.

## Rate limiting

Public endpoints are callable by anyone who finds the URL — every call costs real API credits.

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/plan")
@limiter.limit("5/minute")
def plan_trip(request: PlanRequest):
    ...
```

## Deploying to the internet (overview)

- **Render / Railway** — connect GitHub repo, set env vars in dashboard, minimal config — good starting point
- **AWS / GCP / Azure** — more control/setup, better for scaling later
- Common pattern across all: push code, set env vars in the platform's UI (never in code), platform runs `uvicorn main:app` for you

## Key Lessons

1. Deployment = giving existing logic a network entry point (an HTTP route) — the agent code itself doesn't need to change.
2. Global mutable state (a shared `History` list) breaks the moment there's more than one concurrent user — session keys are the fix, same principle threading through Modules 1, 6, and 12 (state persistence is always an explicit design decision).
3. Secrets belong in the hosting platform's environment variable settings, never in a deployed `.env` file or hardcoded in source.
4. Logging and rate limiting aren't optional polish — they're the difference between "works on my machine" and "safe to expose publicly."

## Verified

Ran the FastAPI wrapper locally via `uvicorn`, tested through `/docs`, confirmed two different `session_id`s maintain separate, non-interfering conversation histories.
