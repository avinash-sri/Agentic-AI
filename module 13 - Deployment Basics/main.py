main.py


from fastapi import FastAPI
from pydantic import BaseModel
import uuid

app = FastAPI()
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