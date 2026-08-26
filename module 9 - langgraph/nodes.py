from langchain_openai import ChatOpenAI
from state import TripCodeState
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

def planner_node(state: TripCodeState):
    response = llm.invoke(f"Break this into a short numbered plan, no code: {state['task']}")
    return {"plan": response.content}

def coder_node(state: TripCodeState):
    prompt = f"Task: {state['task']}\nPlan: {state['plan']}\nWrite clean Python code. Output ONLY code. makes sure to have one error atleast"
    response = llm.invoke(prompt)
    return {"code": response.content}

def reviewer_node(state: TripCodeState):
    response = llm.invoke(f"Review this code, and end your review with either APPROVED or NEEDS_REVISION:\n\n{state['code']}")
    review_text = response.content
    approved = "APPROVED" in review_text
    return {"review": review_text, "approved": approved}