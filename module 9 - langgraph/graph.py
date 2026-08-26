from langgraph.graph import StateGraph, END
from state import TripCodeState
from nodes import planner_node, coder_node, reviewer_node

graph = StateGraph(TripCodeState)

graph.add_node("planner", planner_node)
graph.add_node("coder", coder_node)
graph.add_node("reviewer", reviewer_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "coder")
graph.add_edge("coder", "reviewer")

# Conditional edge: THIS is the real loop-back your Module 4 orchestrator lacked
def route_after_review(state: TripCodeState):
    if state["approved"]:
        return END
    else:
        return "coder"  # send it back to the coder to fix, using the review as feedback

graph.add_conditional_edges("reviewer", route_after_review)

app = graph.compile()