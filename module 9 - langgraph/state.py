from typing import TypedDict

class TripCodeState(TypedDict):
    task: str
    plan: str
    code: str
    review: str
    approved: bool