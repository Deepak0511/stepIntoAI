from typing import TypedDict
from typing import List


class GraphState(TypedDict):

    user_query: str
    keywords: List[str]
    research_notes: str
    summary: str
    next_step: str
    review_feedback: str
    iteration_count: int