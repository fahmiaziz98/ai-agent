from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated, List, Literal

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    sql_query: str
    error_str: str
    question_type: str