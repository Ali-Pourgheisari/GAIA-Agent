
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


model = ChatOpenAI(model="gpt-4o", temperature=0.0)

def model_node(state: AgentState) -> dict:
    response = model.invoke(state["messages"])

    return {"messages": [response]}

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


tools = [multiply]
model = model.bind_tools(tools)
tool_node = ToolNode(tools)