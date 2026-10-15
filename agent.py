
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


model = ChatOpenAI(model="gpt-4o", temperature=0.0)

def model_node(state: AgentState) -> dict:
    response = model.invoke(state["messages"])

    return {"messages": [response]}

def route_after_model(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


tools = [multiply]
model = model.bind_tools(tools)
tool_node = ToolNode(tools)

builder = StateGraph(AgentState)
builder.add_node("model", model_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "model")
builder.add_edge("tools", "model")
builder.add_conditional_edges("model", route_after_model)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({
        "messages": [
            HumanMessage(
                content="Use the multiply tool to calculate 12 times 8"
            )
        ]
    })
for message in result["messages"]:
    message.pretty_print()
