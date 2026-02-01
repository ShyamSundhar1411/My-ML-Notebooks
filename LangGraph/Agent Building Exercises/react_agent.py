from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """Adds two numbers together"""
    return a + b


tools = [add]

model = ChatOllama(model="qwen2.5:3b").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, answer queries accurately."
    )

    response = model.invoke([system_prompt] + list(state["messages"]))

    return {
        "messages": state["messages"] + [response]
    }


def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return "end"
    return "continue"


graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_edge("tools", "our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        message.pretty_print()


inputs = {
    "messages": [
        ("user", "what is 5 + 5?")
    ]
}

print_stream(app.stream(inputs, stream_mode="values"))
