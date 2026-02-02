import os
from urllib.parse import unquote
from pathlib import Path
from typing import TypedDict, Sequence, Annotated
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import InjectedState, ToolNode

llm = ChatOllama(model="qwen2.5:3b")

IGNORE_DIRS = {
    ".git",
    ".github",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".next",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
}

IGNORE_FILES = {
    ".DS_Store",
    "package-lock.json",
    "yarn.lock",
}
MAX_CHARS = 2000

class AgentState(TypedDict):
    root_path : str
    messages: Annotated[Sequence[BaseMessage],add_messages]



@tool
def scan_repo(state: Annotated[AgentState,InjectedState]) -> str:
    """
        Returns the folder tree structure of the repository.
    """
    root_path = state["root_path"]
    root = Path(root_path)
    tree = ""
    for path in sorted(root.rglob("*")):
        if any(part in IGNORE_DIRS  for part in path.parts):
            continue
        if path.name in IGNORE_FILES:
            continue
        rel = path.relative_to(root)
        indent = "    " * (len(rel.parts) - 1)
        if path.is_dir():
            
            tree += f"{indent}{path.name}/\n"
        else:
            tree += f"{indent}{path.name}\n"

    return tree

@tool
def read_file(file_path: str, state: Annotated[AgentState, InjectedState]) -> str:
    """     
        Reads a file and returns its contents, truncated to MAX_CHARS.

    """
    full_path = Path(state["root_path"]) / file_path
    if not full_path.exists():
        return "File not found."
    try:
        return full_path.read_text(encoding="utf-8")[:MAX_CHARS]
    except:
        return "Could not read file."

@tool
def write_readme(state:Annotated[AgentState,InjectedState], content: str) -> str:
    """
    Writes README.md to the repository root.
    """
    root_path = state["root_path"]
    out = Path(root_path) / "README.md"
    out.write_text(content, encoding="utf-8")
    return f"README.md written successfully at {out}"

tools = [scan_repo, read_file, write_readme]
llm = llm.bind_tools(tools)
def our_agent(state:AgentState) -> AgentState:
    SYSTEM_PROMPT = """
You are Documenter, a repository documentation agent.

Goal:
- Understand the repository structure using tools
- Explain folders based on actual file contents
- Generate a professional README.md

Rules:
- Always call scan_repo first
- Use read_file for important modules
- Do not hallucinate
- Output must be clean GitHub Markdown
- When README is ready, call write_readme
"""
    system_prompt = SystemMessage(
        content = SYSTEM_PROMPT
    )
    if not state.get("root_path"):
        user_input = input("Enter the repository path: ").strip()
        state["root_path"] = user_input
        user_message = HumanMessage(content=f"My repository path is: {user_input}")
        print(f"\n👤 USER: {user_input}")
       
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = llm.invoke(all_messages)
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    return {
        "root_path": state["root_path"],
        "messages":list(state["messages"])+[user_message,response]
    }


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")
            
graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools",ToolNode(tools))
graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")

agent = graph.compile()
def run_document_agent():
    print("\n ===== DOCUMENTER =====")
    
    state = {"messages": []}
    
    for step in agent.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DOCUMENTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()