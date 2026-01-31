import os
from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END 
from langchain_ollama import ChatOllama


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
llm = ChatOllama(model="phi3")

def process(state: AgentState) -> AgentState:
    """"
    Process the input message and generate a response.
    """
    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}")
    state["messages"].append(AIMessage(content = response.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START,"process")
graph.add_edge("process", END)
agent = graph.compile()
conversation_history = []
user_input = input("Enter your message:")

def preprocess_context(conversation_history: List[Union[HumanMessage,AIMessage]]):
    if len(conversation_history) > 5:
        conversation_history.pop(0)
while user_input != "exit":
    conversation_history.append(HumanMessage(content = user_input))
    
    result = agent.invoke({
        "messages":conversation_history
    })
    conversation_history = result["messages"]
    user_input = input("Enter your message:")
 
with open("logging.txt","w") as file:
    file.write("Your Conversation log:\n")
    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content} \n\n")
    file.write("End of conversation")
print("Conversation saved to logging.txt")
    