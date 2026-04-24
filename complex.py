from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from IPython.display import Image, display

llm = init_chat_model(
    "ollama:qwen3:8b",   # replace with your exact Ollama model tag
    temperature=0,
)

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(..., 
            description="Classify if the message requires an emotional (therapist) or logical response")



class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)
    classification = classifier_llm.invoke([{
        "role": "system",
        "content": """Classify the user message as either:
- emotional: if the message expresses feelings, emotions, or personal struggles that would benefit from empathy and support.
- logical: if the message is more focused on facts, reasoning, or problem-solving that would benefit from a clear and rational response.
"""    }, {
        "role": "user",
        "content": last_message.content
    }])
    print('-------------------')
    print(classification)
    print('--------------------')
    return {"message_type": classification.message_type}

def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        print("Routing to therapist agent")
        return {"next": "therapist"}
    print("Routing to logical agent")
    return {"next": "logical"}

def therapist_agent(state: State):
    print("Therapist agent invoked")
    last_message = state["messages"][-1]
    response = llm.invoke([{
        "role": "system",
        "content": """You are a compassionate therapist. Respond to the user's message with empathy, understanding, and support. Focus on addressing their feelings and emotions."""    
    }, {
        "role": "user",
        "content": last_message.content
    }])
    return {"messages": [{"role": "assistant", "content": response.content}]}

def logical_agent(state: State):
    last_message = state["messages"][-1]
    response = llm.invoke([{
        "role": "system",
        "content": """You are a logical assistant. Respond to the user's message with clear reasoning, facts, and problem-solving advice. Focus on addressing the logical aspects of their message."""    
    }, {
        "role": "user",
        "content": last_message.content
    }])
    return {"messages": [{"role": "assistant", "content": response.content}]}



graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges("router", 
  lambda state: state.get("next"), 
    {"therapist": "therapist", "logical": "logical"})
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)
()
graph = graph_builder.compile()



try:
    with open("graph.md", "w") as f:
        f.write(f"```mermaid\n{graph.get_graph().draw_mermaid()}\n```\n")
except Exception as e:
    print("Could not display graph visualization:", e)

def run_chatbot():
    state = {"messages": [], "message_type": None, "next": None}
    while True:
        user_input = input("Enter a message (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Bye")
            break
        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
        state = graph.invoke(state)
        if state.get("messages") and len(state["messages"]) > 0:
            print("Assistant:", state["messages"][-1].content)

if __name__ == "__main__":
    run_chatbot()   

