from typing import Annotated, TypedDict 
from langgraph.graph import START, END 
from langgraph.graph.state import StateGraph 
from langgraph.graph.message import add_messages 
from langgraph.prebuilt import ToolNode, tools_condition 
from langchain_core.tools import tool 
from langchain_core.messages import BaseMessage
from langchain.chat_models import init_chat_model
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"


llm = init_chat_model("groq:qwen/qwen3.6-27b")

class State(TypedDict): 
    messages: Annotated[list[BaseMessage], add_messages]

def make_tool_graph(): 
    @tool 
    def add(a: float, b: float): 
        """Add two number"""
        return a+b
    tools = [add]


    llm_with_tool = llm.bind_tools([add])

    ## Node functionality 
    def tool_calling_llm(state: State): 
        return [llm_with_tool.invoke(state["messages"])]


    ## Graph intiliaze 
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    ## Add Edges 
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")

    ## compile 
    graph = builder.compile()
    return graph

tool_agent = make_tool_graph()