from langchain.tools import tool
from typing import Annotated
from agent import VisualTools
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

@tool
def visualize(
    query: Annotated[str, "Detailed query for the visualization"],
    name: Annotated[
        str, "name of the output file, should be descriptive of the visualization"
    ],
    output_type=Annotated[
        str,
        'type of the output/media to generate, should be either "animation" or "image"',
    ],
) -> str:
    """
    Generates a visualization using the Manim library based on the provided query.

    Args:
        query (str): Specific description of the concept to visualize.
        name (str): Name for the output file.
        output_type (str): Type of output, "animation" or "image".

    Returns:
        str: Path to the generated visualization file.
    """
    vis_tools = VisualTools()
    return vis_tools.visualize(query, name, output_type)

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

tools = [visualize]
tool_node = ToolNode(tools)

model_with_tools = ChatOpenAI(model="gpt-4o", temperature=0.7).bind_tools(tools)

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")
app = workflow.compile()

# =============================================================================
prompt = """Create an animation to demonstrate how to do inventory valuation using the Lower of cost or market"""
# =============================================================================

for chunk in app.stream(
    {"messages": [("human", prompt)]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
