from langchain.tools import tool
from judge import VisualJudge 
from typing import Annotated
from agent import VisualTools
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI


@tool
def visualize(
    query: Annotated[str, "Detailed query for the visualization"],
    name: Annotated[
        str, "name of the output file, should be descriptive of the visualization"
    ],
    output_type: Annotated[
        str,
        'type of the output/media to generate, should be either "animation" or "image"',
    ],
    feedback: Annotated[str, "Feedback from the visualization judge, if avaliable. Can leave blank if it is not avaliable"] = "",
    current_code: Annotated[str, "Current version of the code to work off of, if avliable. Can leave blank if it is not avaliable"] = ""
) -> str:
    """
    Generates a visualization using the Manim library based on the provided query.

    Args:
        query (str): Specific description of the concept to visualize.
        name (str): Name for the output file.
        output_type (str): Type of output, "animation" or "image".
        feedback (str): Feedback from the visualization judge, if available. Can leave blank if it is not available. 
        current_code (str): Current version of the code that you want to improve to work off of, if available. Can leave blank if it is not available
    Returns:
        str: Path to the generated visualization file.
        str: Current version of the code
    """
    vis_tools = VisualTools()
    vis_file_path, code = vis_tools.visualize(query, name, output_type, feedback, current_code)
    return code, vis_file_path

@tool
def judge_visualization(
    result_path: Annotated[str, "Path to the generated visualization file obtained from calling the visualize tool"]
) -> str:
    """
    Judges the quality of the generated visualization.

    Args:
        result_path (str): Path to the generated visualization file.

    Returns:
        str: Critiques and feedback on the generated visualization.
    """
    judge = VisualJudge()
    critiques = judge.get_critques(result_path)
    return critiques


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


tools = [visualize, judge_visualization]
tool_node = ToolNode(tools)

system_prompt = """
You are an expert visualization agent designed to create high-quality educational animations and images.

Your primary responsibility is to generate clear, intuitive visualizations based on user prompts using the Manim library.

PROCESS:
1. When you receive a user prompt, carefully analyze what they're asking to visualize.
2. Use the `visualize` tool to generate the animation or image requested.
   - Provide a detailed query that specifies exactly what should be visualized
   - Choose a descriptive name for the output file
   - Specify the output_type as "animation" or "image" based on what would best represent the concept

3. After generating the visualization, ALWAYS use the `judge_visualization` tool with the returned file path to get feedback.
4. Evaluate the judge's feedback:
   - If the judge indicates "visualization is passable" or provides a mostly positive feedback, inform the user that the visualization is complete and successful and stop the task. j
   - If the judge indicates the visualization needs improvement, create an improved version by calling the `visualize` tool again keeping the same query, but then passing the feedback into the feedback field to improve it. Pass in the current version of the code for it to be improved as well.
   - Continue this improvement loop until the judge deems the visualization passable.

IMPORTANT GUIDELINES:
- NEVER conclude the task until the judge explicitly indicates the visualization is passable
- For mathematical concepts, ensure accurate representation of principles and relationships
- For animations, consider timing, transitions, and clarity of motion
- Always prioritize visualization clarity, ensuring nothing overlaps and it is readable 
- When improving based on feedback, make substantial changes that address the core issues

Think step by step and be methodical in your approach to creating effective visualizations.
"""

model_with_tools = ChatOpenAI(model="o3-mini", max_tokens=20000).bind_tools(tools)

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")
app = workflow.compile()

# =============================================================================
prompt = """
Visualize the concept of the differention using limits
"""
# =============================================================================

# for chunk in app.stream(
#     {"messages": [("human", prompt)]},
#     stream_mode="values",
# ):
#     chunk["messages"][-1].pretty_print()

for chunk in app.stream(
    {"messages": [(system_prompt), ("human", prompt)]},  # Include system message here
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
