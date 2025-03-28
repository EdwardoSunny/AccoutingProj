import os
import subprocess
import re
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from storm_config import STORMConfig
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import Annotated

class VisualTools:
    def __init__(self, storm_config: STORMConfig = STORMConfig()):
        # Initialize based on the specified model
        self.storm_config = storm_config
        if storm_config.code_model.split("-")[0].lower() == "gpt":
            self.llm = ChatOpenAI(
                model=storm_config.code_model, temperature=storm_config.temperature, api_key=storm_config.openai_api_key
            )
            self.code_writing_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are an expert in the Manim library, specializing in creating animations and visualizations in Python. Your task is to write Manim code based on user-provided topics or queries and you ONLY OUTPUT PYTHON CODE with no explanations. If the user includes a specific example, generate a visualization of that concept directly. If no example is given, create a relevant example that effectively represents the topic. Provide only the Python code as output, formatted to be saved and executed as a .py file.",
                    ),
                    ("user", "{query}"),
                ]
            )
        elif storm_config.code_model.split("-")[0].lower() == "claude":
            self.llm = ChatAnthropic(
                model=storm_config.code_model, temperature=storm_config.temperature, api_key=storm_config.anthropic_api_key
            )
            # Prompt templates for generating and fixing Manim code
            self.code_writing_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are an expert in the Manim library, specializing in creating animations and visualizations in Python. Your task is to write Manim code based on user-provided topics or queries and you ONLY OUTPUT PYTHON CODE with no explanations. If the user includes a specific example, generate a visualization of that concept directly. If no example is given, create a relevant example that effectively represents the topic. Provide only the Python code as output, formatted to be saved and executed as a .py file.",
                    ),
                    ("user", "{query}"),
                ]
            )
        elif "o1" in storm_config.code_model.lower() or "o3" in storm_config.code_model.lower():
            self.llm = ChatOpenAI(
                model=storm_config.code_model
            )
            self.code_writing_prompt = ChatPromptTemplate.from_messages([
                (
                    "user",
                    "You are an expert in the Manim library, specializing in creating animations and visualizations in Python. Your task is to write Manim code based on user-provided topics or queries and you ONLY OUTPUT PYTHON CODE with no explanations. If the query includes a specific example, generate a visualization of that concept directly. If no example is given, create a relevant example that effectively represents the topic. Provide only the Python code as output, formatted to be saved and executed as a .py file.\n\n{query}"
                )
            ])
        else:
            raise Exception(
                f"Visualization Tools: Model {storm_config.code_model} is not available!"
            )

        
        self.code_writer = self.code_writing_prompt | self.llm

    def visualize(
        self,
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
        code = self.code_writer.invoke({"query": query})
        for _ in range(0, self.storm_config.max_retry):
            success, error = self._execute_code(code.content, name, output_type)
            if success:
                # Return the path to the generated file
                if output_type == "animation":
                    return (
                        os.path.join(self.storm_config.media_path, f"{name}.mp4")
                        .replace("sandbox:.", "")
                        .replace("sandbox:", "")
                    )
                if output_type == "image":
                    return (
                        os.path.join(
                            self.storm_config.media_path,
                            f"{name}.png",
                        )
                        .replace("sandbox:.", "")
                        .replace("sandbox:", "")
                    )
            else:
                code = self.code_writer.invoke({"query": query})

        raise Exception(
            f"Visualization Tool: Manim agent code execution error: {error}"
        )

    def remove_code_formatting(self, text):
        """
        Remove any Markdown-style code block formatting from the provided text.

        Args:
            text (str): Text containing formatted code.

        Returns:
            str: Cleaned code text.
        """
        text = re.sub(r"``` ?python", "", text)  # Remove opening code formatting
        return text.replace("```", "")  # Remove closing code block formatting

    def _execute_code(self, code: str, name="default", output_type="animation"):
        """
        Execute Manim code to generate a visualization.

        Args:
            code (str): Manim Python code to execute.
            name (str): Name for the output file.
            output_type (str): Type of output, "animation" or "image".

        Returns:
            tuple: (success (bool), error (str or None))
        """
        code = self.remove_code_formatting(code)
        temp_filename = f"{name}.py"
        with open(temp_filename, "w") as f:
            f.write(code)

        media_dir = os.path.abspath(self.storm_config.media_path)
        print(media_dir)
        os.makedirs(media_dir, exist_ok=True)
        manim_command = [
            "manim",
            temp_filename,
            self.storm_config.quality,
            "--media_dir",
            media_dir,
        ]

        output_ext = ".mp4" if output_type == "animation" else ".png"
        if output_type == "image":
            manim_command.append("-s")
        output_file = os.path.join(media_dir, f"{name}{output_ext}")
        manim_command.extend(["--output_file", output_file])

        try:
            subprocess.run(manim_command, check=True)
            return True, None
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


if __name__ == "__main__":

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
    
    # Solve the following problem and create an animation of the solution:
    prompt = """Create an animation to demonstrate how to do inventory valuation using the Lower of cost or market"""

    for chunk in app.stream(
        {"messages": [("human", prompt)]},
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
