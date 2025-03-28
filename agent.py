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
from langchain.callbacks import get_openai_callback


class VisualTools:
    def __init__(self, storm_config: STORMConfig = STORMConfig()):
        # Initialize based on the specified model
        self.storm_config = storm_config
        if storm_config.code_model.split("-")[0].lower() == "gpt":
            self.llm = ChatOpenAI(
                model=storm_config.code_model,
                temperature=storm_config.temperature,
                api_key=storm_config.openai_api_key,
                max_tokens=20000,
            )
            # Initial code generation prompt
            self.code_writing_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert in the Manim library, specializing in creating animations and visualizations in Python. Your task is to write Manim code based on user-provided topics or queries and you ONLY OUTPUT PYTHON CODE with no explanations. If the user includes a specific example, generate a visualization of that concept directly. If no example is given, create a relevant example that effectively represents the topic. Provide only the raw Python code as output with NO markdown formatting (no ```python or ``` tags). Your output should be plain Python code that can be directly saved and executed as a .py file.",
                    ),
                    ("user", "Here is the user query: {query}\n\nHere are some feedbacks you should pay attention to from the previous code: {feedback}"),
                ]
            )
            # Code fixing prompt
            self.code_fixing_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert in the Manim library, specializing in debugging and fixing Python code for animations and visualizations. Review the provided code and error message, then fix the issues. Your response should ONLY include the complete corrected Python code with NO markdown formatting (no ```python or ``` tags) and no explanations. Make minimal changes necessary to fix the error. Return the plain Python code that can be directly saved and executed as a .py file. DO NOT create new code - edit the existing code to fix the issues.",
                    ),
                    ("user", "Here is the code that needs to be fixed:\n\n{code}\n\nThis code failed with the following error:\n\n{error}\n\nPlease fix the code to resolve this error."),
                ]
            )
        elif storm_config.code_model.split("-")[0].lower() == "claude":
            self.llm = ChatAnthropic(
                model=storm_config.code_model,
                temperature=storm_config.temperature,
                api_key=storm_config.anthropic_api_key,
                max_tokens=20000,
            )
            # Initial code generation prompt
            self.code_writing_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert in the Manim library, specializing in creating animations and visualizations in Python. Your task is to write Manim code based on user-provided topics or queries and you ONLY OUTPUT PYTHON CODE with no explanations. If the user includes a specific example, generate a visualization of that concept directly. If no example is given, create a relevant example that effectively represents the topic. Provide only the Python code as output, formatted to be saved and executed as a .py file.",
                    ),
                    ("user", "Here is the user query: {query}\n\nHere are some feedbacks you should pay attention to from the previous code: {feedback}"),
                ]
            )
            # Code fixing prompt
            self.code_fixing_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert in the Manim library, specializing in debugging and fixing Python code for animations and visualizations. Review the provided code and error message, then fix the issues. Your response should ONLY include the complete corrected Python code with NO markdown formatting (no ```python or ``` tags) and no explanations. Make minimal changes necessary to fix the error. DO NOT create new code - edit the existing code to fix the issues.",
                    ),
                    ("user", "Here is the code that needs to be fixed:\n\n{code}\n\nThis code failed with the following error:\n\n{error}\n\nPlease fix the code to resolve this error."),
                ]
            )
        elif (
            "o1" in storm_config.code_model.lower()
            or "o3" in storm_config.code_model.lower()
        ):
            self.llm = ChatOpenAI(model=storm_config.code_model, max_tokens=20000)
            # Initial code generation prompt
            self.code_writing_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "user",
                        "You are an expert in the Manim library, specializing in creating animations and visualizations in Python. Your task is to write Manim code based on user-provided topics or queries and you ONLY OUTPUT PYTHON CODE with no explanations. If the query includes a specific example, generate a visualization of that concept directly. If no example is given, create a relevant example that effectively represents the topic. Provide only the raw Python code as output with NO markdown formatting (no ```python or ``` tags). Your output should be plain Python code that can be directly saved and executed as a .py file.\n\nHere is the user query: {query}\n\nHere are some feedbacks you should pay attention to from the previous code: {feedback}",
                    )
                ]
            )
            # Code fixing prompt
            self.code_fixing_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "user",
                        "You are an expert in the Manim library, specializing in debugging and fixing Python code for animations and visualizations. Review the provided code and error message, then fix the issues. Your response should ONLY include the complete corrected Python code with NO markdown formatting (no ```python or ``` tags) and no explanations. Make minimal changes necessary to fix the error. Return the plain Python code that can be directly saved and executed as a .py file. DO NOT create new code - edit the existing code to fix the issues.\n\nHere is the code that needs to be fixed:\n\n{code}\n\nThis code failed with the following error:\n\n{error}\n\nPlease fix the code to resolve this error.",
                    )
                ]
            )
        else:
            raise Exception(
                f"Visualization Tools: Model {storm_config.code_model} is not available!"
            )

        self.code_writer = self.code_writing_prompt | self.llm
        self.code_fixer = self.code_fixing_prompt | self.llm

    def visualize(
        self,
        query: Annotated[str, "Detailed query for the visualization"],
        name: Annotated[
            str, "name of the output file, should be descriptive of the visualization"
        ],
        output_type: Annotated[
            str,
            'type of the output/media to generate, should be either "animation" or "image"',
        ],
        feedback: Annotated[
            str,
            "Feedback from the visualization judge, if avaliable. Can leave blank if it is not avaliable",
        ],
        current_code: Annotated[
            str,
            "Current code to edit, if available. If provided, the system will edit this code instead of generating from scratch",
        ] = None
    ) -> tuple[str, str]:
        """
        Generates a visualization using the Manim library based on the provided query.

        Args:
            query (str): Specific description of the concept to visualize.
            name (str): Name for the output file.
            output_type (str): Type of output, "animation" or "image".
            feedback (str): Feedback from the visualization judge, if available. Can leave blank if it is not available.
            current_code (str, optional): Current code to edit if available. If provided, the system will edit this code 
                                          instead of generating from scratch.

        Returns:
            tuple[str, str]: A tuple containing (path to generated file, final code used)
        """
        # Determine if we're generating new code or editing existing code
        if current_code is not None and feedback:
            # We have both existing code and feedback, so edit the code
            print("Editing existing code based on feedback...")
            edit_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a Manim expert who refines animations by updating Python code based on specific feedback. Given existing code and suggestions, update it while preserving its overall structure. Return only the complete updated Python code with no markdown formatting. Ensure your animation stays within the frame and avoids overcrowding by: adjusting object sizes (using smaller mobjects or scale()), setting x_length and y_length for Axes, preventing overlaps with layout methods (like .next_to(), .shift(), or VGroup.arrange()), using relative positioning, and specifying buffer values (buff) for proper spacing."
                ),
                (
                    "user",
                    f"Here is the existing Manim code:\n\n{current_code}\n\nHere is the feedback on how to improve it:\n\n{feedback}\n\nPlease modify the code to address this feedback."
                )
            ])
            
            edit_chain = edit_prompt | self.llm
            code_response = edit_chain.invoke({})
        else:
            # Either we have no existing code or no feedback, generate from query
            code_response = self.code_writer.invoke({"query": query, "feedback": feedback})
        # Extract the content as a string from the response
        if hasattr(code_response, 'content'):
            code = code_response.content
        else:
            # Handle case where response might be an AIMessage or other object type
            code = str(code_response)
            
        # Remove any markdown formatting
        code = self.remove_code_formatting(code)
        
        # Store the original code for reference and iterative improvement
        original_code = code
        
        # Keep track of all errors for context
        error_history = []
        
        for attempt in range(self.storm_config.max_retry):
            # Check for syntax errors before executing
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as syntax_err:
                print(f"Syntax error detected in attempt {attempt+1}: {syntax_err}")
                # Add to error history
                error_history.append(f"Syntax error: {str(syntax_err)}")
                
                # If this is a truncation issue at the end of the code, try to fix it
                if "unexpected EOF" in str(syntax_err) or "was never closed" in str(syntax_err):
                    print("Detected potential truncation issue, requesting code completion...")
                    
                    # Create a prompt to complete the truncated code
                    completion_prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            "You are an expert in completing truncated Python code. The provided code appears to have been cut off mid-statement. Complete ONLY the truncated part without rewriting the entire script. For example, if you see 'def function(arg1, arg2' you should only add '):'."
                        ),
                        (
                            "user",
                            f"This code was truncated. Please complete ONLY the missing parts at the end:\n\n{code}"
                        )
                    ])
                    
                    completion_chain = completion_prompt | self.llm
                    completion_response = completion_chain.invoke({})
                    
                    if hasattr(completion_response, 'content'):
                        completion = completion_response.content
                    else:
                        completion = str(completion_response)
                    
                    completion = self.remove_code_formatting(completion)
                    
                    # Append the completion to the original code
                    code = code + "\n" + completion
                    print("Code completed, retrying...")
                    continue
                else:
                    # Use the code fixer to fix syntax errors
                    error = str(syntax_err)
            
            # Execute the code
            success, error = self._execute_code(code, name, output_type)
            
            if success:
                # Return the path to the generated file and the final code
                output_path = ""
                if output_type == "animation":
                    output_path = (
                        os.path.join(self.storm_config.media_path, f"{name}.mp4")
                        .replace("sandbox:.", "")
                        .replace("sandbox:", "")
                    )
                if output_type == "image":
                    output_path = (
                        os.path.join(
                            self.storm_config.media_path,
                            f"{name}.png",
                        )
                        .replace("sandbox:.", "")
                        .replace("sandbox:", "")
                    )
                return output_path, code
            else:
                # If this is the last attempt, don't try to fix it anymore
                if attempt == self.storm_config.max_retry - 1:
                    break
                
                # Fix the code based on the error
                print(f"Attempt {attempt+1} failed with error: {error}")
                
                
# Add to error history
                error_history.append(f"Execution error (attempt {attempt+1}): {error}")
                
                # Prepare comprehensive error context
                error_context = "\n\n".join(error_history)
                
                with get_openai_callback() as cb:
                    # Use the code fixer to fix the code based on the error
                    fix_response = self.code_fixer.invoke({
                        "code": code, 
                        "error": error_context
                    })
                    
                    # Extract content as string from the response
                    if hasattr(fix_response, 'content'):
                        fixed_code = fix_response.content
                    else:
                        # Handle case where response might be an AIMessage or other object type
                        fixed_code = str(fix_response)
                    
                    # Remove any markdown formatting
                    fixed_code = self.remove_code_formatting(fixed_code)
                    
                    # Check if the code changed significantly
                    if self._code_similarity(code, fixed_code) < 0.5:
                        print("Warning: Fixed code differs significantly from original. Using a hybrid approach.")
                        # Try to preserve structure while incorporating fixes
                        fixed_code = self._merge_code_changes(original_code, code, fixed_code)
                    
                    code = fixed_code
                    
                    print(f"Total Tokens: {cb.total_tokens}")
                    print(f"Prompt Tokens: {cb.prompt_tokens}")
                    print(f"Completion Tokens: {cb.completion_tokens}")
                    print(f"Total Cost (USD): ${cb.total_cost}")

        raise Exception(
            f"Visualization Tool: Manim agent code execution failed after {self.storm_config.max_retry} attempts. Last error: {error}"
        )

    def _code_similarity(self, code1, code2):
        """
        Calculate a simple similarity metric between two code snippets.
        Returns a value between 0 (completely different) and 1 (identical).
        """
        # Remove whitespace and normalize
        clean1 = re.sub(r'\s+', '', code1)
        clean2 = re.sub(r'\s+', '', code2)
        
        # Calculate Levenshtein distance (or a simple approximation)
        # For simplicity, we'll just use length ratio for now
        len1, len2 = len(clean1), len(clean2)
        if len1 == 0 and len2 == 0:
            return 1.0
        elif len1 == 0 or len2 == 0:
            return 0.0
        
        # Use length ratio as a basic similarity measure
        return min(len1, len2) / max(len1, len2)
    
    def _merge_code_changes(self, original_code, current_code, fixed_code):
        """
        Attempt to merge changes from fixed_code into current_code while preserving structure.
        This is a simple implementation - in a production system, you might use a more sophisticated
        diff and merge algorithm.
        """
        # Simple case: if the fixes are minor, just use the fixed code
        if self._code_similarity(current_code, fixed_code) > 0.9:
            return fixed_code
            
        # For more complex cases, we'll use the LLM to help merge the changes intelligently
        merge_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert in Python code refactoring. You will be given three versions of code: the original, a current version with bugs, and a fixed version. Your task is to incorporate the fixes from the fixed version into the current code while preserving as much of the original structure as possible. Return ONLY the merged Python code with NO formatting or explanations."
            ),
            (
                "user",
                f"Original code:\n\n{original_code}\n\nCurrent code with bugs:\n\n{current_code}\n\nFixed code:\n\n{fixed_code}\n\nPlease merge the fixes from the fixed code into the current code while preserving the original structure as much as possible."
            )
        ])
        
        merge_chain = merge_prompt | self.llm
        merge_response = merge_chain.invoke({})
        
        if hasattr(merge_response, 'content'):
            merged_code = merge_response.content
        else:
            merged_code = str(merge_response)
        
        # Remove any markdown formatting
        merged_code = self.remove_code_formatting(merged_code)
        
        return merged_code

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
        # Check for syntax errors before trying to execute
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as syntax_err:
            return False, f"Syntax error in generated code: {str(syntax_err)}"
            
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
            result = subprocess.run(manim_command, check=True, capture_output=True, text=True)
            return True, None
        except subprocess.CalledProcessError as e:
            # Capture the error output
            error_message = e.stderr if e.stderr else e.stdout
            return False, error_message
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
