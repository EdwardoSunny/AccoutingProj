from utils import *
from storm_config import STORMConfig
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import tqdm

class VisualJudge:
    def __init__(self, storm_config: STORMConfig = STORMConfig()):
        self.storm_config = storm_config
        print(storm_config.judge_model)
        if storm_config.judge_model.split("-")[0].lower() == "gpt":
            self.judge_llm = ChatOpenAI(
                model=storm_config.judge_model,
                temperature=storm_config.temperature,
                api_key=storm_config.openai_api_key,
            )
        elif storm_config.code_model.split("-")[0].lower() == "claude":
            self.judge_llm = ChatAnthropic(
                model=storm_config.judge_model,
                temperature=storm_config.temperature,
                api_key=storm_config.anthropic_api_key,
            )
        else:
            raise Exception(
                f"Visualization Tools: Model {storm_config.code_model} is not available!"
            )
        self.summary_llm = ChatOpenAI(
            model=storm_config.summary_model,
            temperature=storm_config.temperature,
            api_key=storm_config.openai_api_key,
        )
        self.judge_sys_prompt = SystemMessage(
            content="You are an expert in analyzing Manim visualizations for one specific issue: overlapping content. Focus ONLY on identifying if any elements in the visualization overlap with each other. Be specific on what exactly is overlapping and exactly what needs to be changed. Do not comment on readability, aesthetics, or any other aspects of the visualization.",
        )
        
    def get_critques(self, result_path):
        frames = extract_frames(
            result_path,
            5,
        )
        encoded_frames = frames_to_base64(frames)
        all_critiques = ""
        for frame in tqdm.tqdm(encoded_frames):
            content_elements = [
                {
                    "type": "text",
                    "text": "Check this Manim visualization frame and identify ONLY if there are any overlapping elements. Be specific about what is overlapping and exactly what needs to be changed. Do not comment on any other aspects of the visualization.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                },
            ]
            message = HumanMessage(content=content_elements)
            # Include the system message in the messages list
            response = self.judge_llm.invoke([self.judge_sys_prompt, message])
            all_critiques += response.content + "\n\n"
        
        messages = [
            SystemMessage(
                content="You are tasked with reviewing feedback about a Manim animation. Your only job is to determine if there is any overlapping content in the animation. Nothing else matters. Be very specific on what needs to be changed and what is overlapping. Based on the feedback, provide a one-word evaluation formatted as PASSABLE: YES (if there is no overlap) or PASSABLE: NO (if there is any overlap). Before your final evaluation, briefly summarize only the overlap issues found, if any."
            ),
            HumanMessage(
                content=f"Determine if there is any overlap in this animation based on these critiques: {str(all_critiques)}"
            ),
        ]
        summary = self.summary_llm.invoke(messages).content
        return summary

if __name__ == "__main__":
    judge = VisualJudge()
    judge.get_critques(
        "/home/edward/Documents/Code/ScAI/AccountingProj/generated_visuals/pythagorean_theorem_animation.mp4"
    )
