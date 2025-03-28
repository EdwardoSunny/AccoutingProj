from utils import *
from storm_config import STORMConfig
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import HumanMessage, AIMessage
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
            content="You are an expert in the Manim library and analyzing Manim visualizations for flaws, inconsistencies, and improvements. Your task is to provide feedback on the generated visualizations, pointing out any issues and suggesting improvements. You should provide detailed feedback on the visualizations, including any errors, overlapping content, inconsistencies, or areas for improvement.",
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
                    "text": "Please provide feedback on this Manim visualization, focusing on areas that may be unclear, visually confusing, or overly cluttered. Highlight any issues with overlapping elements, pacing, or visual hierarchy. Suggest specific improvements to enhance clarity, flow, and overall effectiveness of the visual presentation.",
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
                content="You are an expert in summarizing critiques of manim animations. Your task is to create a concise summary of the feedback provided by the judge model, emphasizing key issues and actionable improvements. This summary will guide the coder in enhancing the animation. Finally, conclude with a one-word evaluation—formatted as PASSABLE: YES or PASSABLE: NO—indicating whether the visualization is suitable for teaching and has no overlapping content. Be very specific about what exactly needs to be changed. Focus only on the most pressing/problematic issues."
            ),
            HumanMessage(
                content=f"Please summarize the critiques provided: {str(all_critiques)}"
            ),
        ]

        summary = self.summary_llm.invoke(messages).content

        return summary


if __name__ == "__main__":
    judge = VisualJudge()
    judge.get_critques(
        "/home/edward/Documents/Code/ScAI/AccountingProj/generated_visuals/pythagorean_theorem_animation.mp4"
    )
