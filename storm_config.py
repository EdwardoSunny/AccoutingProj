from dataclasses import dataclass
import os


@dataclass
class STORMConfig:
    """Configuration for language models"""

    # overall configs
    fast_model: str = "gpt-4o-mini"
    smart_model: str = "gpt-4o"
    long_context_model: str = "gpt-4o"
    code_model: str = "claude-3-7-sonnet-latest"
    # code_model: str = "o3-mini"
    embedding_model: str = "text-embedding-3-small"
    max_retry: int = 5

    # visualization tool configs
    media_path: str = f"{os.getcwd()}/generated_visuals/"
    temperature: float = 1.0
    quality: str = "-qh"  # manim docs

    # judge model
    judge_model: str = "gpt-4o"
    summary_model: str = "gpt-4o-mini"

    # output locations
    docs_path: str = f"{os.getcwd()}/generated_docs/"

    # api keys
    openai_api_key: str = (
        ""
    )
    anthropic_api_key: str = (
        ""
    )
