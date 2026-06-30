from __future__ import annotations

from dagqa.config import LLMConfig
from dagqa.llm.base import LanguageModel


def build_llm(config: LLMConfig) -> LanguageModel:
    from dagqa.llm.openai_compatible import OpenAICompatibleLanguageModel  # noqa: PLC0415

    return OpenAICompatibleLanguageModel(config)
