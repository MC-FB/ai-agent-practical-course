from __future__ import annotations

from dagqa.config import LLMConfig
from dagqa.llm.base import LanguageModel
from dagqa.llm.dspy_adapter import DspyLanguageModel


def build_llm(config: LLMConfig) -> LanguageModel:
    if config.provider in {"gemini", "openrouter", "azure_openai", "dspy"}:
        return DspyLanguageModel(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
