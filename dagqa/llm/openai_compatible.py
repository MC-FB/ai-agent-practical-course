from __future__ import annotations

import os
import time

from openai import AsyncOpenAI

from dagqa.config import LLMConfig
from dagqa.llm.base import LanguageModel
from dagqa.schemas import LLMRequest, LLMResponse


class OpenAICompatibleLanguageModel(LanguageModel):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        api_key = os.getenv(config.api_key_env or "") if config.api_key_env else None
        api_base = (
            os.getenv(config.api_base_env) if config.api_base_env else None
        ) or config.api_base
        if not api_key or not api_base:
            raise ValueError("OpenAI-compatible LLM config requires an API key and base URL.")
        self.model = (os.getenv(config.model_env) if config.model_env else None) or config.model
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=request.temperature,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
        )
        message = response.choices[0].message
        usage = response.usage
        return LLMResponse(
            text=message.content or "",
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
