from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import dspy

from dagqa.config import LLMConfig
from dagqa.llm.base import LanguageModel
from dagqa.schemas import LLMRequest, LLMResponse


def _extract_text(result: object) -> str:
    if isinstance(result, list):
        return _extract_text(result[0]) if result else ""
    if isinstance(result, dict):
        text = result.get("text") or result.get("content")
        if text is not None:
            return str(text)
    return str(result)


class DspyLanguageModel(LanguageModel):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        cache_dir = Path(".cache/dspy")
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("DSPY_CACHEDIR", str(cache_dir.resolve()))
        api_key = os.getenv(config.api_key_env or "") if config.api_key_env else None
        model = (os.getenv(config.model_env) if config.model_env else None) or config.model
        api_base = (
            os.getenv(config.api_base_env) if config.api_base_env else None
        ) or config.api_base
        api_version = (
            os.getenv(config.api_version_env) if config.api_version_env else None
        ) or config.api_version
        kwargs = {"api_key": api_key}
        if api_base:
            kwargs["api_base"] = api_base
        if api_version:
            kwargs["api_version"] = api_version
        self.lm = dspy.LM(model=model, **kwargs)
        dspy.configure(lm=self.lm)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        prompt = f"{request.system}\n\n{request.prompt}"

        def _call() -> str:
            result = self.lm(prompt, temperature=request.temperature)
            return _extract_text(result)

        text = await asyncio.to_thread(_call)
        return LLMResponse(
            text=text,
            model=(os.getenv(self.config.model_env) if self.config.model_env else None)
            or self.config.model,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


GeminiLanguageModel = DspyLanguageModel
