from __future__ import annotations

from abc import ABC, abstractmethod

from dagqa.schemas import LLMRequest, LLMResponse


class LanguageModel(ABC):
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
