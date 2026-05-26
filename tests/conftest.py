from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import pytest

from dagqa.config import AppConfig
from dagqa.llm.base import LanguageModel
from dagqa.schemas import LLMRequest, LLMResponse


class StubLLM(LanguageModel):
    def __init__(self, responses: Iterable[str]) -> None:
        self.responses = deque(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("No stub LLM response queued.")
        return LLMResponse(text=self.responses.popleft(), model="stub")


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "llm": {"provider": "gemini", "model": "gemini/gemini-2.0-flash"},
            "planner": {"max_nodes": 10, "max_depth": 4, "repair_rounds": 1},
            "execution": {
                "max_parallel_nodes": 4,
                "node_timeout_seconds": 10,
                "node_repair_rounds": 1,
                "fail_fast": False,
            },
        }
    )
