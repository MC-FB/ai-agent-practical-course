from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from dagqa.config import LLMConfig

T = TypeVar("T")


def is_retryable_llm_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "ratelimit", "connection"))


async def retry_llm_call(
    call: Callable[[], Awaitable[T]],
    config: LLMConfig,
) -> tuple[T, int]:
    attempt = 0
    while True:
        try:
            return await asyncio.wait_for(call(), timeout=config.request_timeout_seconds), attempt
        except Exception as exc:
            if attempt >= config.max_retries or not is_retryable_llm_error(exc):
                raise
            delay = min(
                config.retry_initial_delay_seconds * (2**attempt),
                config.retry_max_delay_seconds,
            )
            delay *= random.uniform(0.75, 1.25)
            attempt += 1
            await asyncio.sleep(delay)
