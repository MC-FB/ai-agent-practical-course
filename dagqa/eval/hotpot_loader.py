from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class HotpotExample(BaseModel):
    id: str
    question: str
    answer: str


HOTPOTQA_DISTRACTOR_VALIDATION_SIZE = 7405


def load_hotpot_examples(
    path: str | Path | None = None,
    limit: int | None = None,
) -> list[HotpotExample]:
    if path is None:
        return _load_from_huggingface(limit)
    examples = list(_load_from_json(path))
    return examples[:limit] if limit is not None else examples


def count_hotpot_examples(path: str | Path | None = None) -> int:
    if path is None:
        return HOTPOTQA_DISTRACTOR_VALIDATION_SIZE
    return sum(1 for _example in _load_from_json(path))


def _load_from_json(path: str | Path) -> Iterator[HotpotExample]:
    data = json.loads(Path(path).read_text())
    records: list[dict[str, Any]]
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        records = data["data"]
    else:
        raise ValueError("HotpotQA file must be a list or an object with a data list.")
    for item in records:
        yield HotpotExample(
            id=str(item.get("_id") or item.get("id")),
            question=str(item["question"]),
            answer=str(item["answer"]),
        )


def _load_from_huggingface(limit: int | None) -> list[HotpotExample]:
    cache_dir = Path(".cache/huggingface")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        cache_dir=str(cache_dir.resolve()),
    )
    examples = []
    for row in dataset:
        examples.append(
            HotpotExample(
                id=str(row["id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples
