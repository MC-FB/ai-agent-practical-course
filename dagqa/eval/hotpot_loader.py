from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dagqa.schemas import EvidenceDocument, GoldSupportingFact

load_dotenv()


class HotpotExample(BaseModel):
    id: str
    question: str
    answer: str
    context: list[EvidenceDocument] = Field(default_factory=list)
    supporting_facts: list[GoldSupportingFact] = Field(default_factory=list)


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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
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
            context=_parse_context(item.get("context")),
            supporting_facts=_parse_supporting_facts(item.get("supporting_facts")),
        )


def _load_from_huggingface(limit: int | None) -> list[HotpotExample]:
    cache_dir = Path(".cache/huggingface")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
    from datasets import load_dataset  # noqa: PLC0415

    load_kwargs: dict[str, Any] = {
        "split": "validation",
        "cache_dir": str(cache_dir.resolve()),
    }
    if token := os.getenv("HF_TOKEN"):
        load_kwargs["token"] = token
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", **load_kwargs)
    examples = []
    for row in dataset:
        examples.append(
            HotpotExample(
                id=str(row["id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                context=_parse_context(row.get("context")),
                supporting_facts=_parse_supporting_facts(row.get("supporting_facts")),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def _parse_context(value: Any) -> list[EvidenceDocument]:
    if not value:
        return []

    documents = []
    if isinstance(value, dict):
        titles = value.get("title") or []
        sentence_groups = value.get("sentences") or []
        items = zip(titles, sentence_groups, strict=False)
    else:
        items = value

    for index, item in enumerate(items):
        if isinstance(item, dict):
            title = str(item.get("title") or "")
            sentences = item.get("sentences") or item.get("text") or []
        else:
            title, sentences = item
            title = str(title)
        if isinstance(sentences, str):
            text = sentences
            sentence_list = [sentences]
        else:
            sentence_list = [str(sentence) for sentence in sentences]
            text = "".join(sentence_list)
        documents.append(
            EvidenceDocument(
                id=f"context-{index}",
                title=title,
                text=text.strip(),
                metadata={
                    "source": "hotpotqa_context",
                    "position": index,
                    "sentences": sentence_list,
                },
            )
        )
    return documents


def _parse_supporting_facts(value: Any) -> list[GoldSupportingFact]:
    if not value:
        return []
    if isinstance(value, dict):
        items = zip(
            value.get("title") or [],
            value.get("sent_id") or value.get("sentence_index") or [],
            strict=False,
        )
    else:
        items = value
    return [
        GoldSupportingFact(title=str(title), sentence_index=int(sentence_index))
        for title, sentence_index in items
    ]
