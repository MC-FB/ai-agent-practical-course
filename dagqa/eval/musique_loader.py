from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from dagqa.eval.hotpot_loader import HotpotExample
from dagqa.schemas import EvidenceDocument, GoldSupportingFact

load_dotenv()

MUSIQUE_DATASET_ID = "dgslibisey/MuSiQue"
MUSIQUE_SPLIT = "validation"
MUSIQUE_MIN_HARD_HOPS = 3


def load_musique_examples(
    path: str | Path | None = None,
    *,
    subset: str = "validation",
    limit: int | None = None,
) -> list[HotpotExample]:
    rows = _load_rows_from_json(path) if path is not None else _load_rows_from_huggingface()
    examples = [_parse_row(row) for row in _filter_rows(rows, subset)]
    return examples[:limit] if limit is not None else examples


def count_musique_examples(path: str | Path | None = None, *, subset: str = "validation") -> int:
    rows = _load_rows_from_json(path) if path is not None else _load_rows_from_huggingface()
    return sum(1 for _row in _filter_rows(rows, subset))


def _load_rows_from_huggingface() -> list[dict[str, Any]]:
    cache_dir = Path(".cache/huggingface")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
    from datasets import load_dataset  # noqa: PLC0415

    load_kwargs: dict[str, Any] = {
        "split": MUSIQUE_SPLIT,
        "cache_dir": str(cache_dir.resolve()),
    }
    if token := os.getenv("HF_TOKEN"):
        load_kwargs["token"] = token
    dataset = load_dataset(MUSIQUE_DATASET_ID, **load_kwargs)
    return [dict(row) for row in dataset]


def _load_rows_from_json(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "examples", "validation"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("MuSiQue file must be a list or an object with a data/examples list.")


def _filter_rows(rows: Iterable[dict[str, Any]], subset: str) -> Iterator[dict[str, Any]]:
    for row in rows:
        if subset == "validation_3hop_plus" and _decomposition_length(row) < MUSIQUE_MIN_HARD_HOPS:
            continue
        if subset not in {"validation", "validation_3hop_plus", "marked_failures_2026_06_28"}:
            raise ValueError(f"Unknown MuSiQue subset: {subset}")
        yield row


def _decomposition_length(row: dict[str, Any]) -> int:
    decomposition = row.get("question_decomposition") or []
    return len(decomposition) if isinstance(decomposition, list) else 0


def _parse_row(row: dict[str, Any]) -> HotpotExample:
    documents, supporting_facts = _parse_paragraphs(row.get("paragraphs"))
    return HotpotExample(
        id=str(row.get("id") or row.get("_id")),
        question=str(row["question"]),
        answer=str(row["answer"]),
        context=documents,
        supporting_facts=supporting_facts,
    )


def _parse_paragraphs(value: Any) -> tuple[list[EvidenceDocument], list[GoldSupportingFact]]:
    if not value:
        return [], []

    documents: list[EvidenceDocument] = []
    supporting_facts: list[GoldSupportingFact] = []
    for position, paragraph in enumerate(value):
        if not isinstance(paragraph, dict):
            continue
        title = str(paragraph.get("title") or paragraph.get("paragraph_title") or "")
        text = str(
            paragraph.get("paragraph_text")
            or paragraph.get("text")
            or paragraph.get("paragraph")
            or ""
        ).strip()
        original_idx = paragraph.get("idx", paragraph.get("paragraph_idx", position))
        document = EvidenceDocument(
            id=f"context-{position}",
            title=title,
            text=text,
            metadata={
                "source": "musique_context",
                "position": position,
                "original_idx": original_idx,
                "is_supporting": bool(paragraph.get("is_supporting")),
                "sentences": [text],
            },
        )
        documents.append(document)
        if paragraph.get("is_supporting"):
            supporting_facts.append(GoldSupportingFact(title=title, sentence_index=0))
    return documents, supporting_facts
