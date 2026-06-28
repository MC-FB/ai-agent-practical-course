from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dagqa.eval.hotpot_loader import (
    HOTPOTQA_DISTRACTOR_VALIDATION_SIZE,
    HotpotExample,
    count_hotpot_examples,
    load_hotpot_examples,
)
from dagqa.eval.musique_loader import count_musique_examples, load_musique_examples


@dataclass(frozen=True)
class BenchmarkSubsetInfo:
    id: str
    label: str
    path: str | None = None


@dataclass(frozen=True)
class BenchmarkDataset:
    id: str
    label: str
    split: str
    default_subset: str
    result_prefix: str
    source: str
    subsets: dict[str, BenchmarkSubsetInfo]
    load: Callable[[str, str | Path | None, int | None], list[HotpotExample]]
    count: Callable[[str, str | Path | None], int]

    def subset(self, subset_id: str) -> BenchmarkSubsetInfo:
        try:
            return self.subsets[subset_id]
        except KeyError as exc:
            raise ValueError(f"Unknown subset {subset_id!r} for dataset {self.id!r}.") from exc


def _load_hotpot_subset(
    subset: str,
    path: str | Path | None = None,
    limit: int | None = None,
) -> list[HotpotExample]:
    dataset = DATASETS["hotpotqa"]
    resolved_path = path if path is not None else dataset.subset(subset).path
    return load_hotpot_examples(resolved_path, limit=limit)


def _count_hotpot_subset(subset: str, path: str | Path | None = None) -> int:
    dataset = DATASETS["hotpotqa"]
    resolved_path = path if path is not None else dataset.subset(subset).path
    if resolved_path is None and subset == "validation":
        return HOTPOTQA_DISTRACTOR_VALIDATION_SIZE
    return count_hotpot_examples(resolved_path)


def _load_musique_subset(
    subset: str,
    path: str | Path | None = None,
    limit: int | None = None,
) -> list[HotpotExample]:
    return load_musique_examples(path, subset=subset, limit=limit)


def _count_musique_subset(subset: str, path: str | Path | None = None) -> int:
    return count_musique_examples(path, subset=subset)


DATASETS: dict[str, BenchmarkDataset] = {
    "hotpotqa": BenchmarkDataset(
        id="hotpotqa",
        label="HotpotQA",
        split="validation",
        default_subset="validation",
        result_prefix="hotpotqa",
        source="Hugging Face hotpotqa/hotpot_qa dataset card",
        subsets={
            "validation": BenchmarkSubsetInfo(
                id="validation",
                label="Full HotpotQA validation",
                path=None,
            ),
            "mistral_qwen_fact_retrieval_failures": BenchmarkSubsetInfo(
                id="mistral_qwen_fact_retrieval_failures",
                label="Mistral + Qwen fact-retrieval failures",
                path="data/hotpotqa/mini_mistral_qwen_fact_retrieval_failures.json",
            ),
            "mistral_qwen_graph_construction_failures": BenchmarkSubsetInfo(
                id="mistral_qwen_graph_construction_failures",
                label="Mistral + Qwen graph-construction failures",
                path="data/hotpotqa/mini_mistral_qwen_graph_construction_failures.json",
            ),
        },
        load=_load_hotpot_subset,
        count=_count_hotpot_subset,
    ),
    "musique": BenchmarkDataset(
        id="musique",
        label="MuSiQue",
        split="validation",
        default_subset="validation_3hop_plus",
        result_prefix="musique",
        source="Hugging Face dgslibisey/MuSiQue validation split",
        subsets={
            "validation": BenchmarkSubsetInfo(
                id="validation",
                label="Full MuSiQue validation",
                path=None,
            ),
            "validation_3hop_plus": BenchmarkSubsetInfo(
                id="validation_3hop_plus",
                label="MuSiQue validation, 3+ hops",
                path=None,
            ),
            "marked_failures_2026_06_28": BenchmarkSubsetInfo(
                id="marked_failures_2026_06_28",
                label="Marked MuSiQue failures, 2026-06-28",
                path="data/musique/marked_failures_2026_06_28.json",
            ),
        },
        load=_load_musique_subset,
        count=_count_musique_subset,
    ),
}


def get_benchmark_dataset(dataset_id: str) -> BenchmarkDataset:
    try:
        return DATASETS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark dataset: {dataset_id}") from exc


def load_benchmark_examples(
    dataset_id: str,
    subset: str,
    path: str | Path | None = None,
    limit: int | None = None,
) -> list[HotpotExample]:
    return get_benchmark_dataset(dataset_id).load(subset, path, limit)


def count_benchmark_examples(
    dataset_id: str,
    subset: str,
    path: str | Path | None = None,
) -> int:
    return get_benchmark_dataset(dataset_id).count(subset, path)
