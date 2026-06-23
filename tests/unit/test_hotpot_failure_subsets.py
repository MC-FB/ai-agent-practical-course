from __future__ import annotations

import json
from pathlib import Path

from dagqa.eval.hotpot_loader import count_hotpot_examples, load_hotpot_examples

FACT_RETRIEVAL_PATH = Path("data/hotpotqa/mini_mistral_qwen_fact_retrieval_failures.json")
GRAPH_CONSTRUCTION_PATH = Path("data/hotpotqa/mini_mistral_qwen_graph_construction_failures.json")
EXPECTED_FACT_RETRIEVAL_COUNT = 19
EXPECTED_GRAPH_CONSTRUCTION_COUNT = 6


def test_fact_retrieval_subset_contains_curated_semantic_failures() -> None:
    data = json.loads(FACT_RETRIEVAL_PATH.read_text(encoding="utf-8"))
    examples = data["data"]
    ids = [example["_id"] for example in examples]

    assert data["metadata"]["subset_id"] == "mistral_qwen_fact_retrieval_failures"
    assert data["metadata"]["count"] == EXPECTED_FACT_RETRIEVAL_COUNT
    assert len(ids) == len(set(ids))
    assert "5ae762835542997b22f6a711" in ids
    assert "5a8210f355429926c1cdae24" in ids
    assert "5a78e9d155429970f5fffdcc" not in ids
    assert "5ae712fa554299572ea546b9" not in ids


def test_graph_construction_subset_is_separate_from_fact_retrieval_subset() -> None:
    fact_ids = {
        example["_id"]
        for example in json.loads(FACT_RETRIEVAL_PATH.read_text(encoding="utf-8"))["data"]
    }
    data = json.loads(GRAPH_CONSTRUCTION_PATH.read_text(encoding="utf-8"))
    graph_ids = [example["_id"] for example in data["data"]]

    assert data["metadata"]["subset_id"] == "mistral_qwen_graph_construction_failures"
    assert data["metadata"]["count"] == EXPECTED_GRAPH_CONSTRUCTION_COUNT
    assert len(graph_ids) == len(set(graph_ids))
    assert set(graph_ids).isdisjoint(fact_ids)
    assert "5ae712fa554299572ea546b9" in graph_ids
    assert "5a78e9d155429970f5fffdcc" not in graph_ids


def test_failure_subsets_load_as_hotpot_examples() -> None:
    fact_examples = load_hotpot_examples(FACT_RETRIEVAL_PATH)
    graph_examples = load_hotpot_examples(GRAPH_CONSTRUCTION_PATH)

    assert count_hotpot_examples(FACT_RETRIEVAL_PATH) == EXPECTED_FACT_RETRIEVAL_COUNT
    assert count_hotpot_examples(GRAPH_CONSTRUCTION_PATH) == EXPECTED_GRAPH_CONSTRUCTION_COUNT
    assert fact_examples[0].id == "5a80799b5542992bc0c4a72e"
    assert fact_examples[0].context
    assert fact_examples[0].supporting_facts
    assert graph_examples[0].context
    assert graph_examples[0].supporting_facts
