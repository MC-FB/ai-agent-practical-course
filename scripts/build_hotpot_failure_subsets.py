from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FACT_RETRIEVAL_COSINE_THRESHOLD = 0.5

FACT_RETRIEVAL_ALLOWLIST: dict[str, str] = {
    "5ae762835542997b22f6a711": "wrong_location",
    "5a8210f355429926c1cdae24": "wrong_award",
}

FORMATTING_OR_ALIAS_ONLY_IDS = {
    "5a78e9d155429970f5fffdcc",
    "5a7602d8554299109176e5fb",
    "5a7a9a2255429941d65f26eb",
    "5a776fc15542997042120a3a",
    "5ab959905542996be2020497",
    "5a77731955429972597f1541",
    "5a72a9ab5542992359bc315a",
    "5ae09b7055429906c02daae4",
    "5adf33a05542993344016c22",
    "5ab6c181554299110f219a5f",
    "5a82626655429966c78a6a08",
    "5a7dc7735542990b8f503a93",
    "5a79257b55429974737f79a7",
    "5a7aa9425542992d025e66e4",
    "5a81e097554299676cceb129",
    "5a890b895542995153361278",
    "5a7502715542996c70cfae74",
    "5a8f155e554299458435d54c",
    "5a7319bc5542992359bc3235",
    "5ab92f1f554299753720f77b",
    "5a732ab25542991f29ee2d25",
    "5ac5275755429924173fb617",
    "5abc436f5542993a06baf8c3",
    "5ab3d2b7554299233954ffb8",
    "5ab614e5554299488d4d9a9c",
    "5a806aca5542996402f6a503",
    "5a7b1c6b55429931da12c9ca",
    "5ae17d6855429901ffe4aea7",
    "5a7d119d5542995ed0d165d5",
}


@dataclass(frozen=True)
class SourceRun:
    key: str
    path: Path
    direct_path: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistral-dag", type=Path, required=True)
    parser.add_argument("--mistral-direct", type=Path, required=True)
    parser.add_argument("--qwen-dag", type=Path, required=True)
    parser.add_argument("--qwen-direct", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/hotpotqa"))
    args = parser.parse_args()

    sources = [
        SourceRun("mistral", args.mistral_dag, args.mistral_direct),
        SourceRun("qwen", args.qwen_dag, args.qwen_direct),
    ]
    fact_items: OrderedDict[str, dict[str, Any]] = OrderedDict()
    graph_items: OrderedDict[str, dict[str, Any]] = OrderedDict()
    source_summaries = []

    for source in sources:
        dag_run = _read_json(source.path)
        direct_records = _read_json(source.direct_path)["records"]
        direct_by_id = {record["id"]: record for record in direct_records}
        source_summaries.append(
            {
                "key": source.key,
                "run_id": dag_run["run_id"],
                "model": dag_run["model"],
                "artifact": source.path.name,
            }
        )
        for record in dag_run["records"]:
            if _is_graph_construction_failure(record):
                _add_item(graph_items, record, dag_run, direct_by_id, "graph_construction")
            elif _is_fact_retrieval_failure(record):
                _add_item(fact_items, record, dag_run, direct_by_id, "fact_retrieval")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_subset(
        args.output_dir / "mini_mistral_qwen_fact_retrieval_failures.json",
        "mistral_qwen_fact_retrieval_failures",
        fact_items,
        source_summaries,
        {
            "cosine_threshold": FACT_RETRIEVAL_COSINE_THRESHOLD,
            "allowlist": FACT_RETRIEVAL_ALLOWLIST,
            "excluded_formatting_or_alias_only_ids": sorted(FORMATTING_OR_ALIAS_ONLY_IDS),
        },
    )
    _write_subset(
        args.output_dir / "mini_mistral_qwen_graph_construction_failures.json",
        "mistral_qwen_graph_construction_failures",
        graph_items,
        source_summaries,
        {
            "criteria": [
                "structural_failure == true",
                "record error mentions planner, invalid DAG, dependency, or input_map",
            ]
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _is_graph_construction_failure(record: dict[str, Any]) -> bool:
    error = str(record.get("error") or "").lower()
    return bool(record.get("structural_failure")) or any(
        token in error
        for token in (
            "planner",
            "invalid dag",
            "depends on",
            "dependency",
            "input_map",
            "structured",
        )
    )


def _is_fact_retrieval_failure(record: dict[str, Any]) -> bool:
    record_id = record["id"]
    if record_id in FORMATTING_OR_ALIAS_ONLY_IDS:
        return False
    if not str(record.get("prediction") or "").strip():
        return False
    if record.get("error") or record.get("structural_failure"):
        return False
    cosine = record.get("cosine_sim")
    return (
        isinstance(cosine, int | float) and cosine < FACT_RETRIEVAL_COSINE_THRESHOLD
    ) or record_id in FACT_RETRIEVAL_ALLOWLIST


def _add_item(
    items: OrderedDict[str, dict[str, Any]],
    record: dict[str, Any],
    run: dict[str, Any],
    direct_by_id: dict[str, dict[str, Any]],
    category: str,
) -> None:
    record_id = record["id"]
    if record_id not in items:
        items[record_id] = _hotpot_item(record, direct_by_id.get(record_id))
    item = items[record_id]
    metadata = item.setdefault("metadata", {})
    sources = metadata.setdefault("failure_sources", [])
    if not any(source["run_id"] == run["run_id"] for source in sources):
        sources.append(
            {
                "category": category,
                "run_id": run["run_id"],
                "model": run["model"],
                "system": run["system"],
                "prediction": record.get("prediction"),
                "gold_answer": record.get("gold_answer"),
                "exact_match": record.get("exact_match"),
                "f1": record.get("f1"),
                "cosine_sim": record.get("cosine_sim"),
                "structural_failure": record.get("structural_failure"),
                "error": record.get("error"),
                "rationale": FACT_RETRIEVAL_ALLOWLIST.get(record_id),
            }
        )


def _hotpot_item(record: dict[str, Any], fallback_record: dict[str, Any] | None) -> dict[str, Any]:
    docs = _documents_from_record(record) or (
        _documents_from_record(fallback_record) if fallback_record else []
    )
    if not docs:
        raise ValueError(f"Could not find source documents for {record['id']}")
    return {
        "_id": record["id"],
        "id": record["id"],
        "question": record["question"],
        "answer": record["gold_answer"],
        "context": [
            [
                str(document["title"]),
                list(
                    (document.get("metadata") or {}).get("sentences") or [document.get("text", "")]
                ),
            ]
            for document in docs
        ],
        "supporting_facts": [
            [fact["title"], fact["sentence_index"]]
            for fact in (
                record.get("gold_supporting_facts")
                or (fallback_record or {}).get("gold_supporting_facts")
                or []
            )
        ],
        "metadata": {},
    }


def _documents_from_record(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not record:
        return []
    for node in (record.get("run_trace") or {}).get("nodes", []):
        documents = (node.get("supporting_evidence") or {}).get("documents") or []
        if documents:
            return documents
    return []


def _write_subset(
    path: Path,
    subset_id: str,
    items: OrderedDict[str, dict[str, Any]],
    source_summaries: list[dict[str, Any]],
    criteria: dict[str, Any],
) -> None:
    data = list(items.values())
    payload = {
        "metadata": {
            "subset_id": subset_id,
            "description": (
                "Curated HotpotQA failure subset from Mistral and Qwen DAG benchmark runs."
            ),
            "source_runs": source_summaries,
            "criteria": criteria,
            "count": len(data),
        },
        "data": data,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {path} ({len(data)} examples)")


if __name__ == "__main__":
    main()
