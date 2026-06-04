from __future__ import annotations

import json
from pathlib import Path

from app import api
from dagqa.config import AppConfig
from dagqa.eval.benchmark import BenchmarkRecord
from dagqa.eval.hotpot_loader import HOTPOTQA_DISTRACTOR_VALIDATION_SIZE, HotpotExample
from dagqa.schemas import (
    DagNode,
    DagPlan,
    NodeStatus,
    NodeTrace,
    Operation,
    PromptSpec,
    TaskType,
)


def _single_node_run_trace() -> dict:
    plan = DagPlan(
        question="Who won?",
        final_node="answer",
        nodes=[
            DagNode(
                id="answer",
                label="Answer",
                task_type=TaskType.synthesis,
                question="Who won?",
                operation=Operation.answer,
                depends_on=[],
                prompt=PromptSpec(system="system", user_template="template"),
                input_map={},
                output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            )
        ],
    )
    trace = NodeTrace(
        node_id="answer",
        label="Answer",
        task_type=TaskType.synthesis,
        operation=Operation.answer,
        status=NodeStatus.succeeded,
        returned_value={"answer": "Ada"},
    )
    return {
        "run_id": "trace-1",
        "question": "Who won?",
        "plan": plan.model_dump(mode="json"),
        "waves": [{"index": 0, "node_ids": ["answer"]}],
        "nodes": [trace.model_dump(mode="json")],
        "final_answer": {"answer": "Ada"},
        "status": "succeeded",
        "total_duration_ms": 12.0,
    }


def test_normalize_saved_benchmark_record_rebuilds_missing_mermaid() -> None:
    normalized = api._normalize_benchmark_record(
        {
            "id": "example-1",
            "question": "Who won?",
            "gold_answer": "Ada",
            "prediction": "Ada",
            "exact_match": 1.0,
            "run_trace": _single_node_run_trace(),
        }
    )

    run_trace = normalized["run_trace"]

    assert run_trace is not None
    assert "flowchart TD" in run_trace["mermaid"]
    assert "click answer dagqaSelectGraphNode" in run_trace["mermaid"]


def test_normalize_saved_benchmark_payload_backfills_dataset_size(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "count_hotpot_examples",
        lambda data_path=None: HOTPOTQA_DISTRACTOR_VALIDATION_SIZE,
    )

    normalized = api._normalize_benchmark_payload(
        {"run_id": "run-1", "records": [{"id": "example-1"}]},
        Path("runs/benchmarks/run-1.json"),
    )

    assert normalized["dataset_size"] == HOTPOTQA_DISTRACTOR_VALIDATION_SIZE


def test_normalize_saved_benchmark_payload_preserves_provider_and_model(monkeypatch) -> None:
    monkeypatch.setattr(api, "count_hotpot_examples", lambda data_path=None: 1)

    normalized = api._normalize_benchmark_payload(
        {
            "run_id": "run-1",
            "provider": "cluster",
            "model": "google/gemma-4-31B-it",
            "records": [],
        },
        Path("runs/benchmarks/run-1.json"),
    )

    assert normalized["provider"] == "cluster"
    assert normalized["model"] == "google/gemma-4-31B-it"


def test_list_benchmark_results_sorts_by_created_at_newest_first(monkeypatch, tmp_path) -> None:
    older = tmp_path / "newer-file-mtime.json"
    newer = tmp_path / "older-file-mtime.json"
    older.write_text(json.dumps({"run_id": "older", "created_at": "2026-01-01T00:00:00Z"}))
    newer.write_text(json.dumps({"run_id": "newer", "created_at": "2026-02-01T00:00:00Z"}))
    older.touch()

    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path)
    monkeypatch.setattr(api, "count_hotpot_examples", lambda data_path=None: 2)

    result = api.list_benchmark_results()

    assert [item["run_id"] for item in result["results"]] == ["newer", "older"]


async def test_paired_live_benchmark_reuses_sample_and_saves_two_results(
    monkeypatch,
    tmp_path,
) -> None:
    system_count = 2
    total_completed = 6
    examples = [
        HotpotExample(id=str(index), question=f"q{index}", answer=f"a{index}") for index in range(6)
    ]
    seen: dict[str, list[str]] = {}

    async def run_examples(client, sampled, system, max_parallel, on_complete):  # noqa: ANN001, ARG001
        seen[system] = [example.id for example in sampled]
        records = [
            BenchmarkRecord(
                id=example.id,
                question=example.question,
                gold_answer=example.answer,
                prediction=example.answer,
                exact_match=1,
                f1=1,
                latency_ms=1,
            )
            for example in sampled
        ]
        for record in records:
            on_complete(record)
        return records

    monkeypatch.setattr(api, "load_hotpot_examples", lambda path=None: examples)
    monkeypatch.setattr(api, "_run_examples", run_examples)
    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path)
    cfg = AppConfig()
    dag_client = type("Client", (), {"config": cfg})()
    run_id = "coordinator"
    group_id = "comparison-1"
    api.LIVE_BENCHMARKS[run_id] = {"created_at": "2026-06-04T00:00:00Z"}

    await api._run_live_benchmark(
        run_id,
        api.BenchmarkRequest(limit=3, seed=123, systems=["dag_agent", "direct_llm"]),
        123,
        dag_client,  # type: ignore[arg-type]
        cfg,
        ["dag_agent", "direct_llm"],
        group_id,
    )

    live = api.LIVE_BENCHMARKS[run_id]
    saved = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]

    assert seen["dag_agent"] == seen["direct_llm"]
    assert live["phase"] == "complete"
    assert live["completed"] == total_completed
    assert len(live["comparison_run_ids"]) == system_count
    assert len(saved) == system_count
    assert {result["system"] for result in saved} == {"dag_agent", "direct_llm"}
    assert {result["seed"] for result in saved} == {123}
    assert {result["comparison_group_id"] for result in saved} == {group_id}
