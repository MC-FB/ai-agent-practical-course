from __future__ import annotations

import asyncio
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


def test_list_benchmark_results_includes_benchmark_name(monkeypatch, tmp_path) -> None:
    result_file = tmp_path / "named.json"
    result_file.write_text(
        json.dumps(
            {
                "run_id": "named-run",
                "name": "First big benchmark",
                "created_at": "2026-06-05T08:00:00Z",
                "system": "dag_agent",
            }
        )
    )

    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path)
    monkeypatch.setattr(api, "count_hotpot_examples", lambda data_path=None: 2)

    result = api.list_benchmark_results()

    assert result["results"][0]["name"] == "First big benchmark"


def test_list_benchmark_results_marks_partial_runs(monkeypatch, tmp_path) -> None:
    result_file = tmp_path / "partial.json"
    result_file.write_text(
        json.dumps(
            {
                "run_id": "partial-run",
                "limit": 10,
                "records": [{"id": "one"}],
                "metrics": {"example_count": 1},
            }
        )
    )

    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path)
    monkeypatch.setattr(api, "count_hotpot_examples", lambda data_path=None: 10)

    result = api.list_benchmark_results()

    assert result["results"][0]["completed"] == 1
    assert result["results"][0]["partial"] is True


def test_list_benchmark_results_hides_superseded_partial_runs(monkeypatch, tmp_path) -> None:
    shared = {
        "comparison_group_id": "group-1",
        "system": "direct_llm",
        "limit": 10,
        "seed": 123,
        "model": "model-a",
        "created_at": "2026-06-05T00:00:00Z",
    }
    (tmp_path / "partial.json").write_text(
        json.dumps(
            {
                **shared,
                "run_id": "partial-run",
                "records": [{"id": "one"}],
                "metrics": {"example_count": 1},
            }
        )
    )
    (tmp_path / "complete.json").write_text(
        json.dumps(
            {
                **shared,
                "run_id": "complete-run",
                "records": [{"id": str(index)} for index in range(10)],
                "metrics": {"example_count": 10},
            }
        )
    )

    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path)
    monkeypatch.setattr(api, "count_hotpot_examples", lambda data_path=None: 10)

    result = api.list_benchmark_results()

    assert [item["run_id"] for item in result["results"]] == ["complete-run"]
    assert result["results"][0]["partial"] is False


def test_write_benchmark_result_removes_matching_partial_result(monkeypatch, tmp_path) -> None:
    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps(
            {
                "run_id": "partial-run",
                "comparison_group_id": "group-1",
                "system": "direct_llm",
                "limit": 2,
                "seed": 123,
                "model": "model-a",
                "records": [{"id": "one"}],
            }
        )
    )
    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path)
    result = api.BenchmarkResult(
        run_id="complete-run",
        comparison_group_id="group-1",
        system="direct_llm",
        limit=2,
        model="model-a",
        seed=123,
        created_at="2026-06-05T00:00:00Z",
        total_runtime_ms=1,
        records=[
            BenchmarkRecord(
                id=str(index),
                question=f"q{index}",
                gold_answer=f"a{index}",
                prediction=f"a{index}",
                exact_match=1,
                f1=1,
                latency_ms=1,
            )
            for index in range(2)
        ],
        metrics={"example_count": 2},
    )

    api._write_benchmark_result(result)

    assert not partial.exists()
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_get_live_benchmark_marks_orphaned_running_state_stopped(monkeypatch, tmp_path) -> None:
    run_id = "orphaned-run"
    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path)
    api.LIVE_BENCHMARKS.pop(run_id, None)
    api.LIVE_BENCHMARK_TASKS.pop(run_id, None)
    live_dir = tmp_path / ".live"
    live_dir.mkdir()
    (live_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "phase": "running",
                "status": "running",
                "records": [],
                "metrics": {},
            }
        )
    )

    state = api.get_live_benchmark(run_id)

    assert state["phase"] == "stopped"
    assert state["resumable"] is True


def test_list_live_benchmarks_returns_unfinished_summaries(monkeypatch, tmp_path) -> None:
    completed_count = 1337
    total_count = 2000
    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path)
    api.LIVE_BENCHMARKS.clear()
    api.LIVE_BENCHMARK_TASKS.clear()
    live_dir = tmp_path / ".live"
    live_dir.mkdir()
    (live_dir / "unfinished-run.json").write_text(
        json.dumps(
            {
                "run_id": "unfinished-run",
                "name": "overnight",
                "phase": "running",
                "status": "running",
                "systems": ["dag_agent", "direct_llm"],
                "current_system": "direct_llm",
                "completed": completed_count,
                "total": total_count,
                "model": "openai/gpt-oss-120b",
                "created_at": "2026-06-05T13:55:13Z",
                "records": [],
                "metrics": {},
            }
        )
    )
    (live_dir / "complete-run.json").write_text(
        json.dumps(
            {
                "run_id": "complete-run",
                "phase": "complete",
                "status": "succeeded",
                "completed": 10,
                "total": 10,
                "records": [],
                "metrics": {},
            }
        )
    )

    result = api.list_live_benchmarks()

    assert [item["run_id"] for item in result["runs"]] == ["unfinished-run"]
    summary = result["runs"][0]
    assert summary["phase"] == "stopped"
    assert summary["completed"] == completed_count
    assert summary["total"] == total_count
    assert summary["model"] == "openai/gpt-oss-120b"
    assert "records" not in summary


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
        api.BenchmarkRequest(
            name="paired smoke",
            limit=3,
            seed=123,
            systems=["dag_agent", "direct_llm"],
        ),
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
    assert live["name"] == "paired smoke"
    assert {result["name"] for result in saved} == {"paired smoke"}
    assert {result["system"] for result in saved} == {"dag_agent", "direct_llm"}
    assert {result["seed"] for result in saved} == {123}
    assert {result["comparison_group_id"] for result in saved} == {group_id}


async def test_hotpotqa_preflight_checks_dataset_storage_and_model(
    monkeypatch,
    tmp_path,
) -> None:
    data_path = tmp_path / "hotpot.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "_id": "example-1",
                    "question": "Who?",
                    "answer": "Ada",
                    "context": [["Doc", ["Ada is named."]]],
                    "supporting_facts": [["Doc", 0]],
                }
            ]
        )
    )

    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path / "runs")
    api.LIVE_BENCHMARKS.clear()

    async def cluster_models() -> list[str]:
        return ["model-a"]

    monkeypatch.setattr(api, "_cluster_models", cluster_models)

    result = await api.hotpotqa_preflight(
        api.BenchmarkRequest(
            limit=1,
            systems=["direct_llm"],
            data_path=str(data_path),
            llm=api.LLMSelection(provider="cluster", model="model-a"),
        )
    )

    assert result.ok is True
    assert {check.name for check in result.checks} >= {
        "dataset",
        "storage",
        "model_catalog",
    }


async def test_live_benchmark_stop_saves_partial_result(monkeypatch, tmp_path) -> None:
    examples = [
        HotpotExample(id=str(index), question=f"q{index}", answer=f"a{index}") for index in range(4)
    ]
    stop_event = asyncio.Event()

    async def run_examples(client, sampled, system, max_parallel, on_complete, should_stop):  # noqa: ANN001, ARG001
        record = BenchmarkRecord(
            id=sampled[0].id,
            question=sampled[0].question,
            gold_answer=sampled[0].answer,
            prediction=sampled[0].answer,
            exact_match=1,
            f1=1,
            latency_ms=1,
        )
        on_complete(record)
        stop_event.set()
        return [record]

    monkeypatch.setattr(api, "load_hotpot_examples", lambda path=None: examples)
    monkeypatch.setattr(api, "_run_examples", run_examples)
    monkeypatch.setattr(api, "_benchmark_output_dir", lambda: tmp_path)
    cfg = AppConfig()
    dag_client = type("Client", (), {"config": cfg})()
    run_id = "partial-stop"
    api.LIVE_BENCHMARKS[run_id] = {"created_at": "2026-06-04T00:00:00Z"}

    await api._run_live_benchmark(
        run_id,
        api.BenchmarkRequest(limit=4, seed=123, systems=["dag_agent"]),
        123,
        dag_client,  # type: ignore[arg-type]
        cfg,
        ["dag_agent"],
        None,
        stop_event,
    )

    live = api.LIVE_BENCHMARKS[run_id]
    saved = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]

    assert live["phase"] == "stopped"
    assert live["completed"] == 1
    assert saved[0]["records"][0]["id"] == "0"


async def test_live_benchmark_resume_skips_completed_records(monkeypatch, tmp_path) -> None:
    expected_completed = 3
    examples = [
        HotpotExample(id=str(index), question=f"q{index}", answer=f"a{index}") for index in range(3)
    ]
    seen: list[str] = []

    async def run_examples(client, sampled, system, max_parallel, on_complete):  # noqa: ANN001, ARG001
        seen.extend(example.id for example in sampled)
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
    run_id = "resume-run"
    api.LIVE_BENCHMARKS[run_id] = {
        "created_at": "2026-06-04T00:00:00Z",
        "current_system": "dag_agent",
        "records": [
            BenchmarkRecord(
                id="0",
                question="q0",
                gold_answer="a0",
                prediction="a0",
                exact_match=1,
                f1=1,
                latency_ms=1,
            ).model_dump(mode="json")
        ],
    }

    await api._run_live_benchmark(
        run_id,
        api.BenchmarkRequest(limit=3, seed=123, systems=["dag_agent"]),
        123,
        dag_client,  # type: ignore[arg-type]
        cfg,
        ["dag_agent"],
    )

    assert seen == ["1", "2"]
    assert api.LIVE_BENCHMARKS[run_id]["phase"] == "complete"
    assert api.LIVE_BENCHMARKS[run_id]["completed"] == expected_completed
