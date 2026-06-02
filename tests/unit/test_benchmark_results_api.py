from __future__ import annotations

from pathlib import Path

from app import api
from dagqa.eval.hotpot_loader import HOTPOTQA_DISTRACTOR_VALIDATION_SIZE
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
