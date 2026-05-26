from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from dagqa.client import DagQaClient
from dagqa.eval.hotpot_loader import HotpotExample, load_hotpot_examples
from dagqa.eval.metrics import answer_f1, exact_match
from dagqa.schemas import LLMRequest


class BenchmarkRecord(BaseModel):
    id: str
    question: str
    gold_answer: str
    prediction: str
    exact_match: float
    f1: float
    latency_ms: float
    llm_call_count: int | None = None
    node_count: int | None = None
    graph_depth: int | None = None
    structural_failure: bool = False
    error: str | None = None


class BenchmarkResult(BaseModel):
    system: str
    limit: int
    records: list[BenchmarkRecord]
    metrics: dict[str, float]


async def benchmark_hotpotqa(
    client: DagQaClient,
    *,
    system: Literal["direct_llm", "dag_agent"] = "dag_agent",
    limit: int = 100,
    path: str | Path | None = None,
) -> BenchmarkResult:
    examples = load_hotpot_examples(path, limit)
    records = []
    for example in examples:
        records.append(await _run_example(client, example, system))
    metrics = _aggregate(records)
    return BenchmarkResult(system=system, limit=limit, records=records, metrics=metrics)


async def _run_example(
    client: DagQaClient, example: HotpotExample, system: str
) -> BenchmarkRecord:
    started = time.perf_counter()
    try:
        if system == "direct_llm":
            response = await client.llm.complete(
                LLMRequest(
                    system="Answer the question concisely.",
                    prompt=f"Question: {example.question}\nReturn only the answer.",
                )
            )
            prediction = response.text.strip()
            llm_call_count = 1
            node_count = None
            graph_depth = None
            structural_failure = False
        else:
            run = await client.ask(example.question)
            prediction = _extract_answer(run.final_answer)
            llm_call_count = len(run.nodes) + 1
            node_count = len(run.plan.nodes)
            graph_depth = max((wave.index for wave in run.waves), default=0)
            structural_failure = run.status.value != "succeeded"
        return BenchmarkRecord(
            id=example.id,
            question=example.question,
            gold_answer=example.answer,
            prediction=prediction,
            exact_match=exact_match(prediction, example.answer),
            f1=answer_f1(prediction, example.answer),
            latency_ms=(time.perf_counter() - started) * 1000,
            llm_call_count=llm_call_count,
            node_count=node_count,
            graph_depth=graph_depth,
            structural_failure=structural_failure,
        )
    except Exception as exc:
        return BenchmarkRecord(
            id=example.id,
            question=example.question,
            gold_answer=example.answer,
            prediction="",
            exact_match=0.0,
            f1=0.0,
            latency_ms=(time.perf_counter() - started) * 1000,
            structural_failure=True,
            error=str(exc),
        )


def _extract_answer(final_answer: dict[str, Any] | None) -> str:
    if not final_answer:
        return ""
    value = final_answer.get("answer")
    return "" if value is None else str(value)


def _aggregate(records: list[BenchmarkRecord]) -> dict[str, float]:
    if not records:
        return {}
    return {
        "exact_match": sum(record.exact_match for record in records) / len(records),
        "f1": sum(record.f1 for record in records) / len(records),
        "avg_latency_ms": sum(record.latency_ms for record in records) / len(records),
        "structural_failure_rate": sum(record.structural_failure for record in records)
        / len(records),
    }


def write_jsonl(result: BenchmarkResult, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in result.records:
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")
        handle.write(json.dumps({"metrics": result.metrics, "system": result.system}) + "\n")


def run_benchmark_sync(*args: Any, **kwargs: Any) -> BenchmarkResult:
    return asyncio.run(benchmark_hotpotqa(*args, **kwargs))
