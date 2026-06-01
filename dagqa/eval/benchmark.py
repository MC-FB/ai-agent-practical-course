from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from dagqa.client import DagQaClient
from dagqa.eval.hotpot_loader import HotpotExample, load_hotpot_examples
from dagqa.eval.metrics import answer_f1, exact_match, cosine_sim
from dagqa.graph.render import render_mermaid
from dagqa.schemas import LLMRequest


class BenchmarkRecord(BaseModel):
    id: str
    question: str
    gold_answer: str
    prediction: str
    exact_match: float
    f1: float
    cosine_sim: float
    latency_ms: float
    llm_call_count: int | None = None
    node_count: int | None = None
    graph_depth: int | None = None
    structural_valid: bool | None = None
    structural_issues: list[str] = []
    structural_failure: bool = False
    run_trace: dict[str, Any] | None = None
    error: str | None = None


class BenchmarkResult(BaseModel):
    run_id: str
    system: str
    limit: int
    model: str
    dataset: str = "hotpotqa"
    split: str = "validation"
    seed: int
    created_at: str
    output_path: str | None = None
    total_runtime_ms: float
    records: list[BenchmarkRecord]
    metrics: dict[str, float]


async def benchmark_hotpotqa(
    client: DagQaClient,
    *,
    system: Literal["direct_llm", "dag_agent"] = "dag_agent",
    limit: int = 100,
    path: str | Path | None = None,
    seed: int | None = None,
) -> BenchmarkResult:
    started = time.perf_counter()
    seed = seed if seed is not None else random.SystemRandom().randint(1, 2_147_483_647)
    examples = _sample_examples(load_hotpot_examples(path), limit, seed)
    records = []
    for example in examples:
        records.append(await _run_example(client, example, system))
    metrics = _aggregate(records)
    total_runtime_ms = (time.perf_counter() - started) * 1000
    metrics["total_runtime_ms"] = total_runtime_ms
    return BenchmarkResult(
        run_id=str(uuid4()),
        system=system,
        limit=limit,
        model=client.config.llm.model,
        seed=seed,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_runtime_ms=total_runtime_ms,
        records=records,
        metrics=metrics,
    )


def _sample_examples(
    examples: list[HotpotExample],
    limit: int,
    seed: int,
) -> list[HotpotExample]:
    if limit >= len(examples):
        return examples
    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), limit)
    return [examples[index] for index in indices]


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
            structural_valid = None
            structural_issues = []
            structural_failure = False
        else:
            run = await client.ask(example.question)
            run_trace = run.model_dump(mode="json")
            run_trace["mermaid"] = render_mermaid(run.plan, run.nodes)
            prediction = _extract_answer(run.final_answer)
            llm_call_count = len(run.nodes) + 1
            node_count = len(run.plan.nodes)
            graph_depth = max((wave.index for wave in run.waves), default=0)
            structural_issues = _structural_issues(run)
            structural_valid = not structural_issues
            structural_failure = run.status.value != "succeeded"
        if system == "direct_llm":
            run_trace = None
        return BenchmarkRecord(
            id=example.id,
            question=example.question,
            gold_answer=example.answer,
            prediction=prediction,
            
            exact_match=exact_match(prediction, example.answer),
            f1=answer_f1(prediction, example.answer),
            cosine_sim=cosine_sim(prediction,example.answer),

            latency_ms=(time.perf_counter() - started) * 1000,
            llm_call_count=llm_call_count,
            node_count=node_count,
            graph_depth=graph_depth,
            structural_valid=structural_valid,
            structural_issues=structural_issues,
            structural_failure=structural_failure,
            run_trace=run_trace,
        )
    except Exception as exc:
        return BenchmarkRecord(
            id=example.id,
            question=example.question,
            gold_answer=example.answer,
            prediction="",
            exact_match=0.0,
            f1=0.0,
            cosine_sim=-1.5, 
            latency_ms=(time.perf_counter() - started) * 1000,
            structural_failure=True,
            error=str(exc),
        )


def _extract_answer(final_answer: dict[str, Any] | None) -> str:
    if not final_answer:
        return ""
    value = final_answer.get("answer")
    if value is not None:
        return str(value)
    non_reasoning_values = [
        value
        for key, value in final_answer.items()
        if key not in {"reasoning", "confidence"} and value is not None
    ]
    if len(non_reasoning_values) == 1:
        return str(non_reasoning_values[0])
    return ""


def _structural_issues(run: Any) -> list[str]:
    issues: list[str] = []
    by_trace = {trace.node_id: trace for trace in run.nodes}
    for node in run.plan.nodes:
        if not node.depends_on:
            continue
        if not node.input_map:
            issues.append(f"{node.id}: dependent node has empty input_map")
            continue
        consumed = {reference.split(".", 1)[0] for reference in node.input_map.values()}
        missing = sorted(set(node.depends_on) - consumed)
        if missing:
            issues.append(f"{node.id}: input_map missing dependencies {missing}")
        template = node.prompt.user_template
        uses_dependencies = "{dependencies}" in template
        uses_input_names = any("{" + name + "}" in template for name in node.input_map)
        uses_direct_refs = any("{" + ref + "}" in template for ref in node.input_map.values())
        if not (uses_dependencies or uses_input_names or uses_direct_refs):
            issues.append(f"{node.id}: prompt template does not use child values")
    final_node = next(node for node in run.plan.nodes if node.id == run.plan.final_node)
    final_trace = by_trace.get(run.plan.final_node)
    if "answer" not in final_node.output_schema.get("properties", {}):
        issues.append(f"{final_node.id}: final schema missing answer field")
    if final_node.depends_on and not (final_trace and final_trace.dependency_values):
        issues.append(f"{final_node.id}: final trace has no dependency_values")
    return issues


def _aggregate(records: list[BenchmarkRecord]) -> dict[str, float]:
    if not records:
        return {}
    success_count = sum(
        1 for record in records if not record.structural_failure and not record.error
    )
    failure_count = len(records) - success_count
    error_count = sum(1 for record in records if record.error)
    llm_call_records = [
        record.llm_call_count for record in records if record.llm_call_count is not None
    ]
    node_count_records = [record.node_count for record in records if record.node_count is not None]
    graph_depth_records = [
        record.graph_depth for record in records if record.graph_depth is not None
    ]
    sorted_latencies = sorted(record.latency_ms for record in records)
    p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2]
    cosine_values = [
        cosine_sim(record.prediction, record.gold_answer)
        for record in records
        if record.prediction or record.gold_answer
    ]
    return {
        "example_count": float(len(records)),
        "success_count": float(success_count),
        "failure_count": float(failure_count),
        "error_count": float(error_count),
        "exact_match": sum(record.exact_match for record in records) / len(records),
        "f1": sum(record.f1 for record in records) / len(records),
        "cosine_sim": sum(cosine_values) / len(cosine_values) if cosine_values else 0.0,
        "avg_latency_ms": sum(record.latency_ms for record in records) / len(records),
        "p50_latency_ms": p50_latency_ms,
        "total_llm_call_count": float(sum(llm_call_records)),
        "avg_llm_call_count": (
            sum(llm_call_records) / len(llm_call_records) if llm_call_records else 0.0
        ),
        "avg_node_count": (
            sum(node_count_records) / len(node_count_records) if node_count_records else 0.0
        ),
        "avg_graph_depth": (
            sum(graph_depth_records) / len(graph_depth_records) if graph_depth_records else 0.0
        ),
        "structural_valid_rate": sum(
            1 for record in records if record.structural_valid is not False
        )
        / len(records),
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
