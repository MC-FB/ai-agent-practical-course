from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, model_validator

from dagqa.client import DagQaClient
from dagqa.eval.answer_postprocess import (
    answer_value_to_text,
    canonicalize_prediction,
    is_placeholder_or_unsupported,
)
from dagqa.eval.hotpot_loader import (
    HOTPOTQA_DISTRACTOR_VALIDATION_SIZE,
    HotpotExample,
    load_hotpot_examples,
)
from dagqa.eval.metrics import ANSWER_METRICS, MetricSample
from dagqa.graph.render import render_mermaid
from dagqa.nodes.output_validation import parse_node_output
from dagqa.nodes.prompts import render_evidence_section
from dagqa.schemas import (
    DagNode,
    DagPlan,
    EvidenceCitation,
    EvidenceCitationEvaluation,
    EvidenceSelection,
    GoldSupportingFact,
    LLMRequest,
    NodeStatus,
    NodeTrace,
    Operation,
    PromptSpec,
    RunTrace,
    SchedulerWave,
    TaskType,
    ValidationResult,
)


class BenchmarkRecord(BaseModel):
    id: str
    question: str
    gold_answer: str
    prediction: str
    raw_prediction: str | None = None
    metric_scores: dict[str, float] = Field(default_factory=dict)
    latency_ms: float
    llm_call_count: int | None = None
    llm_retry_count: int | None = None
    node_count: int | None = None
    graph_depth: int | None = None
    structural_valid: bool | None = None
    structural_issues: list[str] = []
    structural_failure: bool = False
    gold_supporting_facts: list[GoldSupportingFact] = Field(default_factory=list)
    evidence_citation_count: int | None = None
    correct_evidence_citation_count: int | None = None
    wrong_evidence_citation_count: int | None = None
    wrong_supporting_text_rate: float | None = None
    gold_supporting_fact_recall: float | None = None
    run_trace: dict[str, Any] | None = None
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_metric_scores(cls, data: Any) -> Any:
        """Fold legacy top-level metric keys into ``metric_scores``.

        Keeps old JSON payloads and call sites that still pass ``exact_match=`` /
        ``f1=`` / ``cosine_sim=`` working after the switch to a dynamic metric dict.
        """
        if not isinstance(data, dict) or data.get("metric_scores"):
            return data
        scores = {
            metric.name: data[metric.name]
            for metric in ANSWER_METRICS
            if data.get(metric.name) is not None
        }
        if scores:
            data = {**data, "metric_scores": scores}
        return data

    @computed_field
    @property
    def exact_match(self) -> float:
        return self.metric_scores.get("exact_match", 0.0)

    @computed_field
    @property
    def f1(self) -> float:
        return self.metric_scores.get("f1", 0.0)

    @computed_field
    @property
    def cosine_sim(self) -> float:
        return self.metric_scores.get("cosine_sim", 0.0)


class BenchmarkResult(BaseModel):
    run_id: str
    name: str | None = None
    comparison_group_id: str | None = None
    system: str
    limit: int
    provider: str | None = None
    model: str
    dataset: str = "hotpotqa"
    split: str = "validation"
    subset: str = "validation"
    dataset_size: int | None = None
    seed: int
    max_parallel_examples: int = 1
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
    subset: str = "validation",
    max_parallel_examples: int | None = None,
    comparison_group_id: str | None = None,
    name: str | None = None,
    on_complete: Callable[[BenchmarkRecord], None] | None = None,
    delay_between_examples_seconds: float = 0.0,
) -> BenchmarkResult:
    started = time.perf_counter()
    seed = seed if seed is not None else random.SystemRandom().randint(1, 2_147_483_647)
    if limit <= 0 and path is None:
        all_examples = []
        dataset_size = HOTPOTQA_DISTRACTOR_VALIDATION_SIZE
        examples = []
    else:
        all_examples = load_hotpot_examples(path)
        dataset_size = len(all_examples)
        examples = _sample_examples(all_examples, limit, seed)
    parallel_examples = (
        max_parallel_examples
        if max_parallel_examples is not None
        else client.config.benchmark.max_parallel_examples
    )
    if parallel_examples < 1:
        raise ValueError("max_parallel_examples must be at least 1.")
    records = await _run_examples(
        client,
        examples,
        system,
        parallel_examples,
        on_complete=on_complete,
        delay_between_examples_seconds=delay_between_examples_seconds,
    )
    metrics = _aggregate(records)
    total_runtime_ms = (time.perf_counter() - started) * 1000
    metrics["total_runtime_ms"] = total_runtime_ms
    return BenchmarkResult(
        run_id=str(uuid4()),
        name=name,
        comparison_group_id=comparison_group_id,
        system=system,
        limit=limit,
        provider=client.config.llm.provider,
        model=client.config.llm.model,
        subset=subset,
        dataset_size=dataset_size,
        seed=seed,
        max_parallel_examples=parallel_examples,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_runtime_ms=total_runtime_ms,
        records=records,
        metrics=metrics,
    )


async def _run_examples(
    client: DagQaClient,
    examples: list[HotpotExample],
    system: str,
    max_parallel_examples: int,
    on_complete: Callable[[BenchmarkRecord], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    delay_between_examples_seconds: float = 0.0,
) -> list[BenchmarkRecord]:
    if not examples:
        return []

    queue: asyncio.Queue[tuple[int, HotpotExample]] = asyncio.Queue()
    for index, example in enumerate(examples):
        queue.put_nowait((index, example))

    records_by_index: dict[int, BenchmarkRecord] = {}
    worker_count = min(max_parallel_examples, len(examples))

    async def worker() -> None:
        while not queue.empty():
            if should_stop is not None and should_stop():
                return
            try:
                index, example = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                record = await _run_example_with_empty_prediction_retries(client, example, system)
            finally:
                queue.task_done()
            records_by_index[index] = record
            if on_complete is not None:
                on_complete(record)
            if delay_between_examples_seconds > 0:
                await asyncio.sleep(delay_between_examples_seconds)

    await asyncio.gather(*(worker() for _ in range(worker_count)))
    return [records_by_index[index] for index in sorted(records_by_index)]


async def _run_example_with_empty_prediction_retries(
    client: DagQaClient,
    example: HotpotExample,
    system: str,
) -> BenchmarkRecord:
    retries = _empty_prediction_retries(client)
    record = await _run_example(client, example, system)
    for _attempt in range(1, retries + 1):
        if record.prediction.strip():
            return record
        retry_record = await _run_example(client, example, system)
        retry_record.llm_retry_count = (retry_record.llm_retry_count or 0) + (
            record.llm_retry_count or 0
        )
        if not retry_record.prediction.strip() and record.error and not retry_record.error:
            retry_record.error = record.error
        record = retry_record
    return record


def _empty_prediction_retries(client: DagQaClient) -> int:
    config = getattr(client, "config", None)
    benchmark = getattr(config, "benchmark", None)
    return int(getattr(benchmark, "empty_prediction_retries", 0))


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


async def _run_example(  # noqa: PLR0915
    client: DagQaClient,
    example: HotpotExample,
    system: str,
) -> BenchmarkRecord:
    started = time.perf_counter()
    try:
        if system == "direct_llm":
            evidence = EvidenceSelection(
                strategy="all_documents",
                total_available=len(example.context),
                documents=example.context,
            )
            prompt = (
                f"Question: {example.question}"
                f"{render_evidence_section(evidence)}"
                "\nReturn JSON only with this exact shape:\n"
                '{"answer": "concise answer", "_evidence_citations": ['
                '{"document_id": "exact document ID", "title": "exact title", '
                '"sentence_indices": [0], "fact": "directly supporting fact"}]}\n'
                "The citations array must contain the sentence-level evidence directly used "
                "to determine the answer."
            )
            system_prompt = "Answer the question concisely using only the supplied sources."
            response_text: str | None = None
            parsed: dict[str, Any] | None = None
            try:
                response = await client.llm.complete(
                    LLMRequest(
                        system=system_prompt,
                        prompt=prompt,
                    )
                )
                response_text = response.text
                parsed = parse_node_output(response.text)
                raw_prediction = answer_value_to_text(parsed.get("answer", "")).strip()
                prediction = canonicalize_prediction(
                    raw_prediction,
                    question=example.question,
                    evidence_documents=example.context,
                    supporting_texts=[
                        str(citation.get("fact", ""))
                        for citation in parsed.get("_evidence_citations", [])
                        if isinstance(citation, dict)
                    ],
                )
                citations = [
                    EvidenceCitation.model_validate(citation)
                    for citation in parsed.get("_evidence_citations", [])
                ]
                if not prediction:
                    raise ValueError("Single-prompt response is missing a non-empty answer.")
                if not citations:
                    raise ValueError("Single-prompt response is missing sentence-level citations.")
            except Exception as exc:
                return _failed_direct_record(
                    example,
                    started,
                    system_prompt,
                    prompt,
                    response_text,
                    parsed,
                    exc,
                    client.config.model_dump(mode="json"),
                )
            returned_value = {"answer": raw_prediction}
            plan = DagPlan(
                question=example.question,
                nodes=[
                    DagNode(
                        id="single_prompt",
                        label="Single prompt",
                        task_type=TaskType.synthesis,
                        operation=Operation.answer,
                        question=example.question,
                        prompt=PromptSpec(
                            system=system_prompt,
                            user_template=prompt,
                        ),
                        output_schema={
                            "type": "object",
                            "required": ["answer"],
                            "properties": {"answer": {"type": "string"}},
                        },
                    )
                ],
                final_node="single_prompt",
            )
            trace = NodeTrace(
                node_id="single_prompt",
                label="Single prompt",
                task_type=TaskType.synthesis,
                operation=Operation.answer,
                status=NodeStatus.succeeded,
                resolved_question=example.question,
                rendered_prompt=prompt,
                raw_response=response.text,
                parsed_output=parsed,
                returned_value=returned_value,
                supporting_evidence=evidence,
                evidence_citations=citations,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            direct_run = RunTrace(
                run_id=str(uuid4()),
                question=example.question,
                plan=plan,
                waves=[SchedulerWave(index=0, node_ids=["single_prompt"])],
                nodes=[trace],
                final_answer=returned_value,
                status=NodeStatus.succeeded,
                total_duration_ms=(time.perf_counter() - started) * 1000,
                config=client.config.model_dump(mode="json"),
            )
            evidence_metrics = _evaluate_evidence_citations(
                direct_run,
                example.supporting_facts,
            )
            run_trace = direct_run.model_dump(mode="json")
            run_trace["mermaid"] = render_mermaid(direct_run.plan, direct_run.nodes)
            llm_call_count = 1
            llm_retry_count = int(response.metadata.get("retry_count", 0))
            node_count = 1
            graph_depth = 0
            structural_valid = True
            structural_issues = []
            structural_failure = False
        else:
            run = await client.ask(example.question, evidence_documents=example.context)
            evidence_metrics = _evaluate_evidence_citations(run, example.supporting_facts)
            raw_prediction = _extract_answer(run.final_answer)
            repair_llm_calls = 0
            repair_retry_count = 0
            if is_placeholder_or_unsupported(
                raw_prediction,
                question=example.question,
                evidence_documents=example.context,
            ):
                repaired, repair_retry_count = await _repair_unsupported_prediction(
                    client,
                    example,
                    raw_prediction,
                    run_trace=run.model_dump(mode="json"),
                )
                repair_llm_calls = 1
                if repaired:
                    raw_prediction = repaired
                    run.final_answer = {"answer": repaired}
            prediction = canonicalize_prediction(
                raw_prediction,
                question=example.question,
                evidence_documents=example.context,
                supporting_texts=_supporting_texts_from_run(run),
            )
            run_trace = run.model_dump(mode="json")
            run_trace["mermaid"] = render_mermaid(run.plan, run.nodes)
            llm_call_count = len(run.nodes) + 1 + repair_llm_calls
            llm_retry_count = sum(trace.llm_retry_count for trace in run.nodes) + repair_retry_count
            node_count = len(run.plan.nodes)
            graph_depth = max((wave.index for wave in run.waves), default=0)
            structural_issues = _structural_issues(run)
            structural_valid = not structural_issues
            structural_failure = run.status.value != "succeeded"
        return BenchmarkRecord(
            id=example.id,
            question=example.question,
            gold_answer=example.answer,
            prediction=prediction,
            raw_prediction=raw_prediction if raw_prediction != prediction else None,
            metric_scores={
                metric.name: metric.score(prediction, example.answer) for metric in ANSWER_METRICS
            },
            latency_ms=(time.perf_counter() - started) * 1000,
            llm_call_count=llm_call_count,
            llm_retry_count=llm_retry_count,
            node_count=node_count,
            graph_depth=graph_depth,
            structural_valid=structural_valid,
            structural_issues=structural_issues,
            structural_failure=structural_failure,
            gold_supporting_facts=example.supporting_facts,
            **evidence_metrics,
            run_trace=run_trace,
        )
    except Exception as exc:
        return BenchmarkRecord(
            id=example.id,
            question=example.question,
            gold_answer=example.answer,
            prediction="",
            metric_scores={metric.name: metric.error_default for metric in ANSWER_METRICS},
            latency_ms=(time.perf_counter() - started) * 1000,
            structural_failure=True,
            error=str(exc),
        )


def _failed_direct_record(
    example: HotpotExample,
    started: float,
    system_prompt: str,
    prompt: str,
    response_text: str | None,
    parsed_output: dict[str, Any] | None,
    exc: Exception,
    config: dict[str, Any],
) -> BenchmarkRecord:
    duration_ms = (time.perf_counter() - started) * 1000
    error = str(exc) or exc.__class__.__name__
    plan = DagPlan(
        question=example.question,
        nodes=[
            DagNode(
                id="single_prompt",
                label="Single prompt",
                task_type=TaskType.synthesis,
                operation=Operation.answer,
                question=example.question,
                prompt=PromptSpec(system=system_prompt, user_template=prompt),
                output_schema={
                    "type": "object",
                    "required": ["answer"],
                    "properties": {"answer": {"type": "string"}},
                },
            )
        ],
        final_node="single_prompt",
    )
    trace = NodeTrace(
        node_id="single_prompt",
        label="Single prompt",
        task_type=TaskType.synthesis,
        operation=Operation.answer,
        status=NodeStatus.failed,
        resolved_question=example.question,
        rendered_prompt=prompt,
        raw_response=response_text,
        parsed_output=parsed_output,
        validation=ValidationResult(valid=False, errors=[error]),
        supporting_evidence=EvidenceSelection(
            strategy="all_documents",
            total_available=len(example.context),
            documents=example.context,
        ),
        duration_ms=duration_ms,
        error=error,
    )
    run_trace = RunTrace(
        run_id=str(uuid4()),
        question=example.question,
        plan=plan,
        waves=[SchedulerWave(index=0, node_ids=["single_prompt"])],
        nodes=[trace],
        final_answer=None,
        status=NodeStatus.failed,
        total_duration_ms=duration_ms,
        config=config,
    ).model_dump(mode="json")
    run_trace["mermaid"] = render_mermaid(plan, [trace])
    return BenchmarkRecord(
        id=example.id,
        question=example.question,
        gold_answer=example.answer,
        prediction="",
        metric_scores={metric.name: metric.error_default for metric in ANSWER_METRICS},
        latency_ms=duration_ms,
        llm_call_count=1,
        node_count=1,
        graph_depth=0,
        structural_valid=False,
        structural_failure=True,
        gold_supporting_facts=example.supporting_facts,
        run_trace=run_trace,
        error=error,
    )


def _extract_answer(final_answer: dict[str, Any] | None) -> str:
    if not final_answer:
        return ""
    value = final_answer.get("answer")
    if value is not None:
        return answer_value_to_text(value)
    non_reasoning_values = [
        value
        for key, value in final_answer.items()
        if key not in {"reasoning", "confidence"} and value is not None
    ]
    if len(non_reasoning_values) == 1:
        return answer_value_to_text(non_reasoning_values[0])
    return ""


def _supporting_texts_from_run(run: Any) -> list[str]:
    texts: list[str] = []
    for trace in run.nodes:
        texts.extend(citation.fact for citation in trace.evidence_citations if citation.fact)
    return texts


async def _repair_unsupported_prediction(
    client: DagQaClient,
    example: HotpotExample,
    raw_prediction: str,
    *,
    run_trace: dict[str, Any],
) -> tuple[str | None, int]:
    evidence = EvidenceSelection(
        strategy="all_documents",
        total_available=len(example.context),
        documents=example.context,
    )
    child_outputs = [
        {
            "node_id": node.get("node_id"),
            "question": node.get("resolved_question"),
            "returned_value": node.get("returned_value"),
            "citations": node.get("evidence_citations", []),
        }
        for node in run_trace.get("nodes", [])
        if isinstance(node, dict)
    ]
    prompt = (
        "The DAG final answer appears unsupported by the supplied evidence. "
        "Repair only the concise final answer for scoring.\n\n"
        f"Original question: {example.question}\n"
        f"Unsupported answer: {raw_prediction}\n\n"
        "DAG node outputs and citations:\n"
        f"{json.dumps(child_outputs, indent=2, ensure_ascii=False)}"
        f"{render_evidence_section(evidence)}"
        "\nReturn JSON only with this exact shape:\n"
        '{"answer": "concise answer", "_evidence_citations": ['
        '{"document_id": "exact document ID", "title": "exact title", '
        '"sentence_indices": [0], "fact": "directly supporting fact"}]}\n'
        "Rules: use only supplied evidence; do not invent organisations, people, or films; "
        'do not answer "none" when evidence contains a supported entity; keep yes/no as yes or no.'
    )
    response = await client.llm.complete(
        LLMRequest(
            system="Repair an unsupported benchmark answer using only cited evidence.",
            prompt=prompt,
            temperature=client.config.llm.temperature,
        )
    )
    retry_count = int(response.metadata.get("retry_count", 0))
    try:
        parsed = parse_node_output(response.text)
    except Exception:
        return None, retry_count
    repaired = answer_value_to_text(parsed.get("answer", "")).strip()
    citations = parsed.get("_evidence_citations", [])
    if not repaired or not citations:
        return None, retry_count
    return repaired, retry_count


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


def _evaluate_evidence_citations(
    run: Any,
    gold_supporting_facts: list[GoldSupportingFact],
) -> dict[str, int | float | None]:
    gold_by_key = {
        (_normalize_title(fact.title), fact.sentence_index): fact for fact in gold_supporting_facts
    }
    matched_gold_keys: set[tuple[str, int]] = set()
    citation_count = 0
    correct_count = 0

    for trace in run.nodes:
        evaluations = []
        evidence_titles = {
            document.id: document.title
            for document in (
                trace.supporting_evidence.documents if trace.supporting_evidence is not None else []
            )
        }
        for citation in trace.evidence_citations:
            citation_count += 1
            cited_title = evidence_titles.get(citation.document_id, citation.title)
            matched = [
                gold_by_key[key]
                for sentence_index in citation.sentence_indices
                if (key := (_normalize_title(cited_title), sentence_index)) in gold_by_key
            ]
            matched_gold_keys.update(
                (_normalize_title(fact.title), fact.sentence_index) for fact in matched
            )
            if matched:
                correct_count += 1
            evaluations.append(
                EvidenceCitationEvaluation(
                    citation=citation,
                    matches_gold=bool(matched),
                    matched_gold_facts=matched,
                )
            )
        trace.evidence_citation_evaluations = evaluations

    wrong_count = citation_count - correct_count
    return {
        "evidence_citation_count": citation_count,
        "correct_evidence_citation_count": correct_count,
        "wrong_evidence_citation_count": wrong_count,
        "wrong_supporting_text_rate": wrong_count / citation_count if citation_count else None,
        "gold_supporting_fact_recall": (
            len(matched_gold_keys) / len(gold_by_key) if gold_by_key else None
        ),
    }


def _normalize_title(title: str) -> str:
    return title.strip().casefold()


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
    retry_count_records = [
        record.llm_retry_count for record in records if record.llm_retry_count is not None
    ]
    graph_depth_records = [
        record.graph_depth for record in records if record.graph_depth is not None
    ]
    citation_records = [
        record.evidence_citation_count
        for record in records
        if record.evidence_citation_count is not None
    ]
    correct_citation_records = [
        record.correct_evidence_citation_count
        for record in records
        if record.correct_evidence_citation_count is not None
    ]
    wrong_citation_records = [
        record.wrong_evidence_citation_count
        for record in records
        if record.wrong_evidence_citation_count is not None
    ]
    gold_recall_records = [
        record.gold_supporting_fact_recall
        for record in records
        if record.gold_supporting_fact_recall is not None
    ]
    sorted_latencies = sorted(record.latency_ms for record in records)
    p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2]
    metric_aggregates = {}
    for metric in ANSWER_METRICS:
        samples = [
            MetricSample(
                record.metric_scores.get(metric.name, 0.0),
                record.prediction,
                record.gold_answer,
            )
            for record in records
        ]
        metric_aggregates[metric.name] = metric.aggregate(samples)
    return {
        "example_count": float(len(records)),
        "success_count": float(success_count),
        "failure_count": float(failure_count),
        "error_count": float(error_count),
        **metric_aggregates,
        "avg_latency_ms": sum(record.latency_ms for record in records) / len(records),
        "p50_latency_ms": p50_latency_ms,
        "total_llm_call_count": float(sum(llm_call_records)),
        "total_llm_retry_count": float(sum(retry_count_records)),
        "avg_llm_retry_count": (
            sum(retry_count_records) / len(retry_count_records) if retry_count_records else 0.0
        ),
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
        "total_evidence_citation_count": float(sum(citation_records)),
        "correct_evidence_citation_count": float(sum(correct_citation_records)),
        "wrong_evidence_citation_count": float(sum(wrong_citation_records)),
        "wrong_supporting_text_rate": (
            sum(wrong_citation_records) / sum(citation_records) if sum(citation_records) else 0.0
        ),
        "avg_gold_supporting_fact_recall": (
            sum(gold_recall_records) / len(gold_recall_records) if gold_recall_records else 0.0
        ),
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
