from __future__ import annotations

import asyncio
from types import SimpleNamespace

import dagqa.eval.benchmark as benchmark_module
from dagqa.config import AppConfig, BenchmarkConfig
from dagqa.eval.benchmark import (
    BenchmarkRecord,
    _aggregate,
    _evaluate_evidence_citations,
    _extract_answer,
    _recover_evidence_pattern_answer,
    _run_example,
    _run_examples,
    _sample_examples,
)
from dagqa.eval.hotpot_loader import HotpotExample
from dagqa.eval.metrics import (
    ANSWER_METRICS,
    MetricSample,
    _cosine_mean,
    answer_f1,
    exact_match,
    normalize_answer,
)
from dagqa.schemas import (
    DagNode,
    DagPlan,
    EvidenceCitation,
    EvidenceDocument,
    EvidenceSelection,
    GoldSupportingFact,
    LLMResponse,
    NodeStatus,
    NodeTrace,
    Operation,
    PromptSpec,
    RunTrace,
    TaskType,
)

EXPECTED_EMPTY_RETRY_CALLS = 2


def test_hotpot_style_answer_metrics() -> None:
    assert normalize_answer("The United States.") == "united states"
    assert exact_match("the United States", "United States", "") == 1.0
    assert answer_f1("Ada Lovelace", "Augusta Ada Lovelace", "") > 0.0


def test_benchmark_sampling_is_seeded() -> None:
    examples = [
        HotpotExample(id=str(index), question=f"q{index}", answer=f"a{index}")
        for index in range(10)
    ]

    first = _sample_examples(examples, 4, 123)
    second = _sample_examples(examples, 4, 123)
    different = _sample_examples(examples, 4, 456)

    assert [example.id for example in first] == [example.id for example in second]
    assert [example.id for example in first] != [example.id for example in different]


async def test_run_examples_bounds_concurrency_and_preserves_order(monkeypatch) -> None:
    parallel_examples = 2
    active = 0
    max_active = 0
    completion_order = []
    examples = [
        HotpotExample(id=str(index), question=f"q{index}", answer=f"a{index}") for index in range(4)
    ]

    async def run_example(client, example, system):  # noqa: ANN001, ARG001
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01 if example.id == "0" else 0)
        active -= 1
        return BenchmarkRecord(
            id=example.id,
            question=example.question,
            gold_answer=example.answer,
            prediction=example.answer,
            exact_match=1,
            f1=1,
            latency_ms=1,
        )

    monkeypatch.setattr(benchmark_module, "_run_example", run_example)

    records = await _run_examples(
        object(),  # type: ignore[arg-type]
        examples,
        "direct_llm",
        max_parallel_examples=parallel_examples,
        on_complete=lambda record: completion_order.append(record.id),
    )

    assert max_active == parallel_examples
    assert completion_order != [example.id for example in examples]
    assert [record.id for record in records] == [example.id for example in examples]


async def test_run_examples_retries_empty_predictions(monkeypatch) -> None:
    calls = 0
    completed: list[BenchmarkRecord] = []
    example = HotpotExample(id="example-1", question="q1", answer="answer")

    async def run_example(client, current_example, system):  # noqa: ANN001, ARG001
        nonlocal calls
        calls += 1
        prediction = "" if calls == 1 else "answer"
        return BenchmarkRecord(
            id=current_example.id,
            question=current_example.question,
            gold_answer=current_example.answer,
            prediction=prediction,
            exact_match=1.0 if prediction else 0.0,
            f1=1.0 if prediction else 0.0,
            latency_ms=1,
            llm_retry_count=0,
            error="empty" if not prediction else None,
        )

    class Client:
        config = AppConfig(benchmark=BenchmarkConfig(empty_prediction_retries=1))

    monkeypatch.setattr(benchmark_module, "_run_example", run_example)

    records = await _run_examples(
        Client(),  # type: ignore[arg-type]
        [example],
        "direct_llm",
        max_parallel_examples=1,
        on_complete=completed.append,
    )

    assert calls == EXPECTED_EMPTY_RETRY_CALLS
    assert records[0].prediction == "answer"
    assert completed == records


def test_extract_answer_accepts_single_named_answer_field() -> None:
    assert _extract_answer({"earlier_person": "Ada Lovelace"}) == "Ada Lovelace"


def test_extract_answer_prefers_concise_source_span_for_bare_number() -> None:
    assert _extract_answer({"answer": 8.11, "answer_source_span": "8.11 million"}) == "8.11 million"


def test_extract_answer_keeps_answer_when_source_span_is_explanatory() -> None:
    assert (
        _extract_answer({"answer": "Ada Lovelace", "answer_source_span": "Ada Lovelace: 1815"})
        == "Ada Lovelace"
    )


def test_recover_evidence_pattern_answer_finds_capital_duration_from_trace_location() -> None:
    run = SimpleNamespace(
        nodes=[
            SimpleNamespace(returned_value={"city": "Yangzhou"}),
            SimpleNamespace(returned_value={"answer": "0"}),
        ]
    )
    documents = [
        EvidenceDocument(
            id="context-0",
            title="Nanjing",
            text="Nanjing had been the capital city of Yangzhou for about 400 years.",
        )
    ]

    assert (
        _recover_evidence_pattern_answer(
            "How long had X been the capitol city of Y's headquarters location?",
            "0",
            run,
            documents,
        )
        == "about 400 years"
    )


def test_recover_evidence_pattern_answer_finds_latest_table_win_against_opponent() -> None:
    run = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                label="Identify cup winner",
                returned_value={"team": "Aston Villa", "answer": "Aston Villa"},
            )
        ]
    )
    documents = [
        EvidenceDocument(
            id="context-0",
            title="Second City derby",
            text=(
                "Date Venue Home team Score Competition "
                "23 March 1901 Muntz Street Small Heath 0 -- 0 FA Cup "
                "1 December 2010 St Andrew's Birmingham City 2 -- 1 League Cup "
                "22 September 2015 Villa Park Aston Villa 1 -- 0 League Cup"
            ),
        )
    ]

    assert (
        _recover_evidence_pattern_answer(
            "When was the last time George Hollis's team beat the 1894-95 FA Cup winner?",
            "never",
            run,
            documents,
        )
        == "1 December 2010"
    )


async def test_benchmark_passes_hotpot_context_to_dag_agent() -> None:
    documents = [
        EvidenceDocument(id="context-0", title="Ada Lovelace", text="Ada was born in London.")
    ]

    class RecordingClient:
        def __init__(self) -> None:
            self.evidence_documents: list[EvidenceDocument] | None = None

        async def ask(self, question: str, evidence_documents=None):  # noqa: ANN001
            self.evidence_documents = evidence_documents
            raise RuntimeError("stop after recording")

    client = RecordingClient()
    example = HotpotExample(
        id="example-1",
        question="Where was Ada born?",
        answer="London",
        context=documents,
    )

    record = await _run_example(client, example, "dag_agent")  # type: ignore[arg-type]

    assert client.evidence_documents == documents
    assert record.error == "stop after recording"
    assert record.prediction == ""
    assert record.structural_failure
    assert record.run_trace is None


async def test_direct_baseline_sends_all_sources_and_stores_cited_trace() -> None:
    source_count = 2
    documents = [
        EvidenceDocument(
            id="context-0",
            title="Ada Lovelace",
            text="Ada was born in London.",
            metadata={"sentences": ["Ada was born in London."]},
        ),
        EvidenceDocument(
            id="context-1",
            title="Distractor",
            text="This is unrelated.",
            metadata={"sentences": ["This is unrelated."]},
        ),
    ]

    class RecordingLLM:
        prompt = ""

        async def complete(self, request):  # noqa: ANN001, ANN202
            self.prompt = request.prompt
            return LLMResponse(
                text=(
                    '{"answer":"London","_evidence_citations":[{"document_id":"context-0",'
                    '"title":"Ada Lovelace","sentence_indices":[0],'
                    '"fact":"Ada was born in London."}]}'
                ),
                model="test",
            )

    class RecordingClient:
        config = AppConfig()
        llm = RecordingLLM()

    client = RecordingClient()
    record = await _run_example(
        client,  # type: ignore[arg-type]
        HotpotExample(
            id="example-1",
            question="Where was Ada born?",
            answer="London",
            context=documents,
            supporting_facts=[GoldSupportingFact(title="Ada Lovelace", sentence_index=0)],
        ),
        "direct_llm",
    )

    assert "Ada was born in London." in client.llm.prompt
    assert "This is unrelated." in client.llm.prompt
    assert record.prediction == "London"
    assert record.evidence_citation_count == 1
    assert record.gold_supporting_fact_recall == 1
    assert record.node_count == 1
    assert record.run_trace is not None
    assert record.run_trace["nodes"][0]["supporting_evidence"]["total_available"] == source_count


async def test_direct_baseline_failure_stores_prompt_and_raw_response_trace() -> None:
    class FailingLLM:
        async def complete(self, request):  # noqa: ANN001, ANN202
            return LLMResponse(text='{"answer": ""}', model="test")

    class RecordingClient:
        config = AppConfig()
        llm = FailingLLM()

    record = await _run_example(
        RecordingClient(),  # type: ignore[arg-type]
        HotpotExample(
            id="example-1",
            question="Where was Ada born?",
            answer="London",
            context=[
                EvidenceDocument(
                    id="context-0",
                    title="Ada Lovelace",
                    text="Ada was born in London.",
                )
            ],
        ),
        "direct_llm",
    )

    assert record.prediction == ""
    assert record.error == "Single-prompt response is missing a non-empty answer."
    assert record.run_trace is not None
    assert record.run_trace["status"] == "failed"
    system_prompt = record.run_trace["plan"]["nodes"][0]["prompt"]["system"]
    assert system_prompt.startswith("Answer the question")
    trace = record.run_trace["nodes"][0]
    assert trace["status"] == "failed"
    assert "Question: Where was Ada born?" in trace["rendered_prompt"]
    assert trace["raw_response"] == '{"answer": ""}'
    assert trace["validation"]["errors"] == [
        "Single-prompt response is missing a non-empty answer."
    ]


def test_evidence_citation_metrics_compare_against_gold_supporting_facts() -> None:
    expected_citation_count = 2
    expected_wrong_rate = 0.5
    document = EvidenceDocument(
        id="context-0",
        title="Ada Lovelace",
        text="Ada was born in London.",
    )
    node = DagNode(
        id="q1",
        label="Find birthplace",
        task_type=TaskType.fact_lookup,
        operation=Operation.answer,
        question="Where was Ada born?",
        prompt=PromptSpec(system="system", user_template="template"),
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    trace = NodeTrace(
        node_id="q1",
        label="Find birthplace",
        task_type=TaskType.fact_lookup,
        operation=Operation.answer,
        status=NodeStatus.succeeded,
        supporting_evidence=EvidenceSelection(
            strategy="all_documents",
            total_available=1,
            documents=[document],
        ),
        evidence_citations=[
            EvidenceCitation(
                document_id="context-0",
                title="Ada Lovelace",
                sentence_indices=[0],
                fact="Ada was born in London.",
            ),
            EvidenceCitation(
                document_id="context-0",
                title="Ada Lovelace",
                sentence_indices=[1],
                fact="A non-gold fact.",
            ),
        ],
    )
    run = RunTrace(
        run_id="run-1",
        question="Where was Ada born?",
        plan=DagPlan(question="Where was Ada born?", nodes=[node], final_node="q1"),
        waves=[],
        nodes=[trace],
        status=NodeStatus.succeeded,
        total_duration_ms=1,
    )

    metrics = _evaluate_evidence_citations(
        run,
        [GoldSupportingFact(title="Ada Lovelace", sentence_index=0)],
    )

    assert metrics["evidence_citation_count"] == expected_citation_count
    assert metrics["correct_evidence_citation_count"] == 1
    assert metrics["wrong_evidence_citation_count"] == 1
    assert metrics["wrong_supporting_text_rate"] == expected_wrong_rate
    assert metrics["gold_supporting_fact_recall"] == 1.0
    assert run.nodes[0].evidence_citation_evaluations[0].matches_gold
    assert not run.nodes[0].evidence_citation_evaluations[1].matches_gold


def test_aggregate_reports_wrong_supporting_text_rate() -> None:
    expected_wrong_rate = 0.25
    expected_recall = 0.75
    records = [
        BenchmarkRecord(
            id="1",
            question="q1",
            gold_answer="a1",
            prediction="a1",
            exact_match=1,
            f1=1,
            latency_ms=1,
            evidence_citation_count=2,
            correct_evidence_citation_count=1,
            wrong_evidence_citation_count=1,
            gold_supporting_fact_recall=0.5,
        ),
        BenchmarkRecord(
            id="2",
            question="q2",
            gold_answer="a2",
            prediction="a2",
            exact_match=1,
            f1=1,
            latency_ms=1,
            evidence_citation_count=2,
            correct_evidence_citation_count=2,
            wrong_evidence_citation_count=0,
            gold_supporting_fact_recall=1,
        ),
    ]

    metrics = _aggregate(records)

    assert metrics["wrong_supporting_text_rate"] == expected_wrong_rate
    assert metrics["avg_gold_supporting_fact_recall"] == expected_recall


def test_legacy_metric_kwargs_fold_into_metric_scores() -> None:
    """Legacy ``exact_match=``/``f1=`` kwargs (cosine omitted) populate the dict.

    Also verifies the computed-field attribute access that ``cli.py`` relies on,
    and that ``model_dump`` re-emits the legacy top-level keys for the frontend.
    """
    record = BenchmarkRecord(
        id="1", question="q", gold_answer="a", prediction="a", exact_match=1.0, f1=1.0, latency_ms=1
    )

    assert record.metric_scores == {"exact_match": 1.0, "f1": 1.0}
    assert record.exact_match == 1.0
    assert record.f1 == 1.0
    assert record.cosine_sim == 0.0  # missing metric falls back to 0.0

    dumped = record.model_dump(mode="json")
    assert dumped["exact_match"] == 1.0
    assert dumped["cosine_sim"] == 0.0
    assert dumped["metric_scores"]["f1"] == 1.0


def test_old_json_record_migrates_on_validate() -> None:
    """Old-format records (named keys, no ``metric_scores``) load into the dict."""
    old = {
        "id": "2",
        "question": "q",
        "gold_answer": "a",
        "prediction": "a",
        "exact_match": 1.0,
        "f1": 0.5,
        "cosine_sim": 0.9,
        "latency_ms": 1,
    }

    record = BenchmarkRecord.model_validate(old)

    assert record.metric_scores == {"exact_match": 1.0, "f1": 0.5, "cosine_sim": 0.9}


def test_aggregate_emits_one_key_per_registered_metric() -> None:
    """Adding a metric to the registry makes it appear in the aggregate output."""
    records = [
        BenchmarkRecord(
            id=str(index),
            question="q",
            gold_answer="Paris",
            prediction="Paris",
            metric_scores={"exact_match": 1.0, "f1": 1.0, "cosine_sim": 1.0},
            latency_ms=1,
        )
        for index in range(2)
    ]

    metrics = _aggregate(records)

    for metric in ANSWER_METRICS:
        assert metric.name in metrics


def test_aggregate_includes_error_record_cosine_as_zero() -> None:
    """Error records store cosine 0.0 and are averaged in as 0.0 (not the old -1.5)."""
    expected_mean = 0.5  # one perfect record (1.0) + one errored record (0.0)
    ok = BenchmarkRecord(
        id="ok",
        question="q",
        gold_answer="Paris",
        prediction="Paris",
        metric_scores={"exact_match": 1.0, "f1": 1.0, "cosine_sim": 1.0},
        latency_ms=1,
    )
    errored = BenchmarkRecord(
        id="err",
        question="q",
        gold_answer="London",
        prediction="",
        metric_scores={metric.name: metric.error_default for metric in ANSWER_METRICS},
        latency_ms=1,
        error="boom",
    )

    metrics = _aggregate([ok, errored])

    assert errored.metric_scores["cosine_sim"] == 0.0
    assert metrics["cosine_sim"] == expected_mean
    assert metrics["exact_match"] == expected_mean


def test_cosine_mean_skips_empty_pairs() -> None:
    samples = [MetricSample(1.0, "a", "a"), MetricSample(0.5, "", "")]

    assert _cosine_mean(samples) == 1.0
