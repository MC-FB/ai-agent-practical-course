from __future__ import annotations

from dagqa.eval.benchmark import (
    BenchmarkRecord,
    _aggregate,
    _evaluate_evidence_citations,
    _extract_answer,
    _run_example,
    _sample_examples,
)
from dagqa.eval.hotpot_loader import HotpotExample
from dagqa.eval.metrics import answer_f1, exact_match, normalize_answer
from dagqa.schemas import (
    DagNode,
    DagPlan,
    EvidenceCitation,
    EvidenceDocument,
    EvidenceSelection,
    GoldSupportingFact,
    NodeStatus,
    NodeTrace,
    Operation,
    PromptSpec,
    RunTrace,
    TaskType,
)


def test_hotpot_style_answer_metrics() -> None:
    assert normalize_answer("The United States.") == "united states"
    assert exact_match("the United States", "United States") == 1.0
    assert answer_f1("Ada Lovelace", "Augusta Ada Lovelace") > 0.0


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


def test_extract_answer_accepts_single_named_answer_field() -> None:
    assert _extract_answer({"earlier_person": "Ada Lovelace"}) == "Ada Lovelace"


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
