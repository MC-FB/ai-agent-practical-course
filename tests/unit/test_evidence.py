from __future__ import annotations

from dagqa.evidence import (
    all_documents_selection,
    select_evidence,
    select_planner_sources,
    select_source_union,
)
from dagqa.nodes.prompts import render_node_prompt
from dagqa.schemas import DagNode, EvidenceDocument, Operation, PromptSpec, TaskType


def _node(
    task_type: TaskType,
    depends_on: list[str] | None = None,
    sources: list[str] | None = None,
) -> DagNode:
    return DagNode(
        id="q1",
        label="Find fact",
        task_type=task_type,
        operation=Operation.answer,
        question="Where was Ada born?",
        depends_on=depends_on or [],
        sources=sources or [],
        prompt=PromptSpec(
            system="Return JSON only.",
            user_template="Question: {resolved_question}",
        ),
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )


def _documents() -> list[EvidenceDocument]:
    return [
        EvidenceDocument(id="context-0", title="Ada", text="Ada was born in London."),
        EvidenceDocument(id="context-1", title="Distractor", text="This is not relevant."),
    ]


def test_select_evidence_returns_all_documents_for_fact_nodes() -> None:
    expected_document_count = 2
    selection = select_evidence(_node(TaskType.fact_lookup), _documents())

    assert selection is not None
    assert selection.strategy == "all_documents"
    assert selection.total_available == expected_document_count
    assert [document.id for document in selection.documents] == ["context-0", "context-1"]


def test_select_evidence_returns_documents_for_synthesis_nodes() -> None:
    selection = select_evidence(_node(TaskType.synthesis), _documents())

    assert selection is not None
    assert [document.id for document in selection.documents] == ["context-0", "context-1"]


def test_select_evidence_limits_synthesis_nodes_to_dependency_citations() -> None:
    selection = select_evidence(
        _node(TaskType.synthesis, depends_on=["q0"]),
        _documents(),
        {"q0": {"_evidence_citations": [{"document_id": "context-0"}]}},
    )

    assert selection is not None
    assert selection.strategy == "dependency_cited_documents"
    assert [document.id for document in selection.documents] == ["context-0"]


def test_select_evidence_skips_dependent_synthesis_without_citations() -> None:
    selection = select_evidence(
        _node(TaskType.synthesis, depends_on=["q0"]),
        _documents(),
        {"q0": {"answer": "Ada"}},
    )

    assert selection is None


def test_select_planner_sources_returns_only_assigned_documents() -> None:
    expected_document_count = 2
    selection = select_planner_sources(
        _node(TaskType.fact_lookup, sources=["context-1"]), _documents()
    )

    assert selection is not None
    assert selection.strategy == "planner_assigned_sources"
    assert selection.total_available == expected_document_count
    assert [document.id for document in selection.documents] == ["context-1"]


def test_select_planner_sources_drops_unknown_ids_and_dedups() -> None:
    selection = select_planner_sources(
        _node(TaskType.fact_lookup, sources=["context-0", "context-0", "context-9"]),
        _documents(),
    )

    assert selection is not None
    assert [document.id for document in selection.documents] == ["context-0"]


def test_select_planner_sources_returns_none_when_nothing_resolves() -> None:
    docs = _documents()
    assert select_planner_sources(_node(TaskType.fact_lookup, sources=["context-9"]), docs) is None
    assert select_planner_sources(_node(TaskType.fact_lookup), docs) is None
    assert select_planner_sources(_node(TaskType.fact_lookup, sources=["context-0"]), []) is None


def test_select_source_union_preserves_first_appearance_order() -> None:
    selection = select_source_union(["context-1", "context-0", "context-1"], _documents())

    assert selection is not None
    assert selection.strategy == "planner_assigned_sources"
    assert [document.id for document in selection.documents] == ["context-1", "context-0"]


def test_all_documents_selection_includes_every_document() -> None:
    expected_document_count = 2
    selection = all_documents_selection(_documents())

    assert selection is not None
    assert selection.strategy == "all_documents"
    assert selection.total_available == expected_document_count
    assert [document.id for document in selection.documents] == ["context-0", "context-1"]
    assert all_documents_selection([]) is None
    assert all_documents_selection(None) is None


def test_render_prompt_includes_all_evidence_and_distractor_instruction() -> None:
    node = _node(TaskType.fact_lookup)
    selection = select_evidence(node, _documents())

    rendered = render_node_prompt(node, node.question, {}, supporting_evidence=selection)

    assert "Only some of these documents may be relevant" in rendered
    assert "Document ID: context-0" in rendered
    assert "Title: Ada" in rendered
    assert "Ada was born in London." in rendered
    assert "Title: Distractor" in rendered
    # Non-final intermediate nodes do not get _evidence_citations in schema
    assert '"_evidence_citations"' not in rendered
