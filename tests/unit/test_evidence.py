from __future__ import annotations

from dagqa.evidence import select_evidence
from dagqa.nodes.prompts import render_node_prompt
from dagqa.schemas import DagNode, EvidenceDocument, Operation, PromptSpec, TaskType


def _node(task_type: TaskType) -> DagNode:
    return DagNode(
        id="q1",
        label="Find fact",
        task_type=task_type,
        operation=Operation.answer,
        question="Where was Ada born?",
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


def test_select_evidence_skips_reasoning_only_nodes() -> None:
    assert select_evidence(_node(TaskType.synthesis), _documents()) is None


def test_render_prompt_includes_all_evidence_and_distractor_instruction() -> None:
    node = _node(TaskType.fact_lookup)
    selection = select_evidence(node, _documents())

    rendered = render_node_prompt(node, node.question, {}, supporting_evidence=selection)

    assert "Only some of these documents may be relevant" in rendered
    assert "Do not cite documents or candidate facts that you merely read" in rendered
    assert "Do not cite an exhaustive list when only one item is needed" in rendered
    assert "Every returned field value must be directly supported" in rendered
    assert "Do not fill broad lists from partial evidence" in rendered
    assert 'do not return "yes" or "no" unless the question asks yes/no' in rendered
    assert "Document ID: context-0" in rendered
    assert "Title: Ada" in rendered
    assert "Ada was born in London." in rendered
    assert "Title: Distractor" in rendered
    assert '"_evidence_citations"' in rendered
