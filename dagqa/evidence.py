from __future__ import annotations

from typing import Any

from dagqa.schemas import DagNode, EvidenceDocument, EvidenceSelection, TaskType

EVIDENCE_TASK_TYPES = {
    TaskType.fact_lookup,
    TaskType.entity_resolution,
    TaskType.date_lookup,
    TaskType.comparison,
    TaskType.synthesis,
}


def select_evidence(
    node: DagNode,
    documents: list[EvidenceDocument] | None,
    outputs: dict[str, dict[str, Any]] | None = None,
) -> EvidenceSelection | None:
    """Select evidence for a node.

    The initial HotpotQA distractor implementation intentionally sends every
    supplied document. Keeping selection behind this function makes later
    ranking or filtering a local change.
    """
    if not documents or node.task_type not in EVIDENCE_TASK_TYPES:
        return None
    if node.task_type in {TaskType.comparison, TaskType.synthesis} and node.depends_on:
        cited_ids = _dependency_cited_document_ids(node, outputs or {})
        if not cited_ids:
            return None
        selected_documents = [document for document in documents if document.id in cited_ids]
        if not selected_documents:
            return None
        return EvidenceSelection(
            strategy="dependency_cited_documents",
            total_available=len(documents),
            documents=selected_documents,
        )
    return EvidenceSelection(
        strategy="all_documents",
        total_available=len(documents),
        documents=list(documents),
    )


def _dependency_cited_document_ids(
    node: DagNode,
    outputs: dict[str, dict[str, Any]],
) -> set[str]:
    cited_ids: set[str] = set()
    for dependency_id in node.depends_on:
        citations = outputs.get(dependency_id, {}).get("_evidence_citations")
        if not isinstance(citations, list):
            continue
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            document_id = citation.get("document_id")
            if isinstance(document_id, str) and document_id:
                cited_ids.add(document_id)
    return cited_ids
