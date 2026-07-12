from __future__ import annotations

from collections.abc import Iterable
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


def _selection_from_ids(
    document_ids: Iterable[str],
    documents: list[EvidenceDocument],
) -> EvidenceSelection | None:
    """Resolve document IDs to a selection, dropping unknown/duplicate IDs.

    Order follows first appearance in ``document_ids``. Returns ``None`` when no
    ID resolves to an available document.
    """
    by_id = {document.id: document for document in documents}
    seen: set[str] = set()
    selected: list[EvidenceDocument] = []
    for source_id in document_ids:
        if source_id in by_id and source_id not in seen:
            seen.add(source_id)
            selected.append(by_id[source_id])
    if not selected:
        return None
    return EvidenceSelection(
        strategy="planner_assigned_sources",
        total_available=len(documents),
        documents=selected,
    )


def select_planner_sources(
    node: DagNode,
    documents: list[EvidenceDocument] | None,
) -> EvidenceSelection | None:
    """Resolve a node's planner-assigned source IDs to evidence documents.

    The planner declares, per node, which document IDs supply its supporting
    evidence. Invented IDs (not among ``documents``) are dropped silently.
    Returns ``None`` when the node declared no sources or none of them resolve,
    letting the caller apply its own fallback (e.g. all documents).
    """
    if not documents:
        return None
    return _selection_from_ids(node.sources, documents)


def select_source_union(
    document_ids: Iterable[str],
    documents: list[EvidenceDocument] | None,
) -> EvidenceSelection | None:
    """Build a selection from the union of document IDs gathered across a plan.

    Used for the final synthesis node/turn, which should see everything the
    sub-questions relied on. Returns ``None`` when nothing resolves.
    """
    if not documents:
        return None
    return _selection_from_ids(document_ids, documents)


def all_documents_selection(
    documents: list[EvidenceDocument] | None,
) -> EvidenceSelection | None:
    """Build a selection over every available document.

    Fallback used when the planner assigned no resolvable sources to a node.
    """
    if not documents:
        return None
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
