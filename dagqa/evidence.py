from __future__ import annotations

from dagqa.schemas import DagNode, EvidenceDocument, EvidenceSelection, TaskType

EVIDENCE_TASK_TYPES = {
    TaskType.fact_lookup,
    TaskType.entity_resolution,
    TaskType.date_lookup,
}


def select_evidence(
    node: DagNode,
    documents: list[EvidenceDocument] | None,
) -> EvidenceSelection | None:
    """Select evidence for a node.

    The initial HotpotQA distractor implementation intentionally sends every
    supplied document. Keeping selection behind this function makes later
    ranking or filtering a local change.
    """
    if not documents or node.task_type not in EVIDENCE_TASK_TYPES:
        return None
    return EvidenceSelection(
        strategy="all_documents",
        total_available=len(documents),
        documents=list(documents),
    )
