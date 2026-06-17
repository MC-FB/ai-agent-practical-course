from __future__ import annotations

from dagqa.nodes.output_validation import validate_evidence_citations
from dagqa.schemas import EvidenceDocument, EvidenceSelection


def _evidence() -> EvidenceSelection:
    return EvidenceSelection(
        strategy="all_documents",
        total_available=1,
        documents=[
            EvidenceDocument(
                id="context-0",
                title="Final Fantasy",
                text="Final Fantasy contains Chocobos and Moogles. Nobuo Uematsu composed music.",
                metadata={
                    "sentences": [
                        "Final Fantasy contains Chocobos and Moogles.",
                        "Nobuo Uematsu composed music.",
                    ]
                },
            )
        ],
    )


def test_valid_evidence_citation_passes() -> None:
    result = validate_evidence_citations(
        {
            "answer": "Nobuo Uematsu",
            "_evidence_citations": [
                {
                    "document_id": "context-0",
                    "title": "Final Fantasy",
                    "sentence_indices": [1],
                    "fact": "Nobuo Uematsu composed music.",
                }
            ],
        },
        _evidence(),
    )

    assert result.valid
    assert result.errors == []


def test_nonexistent_document_id_fails() -> None:
    result = validate_evidence_citations(
        {
            "answer": "Nobuo Uematsu",
            "_evidence_citations": [
                {
                    "document_id": "doc1",
                    "title": "Final Fantasy Composer",
                    "sentence_indices": [0],
                    "fact": "Nobuo Uematsu composed music.",
                }
            ],
        },
        _evidence(),
    )

    assert not result.valid
    assert "not one of the supplied evidence document IDs" in result.errors[0]


def test_mismatched_document_title_fails() -> None:
    result = validate_evidence_citations(
        {
            "answer": "Nobuo Uematsu",
            "_evidence_citations": [
                {
                    "document_id": "context-0",
                    "title": "Final Fantasy Composer",
                    "sentence_indices": [1],
                    "fact": "Nobuo Uematsu composed music.",
                }
            ],
        },
        _evidence(),
    )

    assert not result.valid
    assert "title must match the supplied title" in result.errors[0]


def test_out_of_range_sentence_index_fails() -> None:
    result = validate_evidence_citations(
        {
            "answer": "Nobuo Uematsu",
            "_evidence_citations": [
                {
                    "document_id": "context-0",
                    "title": "Final Fantasy",
                    "sentence_indices": [2],
                    "fact": "Nobuo Uematsu composed music.",
                }
            ],
        },
        _evidence(),
    )

    assert not result.valid
    assert "only has sentence indices 0 through 1" in result.errors[0]
