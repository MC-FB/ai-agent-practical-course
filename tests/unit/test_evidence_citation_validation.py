from __future__ import annotations

from dagqa.nodes.output_validation import (
    coerce_output_to_schema,
    validate_evidence_citations,
    validate_factual_output_grounding,
)
from dagqa.nodes.prompts import render_repair_prompt
from dagqa.schemas import EvidenceDocument, EvidenceSelection, TaskType


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


def test_citation_fact_must_be_copied_from_cited_document() -> None:
    result = validate_evidence_citations(
        {
            "answer": "Calumet",
            "_evidence_citations": [
                {
                    "document_id": "context-0",
                    "title": "Final Fantasy",
                    "sentence_indices": [0],
                    "fact": "Calumet is the answer.",
                }
            ],
        },
        _evidence(),
    )

    assert not result.valid
    assert "fact is not copied from the cited document" in result.errors[0]


def test_fact_lookup_answer_must_be_exact_evidence_span() -> None:
    result = validate_factual_output_grounding(
        {"answer": "Calumet"},
        _evidence(),
        TaskType.fact_lookup,
    )

    assert not result.valid
    assert "not an exact span" in result.errors[0]


def test_fact_lookup_answer_allows_source_surface() -> None:
    result = validate_factual_output_grounding(
        {"answer": "Nobuo Uematsu"},
        _evidence(),
        TaskType.fact_lookup,
    )

    assert result.valid


def test_repair_prompt_lists_valid_evidence_ids() -> None:
    prompt = render_repair_prompt(
        '{"answer": "Nobuo Uematsu", "_evidence_citations": [{"document_id": "default"}]}',
        {"type": "object"},
        [
            "_evidence_citations[0].document_id 'default' is not one of the supplied evidence "
            "document IDs."
        ],
        _evidence(),
    )

    assert "context-0: Final Fantasy" in prompt
    assert "Never use placeholder IDs" in prompt


def test_coerce_output_to_schema_turns_list_answer_into_text() -> None:
    output = coerce_output_to_schema(
        {"answer": [1982, 1980, 1983], "confidence": 0.7},
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
        },
    )

    assert output == {"answer": "1982, 1980 and 1983", "confidence": 0.7}
