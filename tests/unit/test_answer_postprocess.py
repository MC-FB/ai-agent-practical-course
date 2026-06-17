from __future__ import annotations

from dagqa.eval.answer_postprocess import (
    answer_value_to_text,
    canonicalize_prediction,
    is_placeholder_or_unsupported,
)
from dagqa.schemas import EvidenceDocument


def test_answer_value_to_text_compresses_two_person_list_with_shared_surname() -> None:
    assert answer_value_to_text(["Charles Guard", "Thomas Guard"]) == "Charles and Thomas Guard"


def test_canonicalize_prediction_parses_literal_list() -> None:
    assert (
        canonicalize_prediction(
            "['Charles Guard', 'Thomas Guard']",
            question="What are the names of the co-directors?",
        )
        == "Charles and Thomas Guard"
    )


def test_canonicalize_prediction_extracts_between_target_for_town_question() -> None:
    assert (
        canonicalize_prediction(
            "Danglemah is between Tamworth and Walcha.",
            question=(
                "Danglemah, New South Wales is between which town and the town on the "
                "south-eastern edge of the Northern Tablelands, New South Wales Australia?"
            ),
        )
        == "Tamworth"
    )


def test_canonicalize_prediction_restores_quantity_unit_from_evidence() -> None:
    evidence = [
        EvidenceDocument(
            id="context-0",
            title="Boudougate",
            text="The transaction involved 2.3 million pesos in Argentina currency.",
        )
    ]

    assert (
        canonicalize_prediction(
            "2300000",
            question="How much Argentina currency was involved in Boudougate?",
            evidence_documents=evidence,
        )
        == "2.3 million pesos"
    )


def test_canonicalize_prediction_restores_comma_number_surface_from_evidence() -> None:
    evidence = [
        EvidenceDocument(
            id="context-0",
            title="Concord, California",
            text="At the 2010 census, the city had a population of 122,067.",
        )
    ]

    assert (
        canonicalize_prediction(
            "122067",
            question="What was the population at the 2010 census?",
            evidence_documents=evidence,
        )
        == "122,067"
    )


def test_canonicalize_prediction_restores_first_introduced_phrase_from_citation() -> None:
    assert (
        canonicalize_prediction(
            "2014",
            question="When was the character first introduced on Once Upon a Time?",
            supporting_texts=[
                "Zelena was first introduced in the second half of the third season "
                "of Once Upon a Time."
            ],
        )
        == "the second half of the third season"
    )


def test_canonicalize_prediction_restores_subject_painting_phrase_from_citation() -> None:
    assert (
        canonicalize_prediction(
            "Portrait of the Dancer Anita Berber",
            question="Of which painting was this German dancer the subject?",
            supporting_texts=["Anita Berber was the subject of an Otto Dix painting."],
        )
        == "an Otto Dix painting"
    )


def test_canonicalize_prediction_restores_language_descriptor_from_citation() -> None:
    assert (
        canonicalize_prediction(
            "Italian",
            question="The Best Offer was written and directed in what language?",
            supporting_texts=[
                "The Best Offer is a 2013 Italian English-language romantic mystery film "
                "written and directed by Giuseppe Tornatore."
            ],
        )
        == "English-language"
    )


def test_canonicalize_prediction_normalizes_boolean_yes_no() -> None:
    assert canonicalize_prediction("True", question="Are both films documentaries?") == "yes"


def test_canonicalize_prediction_restores_initialed_name_from_evidence() -> None:
    assert (
        canonicalize_prediction(
            "L. M. Montgomery",
            question="Which children's novelist wrote about Anne Shirley?",
            supporting_texts=[
                "Lucy Maud Montgomery was a Canadian author best known for Anne of Green Gables."
            ],
        )
        == "Lucy Maud Montgomery"
    )


def test_canonicalize_prediction_formats_structured_year_conference_pair() -> None:
    assert (
        canonicalize_prediction(
            "year: 2009, conference: Big 12 Conference",
            question="Which year and which conference?",
        )
        == "2009 Big 12 Conference"
    )


def test_canonicalize_prediction_restores_three_other_cast_members() -> None:
    assert (
        canonicalize_prediction(
            "Yu Shaoqun",
            question=(
                "The Chinese actress also known as Crystal Liu stars in Night Peacock "
                "with which three other actresses?"
            ),
            supporting_texts=["It stars Liu Yifei, Liu Ye, Yu Shaoqun and Leon Lai."],
        )
        == "Liu Ye, Yu Shaoqun and Leon Lai"
    )


def test_canonicalize_prediction_normalizes_yes_no() -> None:
    assert (
        canonicalize_prediction("Yes.", question="Are Ian Brown and Dee Snider both actors?")
        == "yes"
    )


def test_canonicalize_prediction_strips_disambiguating_parenthetical() -> None:
    assert (
        canonicalize_prediction(
            "Liberty (Adventist magazine)",
            question="Which has a circulation over 200,000?",
        )
        == "Liberty"
    )


def test_unsupported_placeholder_triggers_for_uncited_entity() -> None:
    evidence = [
        EvidenceDocument(
            id="context-0", title="Mediastan", text="Julian Assange founded WikiLeaks."
        )
    ]

    assert is_placeholder_or_unsupported(
        "TechCorp",
        question="One of the directors of Mediastan founded what organisation?",
        evidence_documents=evidence,
    )


def test_supported_entity_does_not_trigger_repair() -> None:
    evidence = [
        EvidenceDocument(
            id="context-0", title="Mediastan", text="Julian Assange founded WikiLeaks."
        )
    ]

    assert not is_placeholder_or_unsupported(
        "WikiLeaks",
        question="One of the directors of Mediastan founded what organisation?",
        evidence_documents=evidence,
    )


def test_none_triggers_repair_for_factoid_question() -> None:
    assert is_placeholder_or_unsupported(
        "none",
        question="Which actor/director directed the film?",
        evidence_documents=[],
    )


def test_yes_no_triggers_repair_for_non_yes_no_question() -> None:
    assert is_placeholder_or_unsupported(
        "no",
        question="Which children's novelist wrote about Anne Shirley?",
        evidence_documents=[],
    )
