from __future__ import annotations

import json

from dagqa.eval.datasets import (
    count_benchmark_examples,
    get_benchmark_dataset,
    load_benchmark_examples,
)
from dagqa.eval.musique_loader import count_musique_examples, load_musique_examples


def test_musique_loader_maps_supporting_paragraphs_to_citation_units(tmp_path) -> None:
    path = tmp_path / "musique.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "3hop-example",
                    "question": "Who was born where?",
                    "answer": "Denver",
                    "question_decomposition": [{"q": "a"}, {"q": "b"}, {"q": "c"}],
                    "paragraphs": [
                        {
                            "idx": 2,
                            "title": "South Park",
                            "paragraph_text": "South Park includes the episode The Hobbit.",
                            "is_supporting": True,
                        },
                        {
                            "idx": 7,
                            "title": "Distractor",
                            "paragraph_text": "This paragraph is unrelated.",
                            "is_supporting": False,
                        },
                    ],
                }
            ]
        )
    )

    example = load_musique_examples(path, subset="validation_3hop_plus")[0]

    assert example.id == "3hop-example"
    assert example.context[0].metadata["source"] == "musique_context"
    assert example.context[0].metadata["sentences"] == [
        "South Park includes the episode The Hobbit."
    ]
    assert example.supporting_facts[0].title == "South Park"
    assert example.supporting_facts[0].sentence_index == 0


def test_musique_3hop_subset_filters_decomposition_length(tmp_path) -> None:
    path = tmp_path / "musique.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "2hop",
                    "question": "q",
                    "answer": "a",
                    "question_decomposition": [{}, {}],
                    "paragraphs": [],
                },
                {
                    "id": "3hop",
                    "question": "q",
                    "answer": "a",
                    "question_decomposition": [{}, {}, {}],
                    "paragraphs": [],
                },
            ]
        )
    )

    examples = load_musique_examples(path, subset="validation_3hop_plus")

    assert [example.id for example in examples] == ["3hop"]
    assert count_musique_examples(path, subset="validation_3hop_plus") == 1


def test_dataset_registry_exposes_hotpotqa_and_musique(tmp_path) -> None:
    path = tmp_path / "musique.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "example",
                    "question": "q",
                    "answer": "a",
                    "question_decomposition": [{}, {}, {}],
                    "paragraphs": [],
                }
            ]
        )
    )

    assert get_benchmark_dataset("hotpotqa").default_subset == "validation"
    assert get_benchmark_dataset("musique").default_subset == "validation_3hop_plus"
    assert count_benchmark_examples("musique", "validation_3hop_plus", path) == 1
    assert load_benchmark_examples("musique", "validation_3hop_plus", path)[0].id == "example"
