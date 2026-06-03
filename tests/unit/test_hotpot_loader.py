from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from dagqa.eval.hotpot_loader import _load_from_huggingface, load_hotpot_examples


def test_load_hotpot_examples_preserves_context_as_evidence_documents(tmp_path) -> None:
    path = tmp_path / "hotpot.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "example-1",
                    "question": "Where was Ada born?",
                    "answer": "London",
                    "context": [
                        [
                            "Ada Lovelace",
                            [
                                "Ada Lovelace was an English mathematician. ",
                                "She was born in London.",
                            ],
                        ],
                        ["Distractor", ["Unrelated text."]],
                    ],
                    "supporting_facts": [["Ada Lovelace", 1]],
                }
            ]
        )
    )

    example = load_hotpot_examples(path)[0]

    assert [document.title for document in example.context] == ["Ada Lovelace", "Distractor"]
    assert example.context[0].text == (
        "Ada Lovelace was an English mathematician. She was born in London."
    )
    assert example.context[0].metadata["source"] == "hotpotqa_context"
    assert example.context[0].metadata["sentences"] == [
        "Ada Lovelace was an English mathematician. ",
        "She was born in London.",
    ]
    assert example.supporting_facts[0].title == "Ada Lovelace"
    assert example.supporting_facts[0].sentence_index == 1


def test_huggingface_loader_passes_hf_token_from_environment(monkeypatch, tmp_path) -> None:
    calls = []

    def load_dataset(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append((args, kwargs))
        return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))

    assert _load_from_huggingface(limit=1) == []
    assert calls[0][1]["token"] == "test-token"
