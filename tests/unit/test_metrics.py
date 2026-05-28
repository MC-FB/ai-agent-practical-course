from __future__ import annotations

from dagqa.eval.benchmark import _extract_answer, _sample_examples
from dagqa.eval.hotpot_loader import HotpotExample
from dagqa.eval.metrics import answer_f1, exact_match, normalize_answer


def test_hotpot_style_answer_metrics() -> None:
    assert normalize_answer("The United States.") == "united states"
    assert exact_match("the United States", "United States") == 1.0
    assert answer_f1("Ada Lovelace", "Augusta Ada Lovelace") > 0.0


def test_benchmark_sampling_is_seeded() -> None:
    examples = [
        HotpotExample(id=str(index), question=f"q{index}", answer=f"a{index}")
        for index in range(10)
    ]

    first = _sample_examples(examples, 4, 123)
    second = _sample_examples(examples, 4, 123)
    different = _sample_examples(examples, 4, 456)

    assert [example.id for example in first] == [example.id for example in second]
    assert [example.id for example in first] != [example.id for example in different]


def test_extract_answer_accepts_single_named_answer_field() -> None:
    assert _extract_answer({"earlier_person": "Ada Lovelace"}) == "Ada Lovelace"
