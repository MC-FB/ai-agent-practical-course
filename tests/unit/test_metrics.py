from __future__ import annotations

from dagqa.eval.metrics import answer_f1, exact_match, normalize_answer


def test_hotpot_style_answer_metrics() -> None:
    assert normalize_answer("The United States.") == "united states"
    assert exact_match("the United States", "United States") == 1.0
    assert answer_f1("Ada Lovelace", "Augusta Ada Lovelace") > 0.0
