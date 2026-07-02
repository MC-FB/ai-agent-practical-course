from __future__ import annotations

from dagqa.eval.metric_eval import (
    AnnotationRecord,
    _kendall_tau,
    _spearman,
    evaluate_metrics,
    fill_scores,
    score_records,
)
from dagqa.eval.metrics import Metric


def _band_record(pred: str, gold: str, low: float, high: float) -> AnnotationRecord:
    return AnnotationRecord(
        run_id="run",
        record_id=pred,
        question="q",
        gold_answer=gold,
        prediction=pred,
        human_low=low,
        human_high=high,
    )


def test_in_range_rate_and_center_distance() -> None:
    expected_rate = 0.5
    expected_center_distance = 0.5
    always_one = Metric("always_one", lambda _p, _g, _q: 1.0)
    records = [
        _band_record("a", "a", 0.8, 1.0),  # 1.0 is in-band
        _band_record("b", "b", 0.0, 0.2),  # 1.0 is out-of-band
    ]

    results = {result.name: result for result in evaluate_metrics(records, metrics=[always_one])}

    assert results["always_one"].in_range_rate == expected_rate
    # |1 - 0.9| = 0.1 and |1 - 0.1| = 0.9 -> mean 0.5
    assert results["always_one"].mean_center_distance == expected_center_distance
    assert results["always_one"].n == len(records)


def test_skips_records_without_a_band() -> None:
    metric = Metric("m", lambda _p, _g, _q: 1.0)
    records = [
        AnnotationRecord(run_id="r", record_id="1", prediction="x", gold_answer="x"),  # no band
        _band_record("a", "a", 0.9, 1.0),
    ]

    results = evaluate_metrics(records, metrics=[metric])

    assert results[0].n == 1  # only the banded record is scored


def test_ranking_prefers_higher_in_range_rate() -> None:
    good = Metric("good", lambda _p, _g, _q: 0.9)  # lands inside [0.8, 1.0]
    bad = Metric("bad", lambda _p, _g, _q: 0.1)  # misses [0.8, 1.0]
    records = [_band_record("a", "a", 0.8, 1.0), _band_record("b", "b", 0.8, 1.0)]

    results = evaluate_metrics(records, metrics=[bad, good])

    assert results[0].name == "good"
    assert results[0].in_range_rate == 1.0
    assert results[-1].name == "bad"


def test_band_bounds_are_order_normalised() -> None:
    metric = Metric("m", lambda _p, _g, _q: 0.5)
    # Human entered high, low in the "wrong" order — should still count as in-band.
    record = AnnotationRecord(
        run_id="r", record_id="1", prediction="p", gold_answer="g", human_low=0.7, human_high=0.3
    )

    results = evaluate_metrics([record], metrics=[metric])

    assert results[0].in_range_rate == 1.0


def test_score_records_computes_each_metric_fresh() -> None:
    length = Metric("length", lambda p, _g, _q: float(len(p)))
    exact = Metric("exact", lambda p, g, _q: 1.0 if p == g else 0.0)
    records = [
        AnnotationRecord(run_id="r", record_id="1", prediction="ab", gold_answer="ab"),
        AnnotationRecord(run_id="r", record_id="2", prediction="xyz", gold_answer="ab"),
    ]

    rows = score_records(records, metrics=[length, exact])

    assert rows[0].key == "r:1"
    assert rows[0].scores == {"length": 2.0, "exact": 1.0}
    assert rows[1].scores == {"length": 3.0, "exact": 0.0}
    # Scores are recomputed, not read from any cache on the record.
    assert set(rows[0].scores) == {"length", "exact"}


def test_scores_are_cached_and_only_missing_metrics_recomputed() -> None:
    calls = {"count": 0}

    def counted(_p: str, _g: str, _q: str) -> float:
        calls["count"] += 1
        return 1.0

    metric = Metric("counted", counted)
    record = AnnotationRecord(run_id="r", record_id="1", prediction="a", gold_answer="a")

    assert fill_scores([record], metrics=[metric]) is True
    assert record.scores == {"counted": 1.0}
    assert calls["count"] == 1

    # Second pass uses the cache — no recompute, nothing changed.
    assert fill_scores([record], metrics=[metric]) is False
    assert calls["count"] == 1

    # A newly registered metric is still computed once for the already-cached record.
    other = Metric("other", lambda _p, _g, _q: 0.5)
    assert fill_scores([record], metrics=[metric, other]) is True
    assert record.scores == {"counted": 1.0, "other": 0.5}
    assert calls["count"] == 1  # the cached metric was not recomputed


def test_spearman_and_kendall_basic() -> None:
    assert _spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert _kendall_tau([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert _spearman([1.0], [1.0]) is None  # undefined for a single point
