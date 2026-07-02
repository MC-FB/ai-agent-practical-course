"""Meta-evaluation of answer metrics against human-annotated score bands.

Each annotation carries an acceptable score band ``[human_low, human_high]`` for a
(question, prediction, gold) triple. A metric is "good" on that record when its score
lands inside the band. Metric scores are recomputed live from ``ANSWER_METRICS`` so
adding a metric reuses the same human bands with no re-annotation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from dagqa.eval.metrics import ANSWER_METRICS, Metric

_BOOTSTRAP_SAMPLES = 1000
_CI_ALPHA = 0.05
_MIN_CORRELATION_POINTS = 2


class AnnotationRecord(BaseModel):
    """A hand-picked record plus its human-supplied acceptable score band."""

    run_id: str
    record_id: str
    question: str = ""
    gold_answer: str = ""
    prediction: str = ""
    human_low: float | None = None
    human_high: float | None = None
    note: str | None = None
    # Cached metric scores keyed by metric name; filled lazily so re-opening the table is fast.
    scores: dict[str, float] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.run_id}:{self.record_id}"

    @property
    def band(self) -> tuple[float, float] | None:
        """The (low, high) band, order-normalised, or None if not fully annotated."""
        if self.human_low is None or self.human_high is None:
            return None
        low, high = self.human_low, self.human_high
        return (low, high) if low <= high else (high, low)


class RecordScores(BaseModel):
    """An annotation record with every metric's freshly-computed score."""

    key: str
    run_id: str
    record_id: str
    question: str
    gold_answer: str
    prediction: str
    human_low: float | None
    human_high: float | None
    scores: dict[str, float]


def fill_scores(
    records: Sequence[AnnotationRecord],
    *,
    metrics: Sequence[Metric] = tuple(ANSWER_METRICS),
) -> bool:
    """Populate each record's ``scores`` cache for any metrics it is missing.

    Only metrics absent from the cache are computed, so re-opening an already-scored set is
    instant while a newly registered metric is still computed once. Returns ``True`` if any
    record's cache changed (so the caller can persist it).
    """
    changed = False
    for record in records:
        for metric in metrics:
            if metric.name not in record.scores:
                record.scores[metric.name] = metric.score(
                    record.prediction, record.gold_answer, record.question
                )
                changed = True
    return changed


def score_records(
    records: Sequence[AnnotationRecord],
    *,
    metrics: Sequence[Metric] = tuple(ANSWER_METRICS),
) -> list[RecordScores]:
    """Return each record with the requested metrics' scores, using the cache where present.

    Missing scores are computed and cached in place; a metric added after a record was
    captured is therefore still evaluated (once) rather than read from a stale file.
    """
    fill_scores(records, metrics=metrics)
    names = [metric.name for metric in metrics]
    return [
        RecordScores(
            key=record.key,
            run_id=record.run_id,
            record_id=record.record_id,
            question=record.question,
            gold_answer=record.gold_answer,
            prediction=record.prediction,
            human_low=record.human_low,
            human_high=record.human_high,
            scores={name: record.scores[name] for name in names},
        )
        for record in records
    ]


class MetricEvalResult(BaseModel):
    """Per-metric agreement with the human bands."""

    name: str
    n: int  # annotated records the metric was scored on
    in_range_rate: float  # primary: fraction of records the metric lands in-band
    ci_low: float
    ci_high: float
    mean_center_distance: float  # mean |score - band midpoint| (calibration tiebreaker)
    spearman: float | None  # secondary: rank agreement vs band midpoint
    kendall: float | None


def evaluate_metrics(
    records: Sequence[AnnotationRecord],
    *,
    metrics: Sequence[Metric] = tuple(ANSWER_METRICS),
    bootstrap_samples: int = _BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> list[MetricEvalResult]:
    """Rank ``metrics`` by how often they land inside the human bands.

    ``metrics`` is injectable so callers/tests can avoid loading embedding models.
    Records without a full band are skipped. Results are sorted best-first.
    """
    annotated = [record for record in records if record.band is not None]
    if not annotated:
        return []

    results: list[MetricEvalResult] = []
    for metric in metrics:
        indicators: list[float] = []
        center_distances: list[float] = []
        scores: list[float] = []
        midpoints: list[float] = []
        for record in annotated:
            band = record.band
            assert band is not None  # filtered above; narrows Optional for type checkers
            low, high = band
            midpoint = (low + high) / 2
            score = metric.score(record.prediction, record.gold_answer, record.question)
            scores.append(score)
            midpoints.append(midpoint)
            indicators.append(1.0 if low <= score <= high else 0.0)
            center_distances.append(abs(score - midpoint))

        count = len(annotated)
        ci_low, ci_high = _bootstrap_ci(indicators, samples=bootstrap_samples, seed=seed)
        results.append(
            MetricEvalResult(
                name=metric.name,
                n=count,
                in_range_rate=sum(indicators) / count,
                ci_low=ci_low,
                ci_high=ci_high,
                mean_center_distance=sum(center_distances) / count,
                spearman=_spearman(scores, midpoints),
                kendall=_kendall_tau(scores, midpoints),
            )
        )

    results.sort(key=lambda result: (-result.in_range_rate, result.mean_center_distance))
    return results


def _bootstrap_ci(indicators: Sequence[float], *, samples: int, seed: int) -> tuple[float, float]:
    if not indicators:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    values = np.asarray(indicators, dtype=float)
    count = len(values)
    resample_means = values[rng.integers(0, count, size=(samples, count))].mean(axis=1)
    return (
        float(np.quantile(resample_means, _CI_ALPHA / 2)),
        float(np.quantile(resample_means, 1 - _CI_ALPHA / 2)),
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average_rank = (position + end) / 2 + 1  # 1-based average rank for ties
        for index in range(position, end + 1):
            ranks[order[index]] = average_rank
        position = end + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.std() == 0 or right_array.std() == 0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < _MIN_CORRELATION_POINTS:
        return None
    return _pearson(_average_ranks(left), _average_ranks(right))


def _kendall_tau(left: Sequence[float], right: Sequence[float]) -> float | None:
    count = len(left)
    if count < _MIN_CORRELATION_POINTS:
        return None
    concordant = 0
    discordant = 0
    for i in range(count):
        for j in range(i + 1, count):
            direction = (left[i] - left[j]) * (right[i] - right[j])
            if direction > 0:
                concordant += 1
            elif direction < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return (concordant - discordant) / total
