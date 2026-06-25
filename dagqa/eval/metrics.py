from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    try:
        import torch_npu  # noqa: F401, PLC0415

        if torch.npu.is_available():
            return "npu"
    except (ImportError, AttributeError):
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=get_device())


def mini_l6_sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=get_device())


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(prediction: str, ground_truth: str, _question: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def answer_f1(prediction: str, ground_truth: str, _question: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def cosine_sim(prediction: str, ground_truth: str, _question: str) -> float:
    if prediction == ground_truth:
        return 1.0

    if prediction == "" or ground_truth == "":
        return 0.0

    model = _sentence_transformer()
    pred_embedding, gt_embedding = model.encode([prediction, ground_truth])

    pred_mag = np.linalg.norm(pred_embedding)
    gt_mag = np.linalg.norm(gt_embedding)

    if pred_mag == 0.0 or gt_mag == 0.0:
        return 0.0

    norm_pred_embedding = pred_embedding / pred_mag
    norm_gt_embedding = gt_embedding / gt_mag

    metric = np.dot(norm_pred_embedding, norm_gt_embedding)
    return float(metric)


def context_cosine_sim(prediction: str, ground_truth: str, question: str) -> float:
    if prediction == ground_truth:
        return 1.0

    if prediction == "" or ground_truth == "":
        return 0.0

    context_prediction = question + prediction
    context_ground_truth = question + ground_truth

    model = _sentence_transformer()
    context_pred_embedding, context_gt_embedding, que_embedding = model.encode(
        [context_prediction, context_ground_truth, question]
    )

    pred_embedding = context_pred_embedding - que_embedding
    gt_embedding = context_gt_embedding - que_embedding

    pred_mag = np.linalg.norm(pred_embedding)
    gt_mag = np.linalg.norm(gt_embedding)

    if pred_mag == 0.0 or gt_mag == 0.0:
        return 0.0

    norm_pred_embedding = pred_embedding / pred_mag
    norm_gt_embedding = gt_embedding / gt_mag

    metric = np.dot(norm_pred_embedding, norm_gt_embedding)
    return float(metric)


def mini_l6_cosine_sim(prediction: str, ground_truth: str, _question: str) -> float:
    if prediction == ground_truth:
        return 1.0

    if prediction == "" or ground_truth == "":
        return 0.0

    model = mini_l6_sentence_transformer()
    pred_embedding, gt_embedding = model.encode([prediction, ground_truth])

    pred_mag = np.linalg.norm(pred_embedding)
    gt_mag = np.linalg.norm(gt_embedding)

    if pred_mag == 0.0 or gt_mag == 0.0:
        return 0.0

    norm_pred_embedding = pred_embedding / pred_mag
    norm_gt_embedding = gt_embedding / gt_mag

    metric = np.dot(norm_pred_embedding, norm_gt_embedding)
    return float(metric)


class MetricSample(NamedTuple):
    """A single (score, prediction, ground_truth) triple fed to a metric aggregator.

    Carrying the raw strings alongside the score lets an aggregator apply
    metric-specific filtering (e.g. ``cosine_sim`` drops empty pairs) without
    importing the benchmark record type.
    """

    score: float
    prediction: str
    ground_truth: str


def _mean(samples: Sequence[MetricSample]) -> float:
    return sum(sample.score for sample in samples) / len(samples) if samples else 0.0


def _cosine_mean(samples: Sequence[MetricSample]) -> float:
    # Drop pairs where both sides are empty, mirroring the historical aggregate.
    kept = [sample for sample in samples if sample.prediction or sample.ground_truth]
    return sum(sample.score for sample in kept) / len(kept) if kept else 0.0


@dataclass(frozen=True)
class Metric:
    """A benchmark answer metric: a scoring function plus how to aggregate it.

    To add a metric, append a ``Metric`` to ``ANSWER_METRICS`` — the benchmark
    then computes, stores, and aggregates it automatically.
    """

    name: str  # output key, e.g. "f1"
    score: Callable[[str, str, str], float]  # (prediction, ground_truth, question) -> float
    aggregate: Callable[[Sequence[MetricSample]], float] = _mean
    error_default: float = 0.0  # per-record value stored when an example errors


ANSWER_METRICS: list[Metric] = [
    Metric("exact_match", exact_match),
    Metric("f1", answer_f1),
    Metric("cosine_sim", cosine_sim, aggregate=_cosine_mean),
    Metric("context_cosine_sim", context_cosine_sim),
]
