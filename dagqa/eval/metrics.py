from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple
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


@lru_cache(maxsize=1)
def mini_l6_sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=get_device())


# ---------------------------------------------------------------------------
# Additional SBERT bi-encoders trialled as answer metrics. Each is cached for
# the process lifetime (mirroring ``_sentence_transformer``) so the benchmark
# loop loads the weights once, not once per record.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _multi_qa_mpnet_transformer() -> SentenceTransformer:
    # QA-tuned sibling of all-mpnet-base-v2; cosine-normalised, no prefix needed.
    return SentenceTransformer(
        "sentence-transformers/multi-qa-mpnet-base-cos-v1", device=get_device()
    )


@lru_cache(maxsize=1)
def _gte_base_transformer() -> SentenceTransformer:
    # GTE is trained on raw text; no instruction prefix.
    return SentenceTransformer("thenlper/gte-base", device=get_device())


@lru_cache(maxsize=1)
def _bge_base_transformer() -> SentenceTransformer:
    # BGE only prefixes the query in *asymmetric* retrieval; our answer-vs-answer
    # comparison is symmetric, so no prefix.
    return SentenceTransformer("BAAI/bge-base-en-v1.5", device=get_device())


@lru_cache(maxsize=1)
def _e5_base_transformer() -> SentenceTransformer:
    # E5 was trained with every input wearing a prefix; see ``e5_base_cosine_sim``.
    return SentenceTransformer("intfloat/e5-base-v2", device=get_device())


def _prefixed_cosine(
    prediction: str,
    ground_truth: str,
    model_loader: Callable[[], SentenceTransformer],
    query_prefix: str = "",
) -> float:
    """Cosine similarity between two answers, honouring a model's instruction prefix.

    ``model_loader`` is passed uncalled and invoked only *after* the guard clauses,
    so the fast paths (identical or empty strings) never trigger a model load —
    matching how ``cosine_sim`` defers ``_sentence_transformer()`` until it's needed.

    ``query_prefix`` is prepended to BOTH strings because comparing a predicted
    answer to a gold answer is a *symmetric* task (both sides are the same kind of
    text). Models such as E5 were trained with every input wearing this prefix, so
    omitting it feeds them an out-of-distribution input and silently degrades the
    embedding. Models that expect no prefix pass ``query_prefix=""`` and behave
    exactly like the plain ``cosine_sim`` path.
    """
    if prediction == ground_truth:
        return 1.0

    if prediction == "" or ground_truth == "":
        return 0.0

    model = model_loader()
    texts = [query_prefix + prediction, query_prefix + ground_truth]
    pred_embedding, gt_embedding = model.encode(texts)  # (2, D) -> two (D,) rows

    pred_mag = np.linalg.norm(pred_embedding)
    gt_mag = np.linalg.norm(gt_embedding)

    if pred_mag == 0.0 or gt_mag == 0.0:
        return 0.0

    norm_pred_embedding = pred_embedding / pred_mag
    norm_gt_embedding = gt_embedding / gt_mag

    metric = np.dot(norm_pred_embedding, norm_gt_embedding)
    return float(metric)


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


def mini_l6_context_cosine_sim(prediction: str, ground_truth: str, question: str) -> float:
    if prediction == ground_truth:
        return 1.0

    if prediction == "" or ground_truth == "":
        return 0.0

    context_prediction = question + prediction
    context_ground_truth = question + ground_truth

    model = mini_l6_sentence_transformer()
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


def multi_qa_mpnet_cosine_sim(prediction: str, ground_truth: str, _question: str) -> float:
    return _prefixed_cosine(prediction, ground_truth, _multi_qa_mpnet_transformer)


def gte_base_cosine_sim(prediction: str, ground_truth: str, _question: str) -> float:
    return _prefixed_cosine(prediction, ground_truth, _gte_base_transformer)


def bge_base_cosine_sim(prediction: str, ground_truth: str, _question: str) -> float:
    return _prefixed_cosine(prediction, ground_truth, _bge_base_transformer)


def e5_base_cosine_sim(prediction: str, ground_truth: str, _question: str) -> float:
    # E5 expects the "query: " prefix on every input; on a symmetric task both sides get it.
    return _prefixed_cosine(prediction, ground_truth, _e5_base_transformer, query_prefix="query: ")


# ---------------------------------------------------------------------------
# Classical (non-neural) per-pair metrics
#
# Each is a pure function of (prediction, ground_truth) and shares the
# ``(prediction, ground_truth, question) -> float`` signature so it drops
# straight into ``ANSWER_METRICS``. They are CPU-cheap (no model load) and
# operate on ``normalize_answer`` output for consistency with ``exact_match`` /
# ``answer_f1``.
# ---------------------------------------------------------------------------

_BLEU_MAX_ORDER = 4
_CHRF_MAX_ORDER = 6
_CHRF_BETA = 2.0
_METEOR_ALPHA = 0.9  # recall weight in the METEOR F-mean
_METEOR_PENALTY_GAMMA = 0.5
_METEOR_PENALTY_BETA = 3.0


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def _token_ngram_counts(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _char_ngram_counts(text: str, n: int) -> Counter[str]:
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    # Classic dynamic-programming LCS; O(len(left) * len(right)), fine for short answers.
    previous = [0] * (len(right) + 1)
    for token_left in left:
        current = [0]
        for j, token_right in enumerate(right, start=1):
            if token_left == token_right:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def bleu(prediction: str, ground_truth: str, _question: str) -> float:
    """Sentence BLEU (up to 4-grams) with add-1 smoothing and brevity penalty.

    Smoothing and averaging only over n-gram orders present in the prediction keep
    the score usable on short answers. Note this is per-example (macro) BLEU, not
    the corpus BLEU reported on MT leaderboards.
    """
    if prediction == ground_truth:
        return 1.0
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(ground_truth)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    log_precision_sum = 0.0
    effective_order = 0
    for n in range(1, _BLEU_MAX_ORDER + 1):
        pred_ngrams = _token_ngram_counts(pred_tokens, n)
        total = sum(pred_ngrams.values())
        if total == 0:
            continue
        gold_ngrams = _token_ngram_counts(gold_tokens, n)
        overlap = sum((pred_ngrams & gold_ngrams).values())
        precision = (overlap + 1) / (total + 1)  # add-1 (Laplace) smoothing
        log_precision_sum += float(np.log(precision))
        effective_order += 1
    if effective_order == 0:
        return 0.0
    geo_mean = np.exp(log_precision_sum / effective_order)
    brevity_penalty = (
        1.0
        if len(pred_tokens) > len(gold_tokens)
        else np.exp(1 - len(gold_tokens) / len(pred_tokens))
    )
    return float(brevity_penalty * geo_mean)


def bigram_f1(prediction: str, ground_truth: str, _question: str) -> float:
    """F1 over token bigrams — an order-sensitive complement to unigram ``answer_f1``."""
    if prediction == ground_truth:
        return 1.0
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(ground_truth)
    pred_bigrams = _token_ngram_counts(pred_tokens, 2)
    gold_bigrams = _token_ngram_counts(gold_tokens, 2)
    if not pred_bigrams or not gold_bigrams:
        # Fewer than two tokens on a side: fall back to exact token-sequence match.
        return float(pred_tokens == gold_tokens)
    overlap = sum((pred_bigrams & gold_bigrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_bigrams.values())
    recall = overlap / sum(gold_bigrams.values())
    return 2 * precision * recall / (precision + recall)


def chrf(prediction: str, ground_truth: str, _question: str) -> float:
    """Character n-gram F-score (chrF, beta=2) — robust to spelling/entity variants."""
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    if pred == gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    beta_sq = _CHRF_BETA**2
    f_scores: list[float] = []
    for n in range(1, _CHRF_MAX_ORDER + 1):
        pred_ngrams = _char_ngram_counts(pred, n)
        gold_ngrams = _char_ngram_counts(gold, n)
        if not pred_ngrams or not gold_ngrams:
            continue
        overlap = sum((pred_ngrams & gold_ngrams).values())
        if overlap == 0:
            f_scores.append(0.0)
            continue
        precision = overlap / sum(pred_ngrams.values())
        recall = overlap / sum(gold_ngrams.values())
        f_scores.append((1 + beta_sq) * precision * recall / (beta_sq * precision + recall))
    if not f_scores:
        return 0.0
    return sum(f_scores) / len(f_scores)


def rouge_l(prediction: str, ground_truth: str, _question: str) -> float:
    """ROUGE-L: F1 over the longest common subsequence of tokens (order-aware)."""
    if prediction == ground_truth:
        return 1.0
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(ground_truth)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def meteor(prediction: str, ground_truth: str, _question: str) -> float:
    """METEOR (exact-match variant): recall-weighted F-mean with a fragmentation penalty.

    Dependency-free — only exact unigram alignment, without METEOR's WordNet synonym
    or stemming modules. The chunk penalty still rewards contiguous, in-order matches.
    """
    if prediction == ground_truth:
        return 1.0
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(ground_truth)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    # Greedy left-to-right alignment: each pred token maps to the first unused matching
    # gold token, preserving pred order so chunks can be counted.
    gold_positions: dict[str, list[int]] = {}
    for index, token in enumerate(gold_tokens):
        gold_positions.setdefault(token, []).append(index)
    used_gold: set[int] = set()
    alignment: list[tuple[int, int]] = []
    for pred_index, token in enumerate(pred_tokens):
        for gold_index in gold_positions.get(token, []):
            if gold_index not in used_gold:
                used_gold.add(gold_index)
                alignment.append((pred_index, gold_index))
                break
    matches = len(alignment)
    if matches == 0:
        return 0.0
    precision = matches / len(pred_tokens)
    recall = matches / len(gold_tokens)
    fmean = (precision * recall) / (_METEOR_ALPHA * precision + (1 - _METEOR_ALPHA) * recall)
    # A chunk is a maximal run of pairs that are adjacent in both sequences.
    chunks = 1
    for (prev_pred, prev_gold), (curr_pred, curr_gold) in zip(
        alignment, alignment[1:], strict=False
    ):
        if curr_pred != prev_pred + 1 or curr_gold != prev_gold + 1:
            chunks += 1
    penalty = _METEOR_PENALTY_GAMMA * (chunks / matches) ** _METEOR_PENALTY_BETA
    return fmean * (1 - penalty)


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
    Metric("mini_l6_cosine_sim", mini_l6_cosine_sim),
    Metric("mini_l6_context_cosine_sim", mini_l6_context_cosine_sim),
    Metric("multi_qa_mpnet_cosine_sim", multi_qa_mpnet_cosine_sim),
    Metric("gte_base_cosine_sim", gte_base_cosine_sim),
    Metric("bge_base_cosine_sim", bge_base_cosine_sim),
    Metric("e5_base_cosine_sim", e5_base_cosine_sim),
    Metric("bleu", bleu),
    Metric("bigram_f1", bigram_f1),
    Metric("chrf", chrf),
    Metric("rouge_l", rouge_l),
    Metric("meteor", meteor),
]
