from __future__ import annotations

from functools import lru_cache
import re
import string
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    try:
        import torch_npu  # NPU support usually requires importing its specific toolkit
        if torch.npu.is_available():
            return "npu"
    except (ImportError, AttributeError):
        pass
    return "cpu"


@lru_cache(maxsize=1)
def _sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device = get_device())

def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def answer_f1(prediction: str, ground_truth: str) -> float:
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

def cosine_sim(prediction: str, ground_truth: str) -> float:
    if prediction == ground_truth:
        return 1.0
    
    if prediction == "" or ground_truth == "":
        return 0.0
    
    model = _sentence_transformer()
    pred_embedding, gt_embedding  = model.encode([prediction,ground_truth])

    pred_mag = np.linalg.norm(pred_embedding)
    gt_mag = np.linalg.norm(gt_embedding)
    
    if pred_mag == 0.0 or gt_mag == 0.0:
        return 0.0
    
    norm_pred_embedding = pred_embedding / pred_mag 
    norm_gt_embedding = gt_embedding /gt_mag
    
    metric = np.dot(norm_pred_embedding, norm_gt_embedding)
    return metric

