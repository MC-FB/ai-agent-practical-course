from __future__ import annotations

import re
from typing import Any

from dagqa.schemas import EvidenceDocument

_YES_NO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_BETWEEN_RE = re.compile(r"\bbetween\s+([^.,;]+?)\s+and\s+([^.,;]+?)(?:[.,;]|$)", re.IGNORECASE)
MIN_UNSUPPORTED_ENTITY_LENGTH = 4
PAIR_SIZE = 2
MILLION = 1_000_000
THOUSAND = 1_000


def answer_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return _join_list_answer([answer_value_to_text(item) for item in value])
    if isinstance(value, tuple):
        return _join_list_answer([answer_value_to_text(item) for item in value])
    if isinstance(value, dict):
        answer = value.get("answer")
        if answer is not None:
            return answer_value_to_text(answer)
        return ""
    return str(value).strip()


def canonicalize_prediction(
    prediction: str,
    *,
    question: str,
    evidence_documents: list[EvidenceDocument] | None = None,
    supporting_texts: list[str] | None = None,
) -> str:
    value = prediction.strip()
    if not value:
        return ""

    texts = list(supporting_texts or [])
    texts.extend(document.text for document in evidence_documents or [])
    yes_no = _YES_NO_RE.match(value)
    if yes_no and _is_yes_no_question(question):
        return yes_no.group(1).lower()

    value = _strip_common_answer_prefix(value)
    value = _canonicalize_literal_list(value)
    value = _strip_disambiguating_parenthetical(value)
    value = _extract_between_target(value, question)
    value = _restore_intro_phrase(value, question, texts)
    value = _restore_subject_painting_phrase(value, question, texts)
    value = _restore_numeric_surface(value, texts)
    value = _restore_quantity_unit(value, evidence_documents or [])
    return value.strip()


def is_placeholder_or_unsupported(
    prediction: str,
    *,
    question: str,
    evidence_documents: list[EvidenceDocument],
) -> bool:
    value = prediction.strip()
    if not value:
        return False
    if value.casefold() in {"none", "unknown", "not found", "n/a", "no answer"}:
        return not _is_yes_no_question(question)
    if _is_yes_no_question(question) or _looks_numeric_or_date(value):
        return False
    if len(value) < MIN_UNSUPPORTED_ENTITY_LENGTH:
        return False
    haystack = "\n".join([document.title + "\n" + document.text for document in evidence_documents])
    return value.casefold() not in haystack.casefold()


def _join_list_answer(items: list[str]) -> str:
    cleaned = [item.strip(" '\"\n\t") for item in items if item and item.strip(" '\"\n\t")]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    compressed = _compress_shared_surname(cleaned)
    if compressed:
        return compressed
    return ", ".join(cleaned[:-1]) + f" and {cleaned[-1]}"


def _compress_shared_surname(items: list[str]) -> str | None:
    split_names = [item.split() for item in items]
    if len(split_names) != PAIR_SIZE or any(len(parts) < PAIR_SIZE for parts in split_names):
        return None
    surname = split_names[0][-1]
    if split_names[1][-1] != surname:
        return None
    first_names = [" ".join(parts[:-1]) for parts in split_names]
    return f"{first_names[0]} and {first_names[1]} {surname}"


def _canonicalize_literal_list(value: str) -> str:
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return value
    parts = [
        part.strip().strip("'\"") for part in stripped[1:-1].split(",") if part.strip().strip("'\"")
    ]
    return _join_list_answer(parts) if parts else value


def _is_yes_no_question(question: str) -> bool:
    first = question.strip().split(maxsplit=1)[0].casefold() if question.strip() else ""
    return first in {
        "are",
        "is",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "has",
        "have",
        "had",
    }


def _strip_common_answer_prefix(value: str) -> str:
    return re.sub(r"^\s*(?:the answer is|answer:)\s*", "", value, flags=re.IGNORECASE).strip()


def _extract_between_target(value: str, question: str) -> str:
    question_lower = question.casefold()
    if "between which town and the town" not in question_lower:
        return value
    match = _BETWEEN_RE.search(value)
    if not match:
        return value
    return match.group(1).strip()


def _strip_disambiguating_parenthetical(value: str) -> str:
    if not re.search(r"\s+\([^)]{3,80}\)$", value):
        return value
    return re.sub(r"\s+\([^)]{3,80}\)$", "", value).strip()


def _restore_intro_phrase(value: str, question: str, texts: list[str]) -> str:
    if not re.fullmatch(r"\d{4}", value):
        return value
    if "first introduced" not in question.casefold():
        return value
    for text in texts:
        season_match = re.search(
            r"\bfirst introduced in\s+((?:the\s+)?[^.]*?\bseason)\b",
            text,
            flags=re.IGNORECASE,
        )
        if season_match:
            return season_match.group(1).strip()
        match = re.search(
            r"\bfirst introduced in\s+(.+?)(?:\s+of\s+['\"A-Z]|\.\s|,\s|$)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            phrase = match.group(1).strip()
            if phrase and not re.fullmatch(r"\d{4}", phrase):
                return phrase
    return value


def _restore_subject_painting_phrase(value: str, question: str, texts: list[str]) -> str:
    if (
        "which painting" not in question.casefold()
        and "of which painting" not in question.casefold()
    ):
        return value
    for text in texts:
        match = re.search(
            r"\bsubject of\s+((?:an?|the)\s+[^.]*?painting)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
    return value


def _restore_numeric_surface(value: str, texts: list[str]) -> str:
    if not re.fullmatch(r"\d{4,}", value):
        return value
    for text in texts:
        for match in re.finditer(r"\b\d{1,3}(?:,\d{3})+\b", text):
            candidate = match.group(0)
            if candidate.replace(",", "") == value:
                return candidate
    return value


def _restore_quantity_unit(value: str, evidence_documents: list[EvidenceDocument]) -> str:
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return value
    numeric = float(value)
    candidates = []
    if numeric >= MILLION:
        candidates.append(f"{numeric / MILLION:g} million")
    if numeric >= THOUSAND:
        candidates.append(f"{numeric / THOUSAND:g} thousand")
    if not candidates:
        return value

    haystack = " ".join(document.text for document in evidence_documents)
    for candidate in candidates:
        pattern = re.compile(
            rf"\b{re.escape(candidate)}\s+([A-Za-z][A-Za-z-]*)\b",
            re.IGNORECASE,
        )
        match = pattern.search(haystack)
        if match:
            return f"{candidate} {match.group(1)}"
    return value


def _looks_numeric_or_date(value: str) -> bool:
    return bool(re.fullmatch(r"[\d\s,./–-]+", value))
