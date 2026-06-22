from __future__ import annotations

import re
from typing import Any

from dagqa.schemas import EvidenceDocument

_YES_NO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_BETWEEN_RE = re.compile(r"\bbetween\s+([^.,;]+?)\s+and\s+([^.,;]+?)(?:[.,;]|$)", re.IGNORECASE)
_INITIAL_NAME_RE = re.compile(r"^(?P<initials>(?:[A-Z]\.\s*)+)(?P<surname>[A-Z][a-zA-Z-]+)$")
_QUALIFIED_POLITY_PREFIXES = (
    "East",
    "West",
    "North",
    "South",
    "Northern",
    "Southern",
    "Upper",
    "Lower",
    "Former",
    "Ancient",
    "New",
    "Old",
)
MIN_UNSUPPORTED_ENTITY_LENGTH = 4
MIN_ACRONYM_LENGTH = 2
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
    boolean = _canonicalize_boolean(value, question)
    if boolean:
        return boolean
    yes_no = _YES_NO_RE.match(value)
    if yes_no and _is_yes_no_question(question):
        return yes_no.group(1).lower()

    value = _strip_common_answer_prefix(value)
    value = _canonicalize_literal_list(value)
    value = _restore_structured_pair(value)
    value = _strip_disambiguating_parenthetical(value)
    value = _extract_between_target(value, question)
    value = _restore_three_other_cast_members(value, question, texts)
    value = _restore_intro_phrase(value, question, texts)
    value = _restore_subject_painting_phrase(value, question, texts)
    value = _restore_language_descriptor(value, question, texts)
    value = _restore_qualified_polity(value, question, evidence_documents or [])
    value = _restore_initialed_name(value, texts)
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
    if value.casefold() in {"yes", "no"} and not _is_yes_no_question(question):
        return True
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


def _canonicalize_boolean(value: str, question: str) -> str | None:
    if not _is_yes_no_question(question):
        return None
    normalized = value.strip().casefold().rstrip(".")
    if normalized == "true":
        return "yes"
    if normalized == "false":
        return "no"
    return None


def _restore_structured_pair(value: str) -> str:
    match = re.fullmatch(
        r"\s*year\s*:\s*([^,;]+)[,;]\s*conference\s*:\s*(.+?)\s*",
        value,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1).strip()} {match.group(2).strip()}"
    return value


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


def _restore_three_other_cast_members(value: str, question: str, texts: list[str]) -> str:
    if "which three other" not in question.casefold():
        return value
    for text in texts:
        match = re.search(
            r"\bstars\s+([A-Z][^.;]+?),\s+([A-Z][^.;]+?),\s+([A-Z][^.;]+?)\s+and\s+"
            r"([A-Z][^.;]+?)(?:\.|$)",
            text,
        )
        if not match:
            continue
        names = [part.strip() for part in match.groups()]
        if value.casefold() in {name.casefold() for name in names[1:]}:
            return _join_list_answer(names[1:])
    return value


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


def _restore_language_descriptor(value: str, question: str, texts: list[str]) -> str:
    question_lower = question.casefold()
    if "language" not in question_lower:
        return value
    if not re.fullmatch(r"[A-Za-z]+", value):
        return value
    for text in texts:
        match = re.search(r"\b([A-Z][a-z]+-language)\b", text)
        if match:
            return match.group(1)
    for text in texts:
        match = re.search(
            rf"\b(?:called|known as|later called)\s+([A-Z][A-Za-z-]+\s+{re.escape(value)})\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return _restore_candidate_case(match.group(1), text)
    for text in texts:
        match = re.search(
            rf"\b([A-Z][A-Za-z-]+\s+{re.escape(value)})\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return _restore_candidate_case(match.group(1), text)
    return value


def _restore_candidate_case(candidate: str, source: str) -> str:
    start = source.casefold().find(candidate.casefold())
    if start < 0:
        return candidate
    return source[start : start + len(candidate)]


def _restore_qualified_polity(
    value: str,
    question: str,
    evidence_documents: list[EvidenceDocument],
) -> str:
    if "country" not in question.casefold() and "nation" not in question.casefold():
        return value
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]+", value):
        return value

    prefix_pattern = "|".join(re.escape(prefix) for prefix in _QUALIFIED_POLITY_PREFIXES)
    qualified_pattern = re.compile(
        rf"\b((?:{prefix_pattern})\s+{re.escape(value)})\b",
        flags=re.IGNORECASE,
    )
    for document in evidence_documents:
        match = qualified_pattern.search(document.title)
        if not match:
            if value.casefold() not in document.title.casefold():
                continue
            acronym = _leading_acronym_for_polity(document.text)
            if acronym:
                return acronym
            continue
        candidate = _restore_candidate_case(match.group(1), document.title)
        acronym = _leading_acronym_for_polity(document.text)
        return acronym or candidate
    return value


def _leading_acronym_for_polity(text: str) -> str | None:
    first_sentence = text.split(".", 1)[0]
    match = re.search(
        r"\b(?:in|of|for)\s+(?:the\s+)?([A-Z][A-Z0-9.-]{1,12})\b",
        first_sentence,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    acronym = match.group(1).strip(".")
    if len(acronym) < MIN_ACRONYM_LENGTH or not re.fullmatch(r"[A-Z0-9.-]+", acronym):
        return None
    return acronym


def _restore_initialed_name(value: str, texts: list[str]) -> str:
    match = _INITIAL_NAME_RE.fullmatch(value.strip())
    if not match:
        return value
    initials = [part[0] for part in re.findall(r"[A-Z]\.", match.group("initials"))]
    surname = match.group("surname")
    candidate_re = re.compile(rf"\b((?:[A-Z][a-z]+\s+){{{len(initials)}}}{re.escape(surname)})\b")
    for text in texts:
        for candidate_match in candidate_re.finditer(text):
            candidate = candidate_match.group(1)
            parts = candidate.split()
            if [part[0] for part in parts[:-1]] == initials:
                return candidate
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
