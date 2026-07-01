from __future__ import annotations

import json
import re
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from dagqa.schemas import EvidenceSelection, TaskType, ValidationResult

_FENCE_RE = re.compile(r"```(?:json|yaml)?\s*([\s\S]*?)```", re.IGNORECASE)


def parse_node_output(raw: str) -> dict[str, Any]:
    parsed = _parse_raw(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Node output must parse to an object.")
    return parsed


def _parse_raw(raw: str) -> Any:
    text = raw.strip()
    candidates = [text]
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())
    extracted = _extract_json_like(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    errors: list[Exception] = []
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception as exc:
            errors.append(exc)
        try:
            return yaml.safe_load(candidate)
        except Exception as exc:
            errors.append(exc)
    raise ValueError(str(errors[-1]) if errors else "Could not parse node output.")


def _extract_json_like(text: str) -> str | None:
    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not starts:
        return None
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return None
    return text[start : end + 1]


def validate_node_output(output: dict[str, Any], schema: dict[str, Any]) -> ValidationResult:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(output), key=lambda error: list(error.path))
    return ValidationResult(valid=not errors, errors=[error.message for error in errors])


def coerce_output_to_schema(output: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return output
    coerced = dict(output)
    for key, property_schema in properties.items():
        if key not in coerced or not isinstance(property_schema, dict):
            continue
        if property_schema.get("type") == "string" and not isinstance(coerced[key], str):
            coerced[key] = _value_to_text(coerced[key])
    return coerced


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        parts = [_value_to_text(item).strip() for item in value]
        cleaned = [part for part in parts if part]
        if len(cleaned) <= 1:
            return cleaned[0] if cleaned else ""
        return ", ".join(cleaned[:-1]) + f" and {cleaned[-1]}"
    if isinstance(value, dict):
        answer = value.get("answer")
        if answer is not None:
            return _value_to_text(answer)
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def validate_evidence_citations(
    output: dict[str, Any],
    supporting_evidence: EvidenceSelection | None,
) -> ValidationResult:
    if supporting_evidence is None or "_evidence_citations" not in output:
        return ValidationResult(valid=True)

    documents = {document.id: document for document in supporting_evidence.documents}
    errors: list[str] = []
    citations = output.get("_evidence_citations")
    if not isinstance(citations, list):
        return ValidationResult(valid=False, errors=["_evidence_citations must be an array."])

    for citation_index, citation in enumerate(citations):
        if not isinstance(citation, dict):
            errors.append(f"_evidence_citations[{citation_index}] must be an object.")
            continue

        document_id = citation.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            errors.append(f"_evidence_citations[{citation_index}].document_id must be a string.")
            continue
        document = documents.get(document_id)
        if document is None:
            errors.append(
                f"_evidence_citations[{citation_index}].document_id '{document_id}' "
                "is not one of the supplied evidence document IDs."
            )
            continue

        title = citation.get("title")
        if title != document.title:
            errors.append(
                f"_evidence_citations[{citation_index}].title must match the supplied title "
                f"for document_id '{document_id}' exactly: '{document.title}'."
            )

        _validate_citation_fact(errors, citation_index, citation, document_id, document.text)

        sentence_indices = citation.get("sentence_indices")
        if not isinstance(sentence_indices, list):
            errors.append(
                f"_evidence_citations[{citation_index}].sentence_indices must be an array."
            )
            continue
        max_sentence_index = _max_sentence_index(document)
        for sentence_index in sentence_indices:
            if not isinstance(sentence_index, int):
                errors.append(
                    f"_evidence_citations[{citation_index}].sentence_indices contains a "
                    "non-integer value."
                )
            elif sentence_index < 0 or sentence_index > max_sentence_index:
                errors.append(
                    f"_evidence_citations[{citation_index}].sentence_indices contains "
                    f"{sentence_index}, but document_id '{document_id}' only has sentence "
                    f"indices 0 through {max_sentence_index}."
                )

    return ValidationResult(valid=not errors, errors=errors)


def validate_factual_output_grounding(
    output: dict[str, Any],
    supporting_evidence: EvidenceSelection | None,
    task_type: TaskType,
) -> ValidationResult:
    if supporting_evidence is None or task_type not in {
        TaskType.fact_lookup,
        TaskType.entity_resolution,
        TaskType.date_lookup,
    }:
        return ValidationResult(valid=True)

    evidence_text = "\n".join(
        f"{document.title}\n{document.text}" for document in supporting_evidence.documents
    )
    errors: list[str] = []
    for key, value in output.items():
        if key == "_evidence_citations" or key not in _GROUNDED_VALUE_FIELDS:
            continue
        for surface in _iter_grounded_surfaces(value):
            if not _surface_requires_grounding(surface):
                continue
            if not _contains_surface(evidence_text, surface):
                errors.append(
                    f"{key} value '{surface}' is not an exact span in the supplied evidence."
                )
    return ValidationResult(valid=not errors, errors=errors)


def _max_sentence_index(document: Any) -> int:
    sentences = document.metadata.get("sentences")
    if isinstance(sentences, list) and sentences:
        return len(sentences) - 1
    return 0


_GROUNDED_VALUE_FIELDS = {
    "answer",
    "bridge_answer",
    "date",
    "entity",
    "family_or_parent_name",
    "language",
    "local_name",
    "location",
    "name",
    "person",
    "place",
    "value",
}
_UNGROUNDED_PLACEHOLDERS = {"", "unknown", "not found", "not_found", "none", "n/a", "null"}


def _iter_grounded_surfaces(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, int | float | bool):
        return [str(value)]
    if isinstance(value, list):
        surfaces: list[str] = []
        for item in value:
            surfaces.extend(_iter_grounded_surfaces(item))
        return surfaces
    if isinstance(value, dict):
        surfaces = []
        for nested_key in _GROUNDED_VALUE_FIELDS:
            if nested_key in value:
                surfaces.extend(_iter_grounded_surfaces(value[nested_key]))
        return surfaces
    return []


def _surface_requires_grounding(surface: str) -> bool:
    cleaned = surface.strip()
    if cleaned.lower() in _UNGROUNDED_PLACEHOLDERS:
        return False
    return len(cleaned) > 1


def _validate_citation_fact(
    errors: list[str],
    citation_index: int,
    citation: dict[str, Any],
    document_id: str,
    document_text: str,
) -> None:
    fact = citation.get("fact")
    if not isinstance(fact, str) or not fact.strip():
        errors.append(f"_evidence_citations[{citation_index}].fact must be a non-empty string.")
        return
    if not _contains_surface(document_text, fact):
        errors.append(
            f"_evidence_citations[{citation_index}].fact is not copied from the cited "
            f"document_id '{document_id}'."
        )


def _contains_surface(text: str, surface: str) -> bool:
    needle = _normalize_surface(surface)
    if not needle:
        return True
    haystack = _normalize_surface(text)
    return needle in haystack


def _normalize_surface(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()
