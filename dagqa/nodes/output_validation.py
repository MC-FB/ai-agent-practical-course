from __future__ import annotations

import json
import re
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from dagqa.schemas import ValidationResult

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
