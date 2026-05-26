from __future__ import annotations

import json
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from dagqa.schemas import ValidationResult


def parse_node_output(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError("Node output must parse to an object.")
    return parsed


def validate_node_output(output: dict[str, Any], schema: dict[str, Any]) -> ValidationResult:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(output), key=lambda error: list(error.path))
    return ValidationResult(valid=not errors, errors=[error.message for error in errors])
