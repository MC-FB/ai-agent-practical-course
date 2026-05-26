from __future__ import annotations

import re
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z][A-Za-z0-9_-]*)\.([A-Za-z_][A-Za-z0-9_]*)\}")


class MissingDependencyValue(KeyError):
    pass


def get_path(outputs: dict[str, dict[str, Any]], reference: str) -> Any:
    node_id, field = reference.split(".", 1)
    try:
        return outputs[node_id][field]
    except KeyError as exc:
        raise MissingDependencyValue(reference) from exc


def resolve_input_map(
    input_map: dict[str, str],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {name: get_path(outputs, reference) for name, reference in input_map.items()}


def resolve_question(question: str, outputs: dict[str, dict[str, Any]]) -> str:
    def replace(match: re.Match[str]) -> str:
        node_id, field = match.groups()
        value = get_path(outputs, f"{node_id}.{field}")
        return str(value)

    return _PLACEHOLDER_RE.sub(replace, question)
