from __future__ import annotations

import re
import textwrap

import yaml
from pydantic import ValidationError

from dagqa.schemas import DagPlan

_FENCE_RE = re.compile(r"```(?:yaml|yml|json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class PlanParseError(ValueError):
    pass


def strip_code_fence(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def _normalize_plan_data(data: object) -> object:
    if isinstance(data, list):
        data = {"nodes": data}
    if not isinstance(data, dict):
        return data

    nodes = data.get("nodes")
    if isinstance(nodes, list):
        data.setdefault("question", "")
        if nodes and "final_node" not in data:
            last = nodes[-1]
            data["final_node"] = last.get("id", "") if isinstance(last, dict) else ""
        for node in nodes:
            if not isinstance(node, dict):
                continue
            question = str(node.get("question") or node.get("label") or "")
            node.setdefault("depends_on", [])
            node.setdefault("input_map", {})
            if isinstance(node["input_map"], dict):
                node["input_map"] = {
                    key: _normalize_reference(value) for key, value in node["input_map"].items()
                }
            node.setdefault(
                "prompt",
                {
                    "system": "Answer the node question. Return JSON only.",
                    "user_template": question,
                },
            )
    return data


def _normalize_reference(value: object) -> object:
    if not isinstance(value, str):
        return value
    reference = value[1:-1] if value.startswith("{") and value.endswith("}") else value
    return reference.replace(".answer.", ".", 1)


def parse_plan(text: str) -> DagPlan:
    body = textwrap.dedent(strip_code_fence(text)).strip()
    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        raise PlanParseError(f"Could not parse DAG YAML: {exc}") from exc
    data = _normalize_plan_data(data)
    try:
        return DagPlan.model_validate(data)
    except ValidationError as exc:
        raise PlanParseError(str(exc)) from exc
