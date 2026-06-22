from __future__ import annotations

import re

from dagqa.schemas import DagPlan

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z][A-Za-z0-9_-]*)\.([A-Za-z_][A-Za-z0-9_]*)\}")
_REFERENCE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\.([A-Za-z_][A-Za-z0-9_]*)$")


def normalize_plan_dependencies(plan: DagPlan) -> DagPlan:
    node_ids = {node.id for node in plan.nodes}
    if plan.final_node not in node_ids and plan.nodes:
        plan.final_node = plan.nodes[-1].id
    for node in plan.nodes:
        dependencies = list(node.depends_on)
        referenced_nodes = _referenced_node_ids(
            node.question,
            node.prompt.user_template,
            node.input_map,
        )
        for ref_node in referenced_nodes:
            if ref_node in node_ids and ref_node != node.id and ref_node not in dependencies:
                dependencies.append(ref_node)
        node.depends_on = dependencies
        if node.id == plan.final_node:
            _ensure_final_answer_contract(node.output_schema)
    return plan


def _ensure_final_answer_contract(schema: dict) -> None:
    properties = schema.setdefault("properties", {})
    required = schema.setdefault("required", [])
    if "answer" not in properties:
        properties["answer"] = {
            "type": "string",
            "description": "Concise final answer to the original user question.",
        }
    if "answer" not in required:
        required.insert(0, "answer")
    for field, description in {
        "answer_type": (
            "Answer type requested by the original question, such as country, region, event/date, "
            "era/decade, language, organization, person, place, yes/no, or number."
        ),
        "answer_source_span": (
            "Shortest dependency or evidence span that directly supports the final answer."
        ),
    }.items():
        properties.setdefault(field, {"type": "string", "description": description})
        if field not in required:
            required.append(field)


def _referenced_node_ids(
    question: str,
    user_template: str,
    input_map: dict[str, str],
) -> set[str]:
    refs = {node_id for node_id, _field in _PLACEHOLDER_RE.findall(question)}
    refs.update({node_id for node_id, _field in _PLACEHOLDER_RE.findall(user_template)})
    for reference in input_map.values():
        match = _REFERENCE_RE.match(reference)
        if match:
            refs.add(match.group(1))
    return refs
