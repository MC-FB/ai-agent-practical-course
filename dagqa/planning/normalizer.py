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
    return plan


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
