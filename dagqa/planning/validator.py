from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from dagqa.config import PlannerConfig
from dagqa.schemas import DagPlan, ValidationResult

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z][A-Za-z0-9_-]*)\.([A-Za-z_][A-Za-z0-9_]*)\}")
_REFERENCE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\.([A-Za-z_][A-Za-z0-9_]*)$")


def validate_plan(  # noqa: PLR0912
    plan: DagPlan,
    config: PlannerConfig | None = None,
) -> ValidationResult:
    errors: list[str] = []
    node_ids = [node.id for node in plan.nodes]
    node_set = set(node_ids)
    by_id = {node.id: node for node in plan.nodes}

    if len(node_ids) != len(node_set):
        errors.append("Node IDs must be unique.")

    if plan.final_node not in node_set:
        errors.append(f"final_node '{plan.final_node}' does not exist.")

    for node in plan.nodes:
        for dep in node.depends_on:
            if dep not in node_set:
                errors.append(f"Node '{node.id}' depends on missing node '{dep}'.")
            if dep == node.id:
                errors.append(f"Node '{node.id}' cannot depend on itself.")

        declared_deps = set(node.depends_on)
        for ref_node, ref_field in _PLACEHOLDER_RE.findall(node.question):
            _validate_reference(
                errors,
                node.id,
                ref_node,
                ref_field,
                declared_deps,
                by_id.get(ref_node),
                "question placeholder",
            )

        for name, ref in node.input_map.items():
            match = _REFERENCE_RE.match(ref)
            if not match:
                errors.append(f"Node '{node.id}' input_map '{name}' must use '<node>.<field>'.")
                continue
            ref_node, ref_field = match.groups()
            _validate_reference(
                errors,
                node.id,
                ref_node,
                ref_field,
                declared_deps,
                by_id.get(ref_node),
                f"input_map '{name}'",
            )

        try:
            Draft202012Validator.check_schema(node.output_schema)
        except SchemaError as exc:
            errors.append(f"Node '{node.id}' output_schema is invalid: {exc.message}")

    missing_dependency = any(dep not in node_set for node in plan.nodes for dep in node.depends_on)
    if not missing_dependency:
        cycle = _find_cycle(plan)
        if cycle:
            errors.append(f"DAG must be acyclic; cycle detected: {' -> '.join(cycle)}.")

        if config is not None:
            if len(plan.nodes) > config.max_nodes:
                errors.append(f"Plan has {len(plan.nodes)} nodes; max_nodes is {config.max_nodes}.")
            depth = graph_depth(plan)
            if depth > config.max_depth:
                errors.append(f"Plan depth is {depth}; max_depth is {config.max_depth}.")

    return ValidationResult(valid=not errors, errors=errors)


def graph_depth(plan: DagPlan) -> int:
    by_id = {node.id: node for node in plan.nodes}
    memo: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        node = by_id[node_id]
        if not node.depends_on:
            memo[node_id] = 1
        else:
            memo[node_id] = 1 + max(depth(dep) for dep in node.depends_on)
        return memo[node_id]

    return max((depth(node.id) for node in plan.nodes), default=0)


def _validate_reference(
    errors: list[str],
    node_id: str,
    ref_node: str,
    ref_field: str,
    declared_deps: set[str],
    ref_node_obj: Any,
    location: str,
) -> None:
    if ref_node not in declared_deps:
        errors.append(
            f"Node '{node_id}' {location} references '{ref_node}.{ref_field}', "
            f"but '{ref_node}' is not in depends_on."
        )
        return
    if ref_node_obj is None:
        return
    properties = ref_node_obj.output_schema.get("properties", {})
    if ref_field not in properties:
        errors.append(
            f"Node '{node_id}' {location} references '{ref_node}.{ref_field}', "
            f"but '{ref_field}' is not in '{ref_node}' output_schema."
        )


def _find_cycle(plan: DagPlan) -> list[str] | None:
    graph = {node.id: list(node.depends_on) for node in plan.nodes}
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node_id: str) -> list[str] | None:
        if node_id in visiting:
            index = stack.index(node_id)
            return [*stack[index:], node_id]
        if node_id in visited:
            return None
        visiting.add(node_id)
        stack.append(node_id)
        for dep in graph.get(node_id, []):
            cycle = visit(dep)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(node_id)
        visited.add(node_id)
        return None

    for node in graph:
        cycle = visit(node)
        if cycle:
            return cycle
    return None


def scheduler_waves(plan: DagPlan) -> list[list[str]]:
    remaining = {node.id: set(node.depends_on) for node in plan.nodes}
    reverse: dict[str, list[str]] = defaultdict(list)
    for node in plan.nodes:
        for dep in node.depends_on:
            reverse[dep].append(node.id)

    ready = deque(sorted(node_id for node_id, deps in remaining.items() if not deps))
    waves: list[list[str]] = []
    completed: set[str] = set()

    while ready:
        wave = list(ready)
        ready.clear()
        waves.append(wave)
        for node_id in wave:
            completed.add(node_id)
            for child in reverse[node_id]:
                remaining[child].discard(node_id)
                if not remaining[child] and child not in completed:
                    ready.append(child)

    if len(completed) != len(remaining):
        raise ValueError("Cannot create waves for cyclic or invalid plan.")
    return waves
