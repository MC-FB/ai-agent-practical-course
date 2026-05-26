from __future__ import annotations

from dagqa.schemas import DagPlan, NodeTrace


def render_mermaid(plan: DagPlan, traces: list[NodeTrace] | None = None) -> str:
    status_by_id = {trace.node_id: trace.status.value for trace in traces or []}
    lines = ["flowchart TD"]
    for node in plan.nodes:
        status = status_by_id.get(node.id, "pending")
        label = f"{node.id}: {node.label}\\n{node.task_type.value}\\n{status}"
        lines.append(f'  {node.id}["{label}"]')
        for dep in node.depends_on:
            lines.append(f"  {dep} --> {node.id}")
    return "\n".join(lines)
