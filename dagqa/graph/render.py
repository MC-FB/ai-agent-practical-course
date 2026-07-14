from __future__ import annotations

from dagqa.schemas import DagPlan, NodeStatus, NodeTrace


def render_mermaid(
    plan: DagPlan,
    traces: list[NodeTrace] | None = None,
    default_status: NodeStatus = NodeStatus.pending,
) -> str:
    status_by_id = {trace.node_id: trace.status.value for trace in traces or []}
    node_statuses: dict[str, str] = {}
    lines = [
        "flowchart TD",
        "  classDef succeededNode fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:1.5px",
    ]
    for node in plan.nodes:
        status = status_by_id.get(node.id, default_status.value)
        node_statuses[node.id] = status
        label = f"{node.id}: {node.label}\\n{node.task_type.value}\\n{status}"
        lines.append(f'  {node.id}["{label}"]')
        lines.append(f"  click {node.id} dagqaSelectGraphNode")
        for dep in node.depends_on:
            lines.append(f"  {dep} --> {node.id}")
    for node_id, status in node_statuses.items():
        if status == NodeStatus.succeeded.value:
            lines.append(f"  class {node_id} succeededNode")
    return "\n".join(lines)
