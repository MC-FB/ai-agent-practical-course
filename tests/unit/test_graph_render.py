from __future__ import annotations

from dagqa.graph.render import render_mermaid
from dagqa.schemas import (
    DagNode,
    DagPlan,
    NodeStatus,
    NodeTrace,
    Operation,
    PromptSpec,
    TaskType,
)


def test_render_mermaid_adds_click_hooks_for_nodes() -> None:
    plan = DagPlan(
        question="Which answer wins?",
        final_node="b",
        nodes=[
            DagNode(
                id="a",
                label="First",
                task_type=TaskType.fact_lookup,
                question="Find the first fact",
                operation=Operation.answer,
                depends_on=[],
                prompt=PromptSpec(system="system", user_template="template"),
                input_map={},
                output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
            ),
            DagNode(
                id="b",
                label="Second",
                task_type=TaskType.synthesis,
                question="Combine values",
                operation=Operation.answer,
                depends_on=["a"],
                prompt=PromptSpec(system="system", user_template="template"),
                input_map={"value": "a.value"},
                output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            ),
        ],
    )
    traces = [
        NodeTrace(
            node_id="a",
            label="First",
            task_type=TaskType.fact_lookup,
            operation=Operation.answer,
            status=NodeStatus.succeeded,
        )
    ]

    mermaid = render_mermaid(plan, traces)

    assert "click a dagqaSelectGraphNode" in mermaid
    assert "click b dagqaSelectGraphNode" in mermaid
    assert "a: First\\nfact_lookup\\nsucceeded" in mermaid
    assert "b: Second\\nsynthesis\\npending" in mermaid


def test_render_mermaid_uses_default_status_for_unmatched_nodes() -> None:
    plan = DagPlan(
        question="Which answer wins?",
        final_node="b",
        nodes=[
            DagNode(
                id="a",
                label="First",
                task_type=TaskType.fact_lookup,
                question="Find the first fact",
                operation=Operation.answer,
                depends_on=[],
                prompt=PromptSpec(system="system", user_template="template"),
                input_map={},
                output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
            ),
            DagNode(
                id="b",
                label="Second",
                task_type=TaskType.synthesis,
                question="Combine values",
                operation=Operation.answer,
                depends_on=["a"],
                prompt=PromptSpec(system="system", user_template="template"),
                input_map={"value": "a.value"},
                output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            ),
        ],
    )

    mermaid = render_mermaid(plan, [], default_status=NodeStatus.succeeded)

    assert "a: First\\nfact_lookup\\nsucceeded" in mermaid
    assert "b: Second\\nsynthesis\\nsucceeded" in mermaid
