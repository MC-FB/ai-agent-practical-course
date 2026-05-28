from __future__ import annotations

from dagqa.graph.substitution import resolve_input_map, resolve_question
from dagqa.nodes.output_validation import parse_node_output, validate_node_output
from dagqa.nodes.prompts import render_node_prompt
from dagqa.schemas import DagNode, Operation, PromptSpec, TaskType


def test_resolves_question_and_input_map() -> None:
    outputs = {"q1": {"answer": "Ada Lovelace"}}

    assert resolve_question("When was {q1.answer} born?", outputs) == "When was Ada Lovelace born?"
    assert resolve_input_map({"person": "q1.answer"}, outputs) == {"person": "Ada Lovelace"}


def test_render_prompt_injects_dependency_values_when_template_omits_dependencies() -> None:
    node = DagNode(
        id="q2",
        label="Compare",
        task_type=TaskType.comparison,
        operation=Operation.compare,
        question="Which is larger?",
        depends_on=["q1"],
        prompt=PromptSpec(
            system="Return JSON only.",
            user_template="Question: {resolved_question}",
        ),
        input_map={"value": "q1.answer"},
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )

    rendered = render_node_prompt(
        node,
        "Which is larger?",
        {"value": "Mount Fuji"},
        {"q1": {"answer": "Mount Fuji"}},
    )

    assert rendered.startswith("Dependency values available to this node:")
    assert '"value": "Mount Fuji"' in rendered


def test_parses_and_validates_node_output() -> None:
    output = parse_node_output('{"answer": "Ada Lovelace"}')
    validation = validate_node_output(
        output,
        {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
    )

    assert validation.valid


def test_parses_fenced_json_with_surrounding_text() -> None:
    output = parse_node_output(
        'Here is the object:\n```json\n{"answer": "yes"}\n```\n'
    )

    assert output == {"answer": "yes"}
