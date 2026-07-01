from __future__ import annotations

from dagqa.graph.substitution import resolve_input_map, resolve_question
from dagqa.nodes.output_validation import parse_node_output, validate_node_output
from dagqa.nodes.prompts import render_node_prompt
from dagqa.schemas import DagNode, Operation, PromptSpec, TaskType


def test_resolves_question_and_input_map() -> None:
    outputs = {"q1": {"answer": "Ada Lovelace"}}

    assert resolve_question("When was {q1.answer} born?", outputs) == "When was Ada Lovelace born?"
    assert resolve_input_map({"person": "q1.answer"}, outputs) == {"person": "Ada Lovelace"}


def test_render_prompt_injects_chain_of_answers_when_template_omits_dependencies() -> None:
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

    assert "Answers so far:" in rendered
    assert "q1: Mount Fuji" in rendered
    assert "Question: Which is larger?" in rendered


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


def test_render_prompt_includes_chain_of_answers_for_dependent_nodes() -> None:
    node = DagNode(
        id="q2",
        label="Meeting participant",
        task_type=TaskType.fact_lookup,
        operation=Operation.answer,
        question="The leader visiting {q1.answer} met with whom on November 22?",
        depends_on=["q1"],
        prompt=PromptSpec(
            system="Return JSON only.",
            user_template="Question: {resolved_question}",
        ),
        input_map={"origin": "q1.answer"},
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )

    rendered = render_node_prompt(
        node,
        "The leader visiting Ukraine met with whom on November 22?",
        {"origin": "Ukraine"},
        {
            "q1": {
                "answer": "Ukraine",
                "reasoning": "Spielberg's grandparents were from Ukraine.",
            }
        },
        original_question=(
            "The leader visiting where Steven Spielberg's grandparents are from met with whom "
            "on November 22?"
        ),
    )

    assert "Original question: The leader visiting where Steven Spielberg" in rendered
    assert "q1: Ukraine" in rendered
    assert "Question: The leader visiting Ukraine" in rendered


def test_render_prompt_includes_original_context_for_independent_node() -> None:
    node = DagNode(
        id="q1",
        label="Largest scoped state",
        task_type=TaskType.fact_lookup,
        operation=Operation.answer,
        question="What is the largest state in the setting region?",
        depends_on=[],
        prompt=PromptSpec(
            system="Return JSON only.",
            user_template="Question: {resolved_question}",
        ),
        input_map={},
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )

    rendered = render_node_prompt(
        node,
        "What is the largest state in the setting region?",
        {},
        {},
        original_question="What is the population of the largest state where the novel is set?",
    )

    assert rendered.startswith("Question context for this node:")
    assert "Original user question: What is the population of the largest state" in rendered
    assert "scoped superlative" in rendered


def test_parses_fenced_json_with_surrounding_text() -> None:
    output = parse_node_output('Here is the object:\n```json\n{"answer": "yes"}\n```\n')

    assert output == {"answer": "yes"}
