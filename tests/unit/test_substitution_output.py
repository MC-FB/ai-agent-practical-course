from __future__ import annotations

from dagqa.graph.substitution import resolve_input_map, resolve_question
from dagqa.nodes.output_validation import parse_node_output, validate_node_output


def test_resolves_question_and_input_map() -> None:
    outputs = {"q1": {"answer": "Ada Lovelace"}}

    assert resolve_question("When was {q1.answer} born?", outputs) == "When was Ada Lovelace born?"
    assert resolve_input_map({"person": "q1.answer"}, outputs) == {"person": "Ada Lovelace"}


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
