from __future__ import annotations

from dagqa.planning.normalizer import normalize_plan_dependencies
from dagqa.planning.parser import parse_plan
from dagqa.planning.prompts import PLANNER_SYSTEM
from dagqa.planning.validator import graph_depth, scheduler_waves, validate_plan
from dagqa.schemas import DagNode, DagPlan, Operation, PromptSpec, TaskType
from tests.fixtures import PARALLEL_PLAN


def test_parse_and_validate_parallel_plan(app_config) -> None:
    expected_depth = 2
    plan = parse_plan(PARALLEL_PLAN)
    result = validate_plan(plan, app_config.planner)

    assert result.valid
    assert graph_depth(plan) == expected_depth
    assert scheduler_waves(plan) == [["q1", "q2"], ["q3"]]


def test_planner_prompt_discourages_broad_candidate_enumeration() -> None:
    assert "broad candidate-list or exhaustive enumeration nodes" in PLANNER_SYSTEM


def test_parse_accepts_bare_node_list() -> None:
    plan = parse_plan(
        """
        - id: q1
          label: Add numbers
          task_type: calculation
          question: What is 2 plus 2?
          operation: answer
          depends_on: []
          prompt:
            system: Return JSON.
            user_template: What is 2 plus 2?
          output_schema:
            type: object
        """
    )

    assert plan.nodes[0].id == "q1"
    assert plan.final_node == "q1"


def test_validator_rejects_missing_dependency_reference(app_config) -> None:
    plan = parse_plan(PARALLEL_PLAN.replace("depends_on: [q1, q2]", "depends_on: [q1]"))
    result = validate_plan(plan, app_config.planner)

    assert not result.valid
    assert any("not in depends_on" in error for error in result.errors)


def test_normalizer_adds_referenced_nodes_to_dependencies(app_config) -> None:
    plan = parse_plan(PARALLEL_PLAN.replace("depends_on: [q1, q2]", "depends_on: [q2]"))

    normalized = normalize_plan_dependencies(plan)
    result = validate_plan(normalized, app_config.planner)

    assert normalized.nodes[2].depends_on == ["q2", "q1"]
    assert result.valid


def test_normalizer_repairs_missing_final_node_to_last_node(app_config) -> None:
    plan = parse_plan(PARALLEL_PLAN.replace("final_node: q3", "final_node: missing"))

    normalized = normalize_plan_dependencies(plan)
    result = validate_plan(normalized, app_config.planner)

    assert normalized.final_node == "q3"
    assert result.valid


def test_rejects_dependent_node_that_does_not_prompt_with_child_values(app_config) -> None:
    plan = DagPlan(
        question="Which mountain is taller?",
        final_node="q3",
        nodes=[
            DagNode(
                id="q1",
                label="Japan mountain",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="What is the highest mountain in Japan?",
                prompt=PromptSpec(
                    system="Return JSON only.",
                    user_template="Question: {resolved_question}",
                ),
                output_schema={
                    "type": "object",
                    "properties": {
                        "mountain": {"type": "string"},
                        "elevation_meters": {"type": "number"},
                    },
                    "required": ["mountain", "elevation_meters"],
                },
            ),
            DagNode(
                id="q2",
                label="Germany mountain",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="What is the highest mountain in Germany?",
                prompt=PromptSpec(
                    system="Return JSON only.",
                    user_template="Question: {resolved_question}",
                ),
                output_schema={
                    "type": "object",
                    "properties": {
                        "mountain": {"type": "string"},
                        "elevation_meters": {"type": "number"},
                    },
                    "required": ["mountain", "elevation_meters"],
                },
            ),
            DagNode(
                id="q3",
                label="Compare mountains",
                task_type=TaskType.comparison,
                operation=Operation.compare,
                question="Which mountain is taller?",
                depends_on=["q1", "q2"],
                prompt=PromptSpec(
                    system="Return JSON only.",
                    user_template=(
                        "Question: Which mountain is taller: "
                        "the highest mountain in Japan or Germany?"
                    ),
                ),
                input_map={
                    "japan_mountain": "q1.mountain",
                    "japan_elevation": "q1.elevation_meters",
                    "germany_mountain": "q2.mountain",
                    "germany_elevation": "q2.elevation_meters",
                },
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
        ],
    )

    result = validate_plan(plan, app_config.planner)

    assert not result.valid
    assert any(
        "prompt.user_template does not include child values" in error for error in result.errors
    )


def test_accepts_dependent_node_that_uses_input_map_placeholders(app_config) -> None:
    plan = parse_plan(
        """
question: "Which mountain is taller?"
nodes:
  - id: q1
    label: "Japan mountain"
    task_type: fact_lookup
    operation: answer
    question: "What is the highest mountain in Japan?"
    depends_on: []
    prompt:
      system: "Return JSON only."
      user_template: "Question: {resolved_question}"
    input_map: {}
    output_schema:
      type: object
      required: [mountain, elevation_meters]
      properties:
        mountain:
          type: string
        elevation_meters:
          type: number
  - id: q2
    label: "Germany mountain"
    task_type: fact_lookup
    operation: answer
    question: "What is the highest mountain in Germany?"
    depends_on: []
    prompt:
      system: "Return JSON only."
      user_template: "Question: {resolved_question}"
    input_map: {}
    output_schema:
      type: object
      required: [mountain, elevation_meters]
      properties:
        mountain:
          type: string
        elevation_meters:
          type: number
  - id: q3
    label: "Compare mountains"
    task_type: comparison
    operation: compare
    question: "Which mountain is taller: {q1.mountain} or {q2.mountain}?"
    depends_on: [q1, q2]
    prompt:
      system: "Return JSON only."
      user_template: |
        Compare {japan_mountain} ({japan_elevation} meters) with
        {germany_mountain} ({germany_elevation} meters).
    input_map:
      japan_mountain: q1.mountain
      japan_elevation: q1.elevation_meters
      germany_mountain: q2.mountain
      germany_elevation: q2.elevation_meters
    child_output_policy: "Use both child mountain names and elevations."
    output_schema:
      type: object
      required: [answer]
      properties:
        answer:
          type: string
final_node: q3
"""
    )

    result = validate_plan(plan, app_config.planner)

    assert result.valid


def test_validator_rejects_final_node_without_answer_field(app_config) -> None:
    raw_plan = PARALLEL_PLAN.replace(
        "required: [answer, reasoning]",
        "required: [winner, reasoning]",
    )
    plan = parse_plan(raw_plan)
    plan.nodes[-1].output_schema["properties"].pop("answer", None)
    plan.nodes[-1].output_schema["properties"]["winner"] = {"type": "string"}

    result = validate_plan(plan, app_config.planner)

    assert not result.valid
    assert any("must include 'answer'" in error for error in result.errors)
