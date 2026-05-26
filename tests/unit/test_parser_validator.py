from __future__ import annotations

from dagqa.planning.parser import parse_plan
from dagqa.planning.validator import graph_depth, scheduler_waves, validate_plan
from tests.fixtures import PARALLEL_PLAN


def test_parse_and_validate_parallel_plan(app_config) -> None:
    expected_depth = 2
    plan = parse_plan(PARALLEL_PLAN)
    result = validate_plan(plan, app_config.planner)

    assert result.valid
    assert graph_depth(plan) == expected_depth
    assert scheduler_waves(plan) == [["q1", "q2"], ["q3"]]


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
