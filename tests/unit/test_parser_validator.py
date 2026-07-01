from __future__ import annotations

from dagqa.planning.normalizer import normalize_plan_dependencies
from dagqa.planning.parser import parse_plan
from dagqa.planning.planner import STRUCTURED_PLANNER_SYSTEM, structured_planner_prompt
from dagqa.planning.prompts import PLANNER_SYSTEM, planner_user_prompt
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


def test_structured_planner_prompt_discourages_broad_candidate_enumeration() -> None:
    prompt = structured_planner_prompt(
        "Hotel Splendide is a British film from 2000 that features which James Bond actor?",
        max_nodes=5,
        max_depth=3,
    )

    assert "broad candidate-list or exhaustive enumeration nodes" in STRUCTURED_PLANNER_SYSTEM
    assert "Prefer targeted lookup nodes over broad candidate-list nodes" in prompt
    assert "do not ask for all James Bond actors" in prompt


def test_planner_prompts_require_input_map_to_consume_dependencies() -> None:
    prompt = structured_planner_prompt(
        "Which show did the journalist host?",
        max_nodes=5,
        max_depth=3,
    )

    assert "input_map must consume every node listed in depends_on" in PLANNER_SYSTEM
    assert "Every node listed in depends_on must be consumed" in prompt
    assert "Do not declare unused dependencies" in STRUCTURED_PLANNER_SYSTEM


def test_planner_prompts_preserve_bridge_entity_anchoring() -> None:
    prompt = planner_user_prompt(
        "Before composing music for the game which contains Chocobos and Moogles, "
        "what cover band was he involved in?",
        max_nodes=5,
        max_depth=3,
    )

    assert "preserve the bridge entity across hops" in PLANNER_SYSTEM
    assert "Preserve bridge entities" in prompt
    assert "resolved bridge value" in structured_planner_prompt(
        "Before composing music for the game which contains Chocobos and Moogles, "
        "what cover band was he involved in?",
        max_nodes=5,
        max_depth=3,
    )


def test_planner_prompts_preserve_scoped_superlatives() -> None:
    prompt = planner_user_prompt(
        "What is the population of the largest state where The Handmaid's Tale is set?",
        max_nodes=5,
        max_depth=3,
    )
    structured_prompt = structured_planner_prompt(
        "What is the population of the largest state where The Handmaid's Tale is set?",
        max_nodes=5,
        max_depth=3,
    )

    assert "Preserve scoped superlatives" in PLANNER_SYSTEM
    assert "largest state in New England" in prompt
    assert "What is the largest state in New England?" in structured_prompt


def test_validator_rejects_unscoped_global_superlative_lookup(app_config) -> None:
    plan = DagPlan(
        question="What is the population of the largest state where The Handmaid's Tale is set?",
        final_node="q3",
        nodes=[
            DagNode(
                id="q1",
                label="Setting",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="Where is The Handmaid's Tale set?",
                prompt=PromptSpec(system="Return JSON.", user_template="Where is it set?"),
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
            DagNode(
                id="q2",
                label="Largest state",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="What is the largest U.S. state by area?",
                prompt=PromptSpec(
                    system="Return JSON.",
                    user_template="What is the largest U.S. state by area?",
                ),
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
            DagNode(
                id="q3",
                label="Population",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="What is the population of {q2.answer}?",
                depends_on=["q2"],
                prompt=PromptSpec(
                    system="Return JSON.",
                    user_template="What is the population of {q2.answer}? {dependencies}",
                ),
                input_map={"state": "q2.answer"},
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
    assert any("scoped superlative 'largest state'" in error for error in result.errors)


def test_validator_accepts_scoped_superlative_with_bridge_dependency(app_config) -> None:
    plan = DagPlan(
        question="What is the population of the largest state where The Handmaid's Tale is set?",
        final_node="q3",
        nodes=[
            DagNode(
                id="q1",
                label="Setting",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="Where is The Handmaid's Tale set?",
                prompt=PromptSpec(system="Return JSON.", user_template="Where is it set?"),
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
            DagNode(
                id="q2",
                label="Largest state in setting",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="What is the largest state in {q1.answer}?",
                depends_on=["q1"],
                prompt=PromptSpec(
                    system="Return JSON.",
                    user_template=("What is the largest state in {q1.answer}? {dependencies}"),
                ),
                input_map={"setting": "q1.answer"},
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
            DagNode(
                id="q3",
                label="Population",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="What is the population of {q2.answer}?",
                depends_on=["q2"],
                prompt=PromptSpec(
                    system="Return JSON.",
                    user_template="What is the population of {q2.answer}? {dependencies}",
                ),
                input_map={"state": "q2.answer"},
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
        ],
    )

    assert validate_plan(plan, app_config.planner).valid


def test_planner_prompts_preserve_temporal_boundary_wording() -> None:
    prompt = planner_user_prompt(
        "A sparse image is used by FileVault in versions later than which?",
        max_nodes=5,
        max_depth=3,
    )
    structured_prompt = structured_planner_prompt(
        "A sparse image is used by FileVault in versions later than which?",
        max_nodes=5,
        max_depth=3,
    )

    assert "later than" in PLANNER_SYSTEM
    assert "boundary" in prompt
    assert "boundary/threshold" in structured_prompt


def test_planner_prompts_preserve_quoted_title_targets() -> None:
    prompt = planner_user_prompt(
        'Which German philosopher wrote "The opera Lulu"?',
        max_nodes=5,
        max_depth=3,
    )
    structured_prompt = structured_planner_prompt(
        'Which German philosopher wrote "The opera Lulu"?',
        max_nodes=5,
        max_depth=3,
    )

    assert "Preserve quoted title wording" in PLANNER_SYSTEM
    assert "Preserve quoted titles" in prompt
    assert "Preserve quoted titles as answer targets" in structured_prompt


def test_planner_prompts_preserve_score_table_operations() -> None:
    structured_prompt = structured_planner_prompt(
        "When was the last time a team beat the cup winner?",
        max_nodes=10,
        max_depth=6,
    )

    assert "Preserve table operations" in structured_prompt
    assert "scan" in structured_prompt
    assert "all table rows" in structured_prompt
    assert "omitted opponents" in structured_prompt


def test_planner_prompts_preserve_capital_duration_spans() -> None:
    structured_prompt = structured_planner_prompt(
        "How long had one headquarters location been the capitol city of another?",
        max_nodes=10,
        max_depth=6,
    )

    assert "Preserve capital/capitol duration spans" in structured_prompt
    assert "had been the capital city" in structured_prompt
    assert "modern city/province/country reasoning" in structured_prompt


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


def test_normalizer_adds_intermediate_answer_contract_to_nonfinal_bridge_nodes() -> None:
    plan = normalize_plan_dependencies(parse_plan(PARALLEL_PLAN))
    first_child_schema = plan.nodes[0].output_schema
    final_schema = plan.nodes[2].output_schema

    assert {"answer", "reasoning"} <= set(first_child_schema["required"])
    assert "bridge_answer" not in first_child_schema["properties"]
    assert "constraint_status" not in first_child_schema["properties"]
    assert "bridge_answer" not in final_schema["properties"]


def test_normalizer_repairs_missing_final_node_to_last_node(app_config) -> None:
    plan = parse_plan(PARALLEL_PLAN.replace("final_node: q3", "final_node: missing"))

    normalized = normalize_plan_dependencies(plan)
    result = validate_plan(normalized, app_config.planner)

    assert normalized.final_node == "q3"
    assert result.valid


def test_normalizer_adds_final_answer_contract(app_config) -> None:
    plan = parse_plan(PARALLEL_PLAN)

    normalized = normalize_plan_dependencies(plan)
    final = next(node for node in normalized.nodes if node.id == normalized.final_node)
    result = validate_plan(normalized, app_config.planner)

    assert final.output_schema["properties"]["answer"]["type"] == "string"
    assert final.output_schema["properties"]["answer_type"]["type"] == "string"
    assert final.output_schema["properties"]["answer_source_span"]["type"] == "string"
    assert {"answer", "answer_type", "answer_source_span"} <= set(final.output_schema["required"])
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


def test_accepts_dependent_lookup_with_simple_answer_reference(app_config) -> None:
    plan = DagPlan(
        question=(
            "The leader visiting where Steven Spielberg's grandparents are from met with whom "
            "on November 22?"
        ),
        final_node="q2",
        nodes=[
            DagNode(
                id="q1",
                label="Grandparents origin",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="Where are Steven Spielberg's grandparents from?",
                prompt=PromptSpec(system="Return JSON only.", user_template="{resolved_question}"),
                output_schema={
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["answer", "reasoning"],
                },
            ),
            DagNode(
                id="q2",
                label="Meeting participant",
                task_type=TaskType.fact_lookup,
                operation=Operation.answer,
                question="The leader visiting {q1.answer} met with whom on November 22?",
                depends_on=["q1"],
                prompt=PromptSpec(
                    system="Return JSON only.",
                    user_template="The leader visiting {origin} met with whom on November 22?",
                ),
                input_map={"origin": "q1.answer"},
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
        ],
    )

    normalized = normalize_plan_dependencies(plan)
    result = validate_plan(normalized, app_config.planner)

    assert result.valid


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
