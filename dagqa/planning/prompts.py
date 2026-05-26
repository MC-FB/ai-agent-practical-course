PLANNER_SYSTEM = """You are a planner for an inspectable DAG question-answering agent.
Return YAML only. Do not include markdown fences.

Create a DAG where independent subquestions can run in parallel and dependent
subquestions declare their dependencies explicitly.

Every node must include:
- id
- label
- task_type: fact_lookup, entity_resolution, date_lookup, comparison, calculation,
  classification, or synthesis
- operation: answer, transform, compare, or synthesize
- question
- depends_on
- prompt.system
- prompt.user_template
- input_map
- child_output_policy when the node depends on children
- output_schema

Each node output must be a JSON object. Use output_schema to define the object.
Use placeholders like {q1.answer} only when q1 is in depends_on.
"""


def planner_user_prompt(question: str, max_nodes: int, max_depth: int) -> str:
    return f"""User question:
{question}

Constraints:
- max_nodes: {max_nodes}
- max_depth: {max_depth}
- final_node must identify the node containing the final answer.
- Do not add retrieval or document-search nodes.
- Do not verify factual correctness.
- Prefer parallel branches when parts are independent.
"""


def plan_repair_prompt(raw_plan: str, errors: list[str]) -> str:
    return f"""The previous DAG YAML was invalid.

Errors:
{chr(10).join(f"- {error}" for error in errors)}

Return a corrected YAML plan only.

Previous output:
{raw_plan}
"""
