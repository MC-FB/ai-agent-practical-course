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

For every node with depends_on:
- input_map must contain the child values the node needs.
- prompt.user_template must explicitly include those child values using {dependencies},
  input_map placeholders such as {left_date}, or direct placeholders such as {q1.answer}.
- The node must use child outputs to answer. Do not make a final comparison or synthesis
  node ask the original user question without the child values in its prompt.

For comparison questions, create parallel lookup branches that return both the entity label
and the attribute being compared, then create a comparison node that consumes those fields.
Example: return {"composer": "...", "birth_year": 1678}, not only {"answer": "..."} if
a later node needs the composer name and birth year separately.

Use array fields for list-valued facts. The final node must always return an answer field
with the concise final answer, even when it also returns supporting fields.
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
- Dependent nodes must be executable from their child outputs. Prepare prompt templates with
  placeholders for the values returned by child nodes.
"""


def plan_repair_prompt(raw_plan: str, errors: list[str]) -> str:
    return f"""The previous DAG YAML was invalid.

Errors:
{chr(10).join(f"- {error}" for error in errors)}

Return a corrected YAML plan only.

Previous output:
{raw_plan}
"""
