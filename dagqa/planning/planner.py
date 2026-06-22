from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI, AzureOpenAI

from dagqa.config import LLMConfig, PlannerConfig
from dagqa.llm.base import LanguageModel
from dagqa.llm.retry import retry_llm_call
from dagqa.planning.normalizer import normalize_plan_dependencies
from dagqa.planning.parser import PlanParseError, parse_plan
from dagqa.planning.prompts import PLANNER_SYSTEM, plan_repair_prompt, planner_user_prompt
from dagqa.planning.validator import validate_plan
from dagqa.schemas import DagNode, DagPlan, LLMRequest, Operation, PromptSpec, TaskType

logger = logging.getLogger(__name__)


class PlannerError(RuntimeError):
    pass


class Planner:
    def __init__(
        self,
        llm: LanguageModel,
        config: PlannerConfig,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.llm_config = llm_config

    async def plan(self, question: str) -> DagPlan:
        if self.llm_config and self.llm_config.provider == "azure_openai":
            return await self._plan_structured_azure(question)
        if self.llm_config and self.llm_config.provider == "cluster":
            return await self._plan_structured_cluster(question)

        request = LLMRequest(
            system=PLANNER_SYSTEM,
            prompt=planner_user_prompt(question, self.config.max_nodes, self.config.max_depth),
        )

        logger.debug("Starting planner request")
        response = await self.llm.complete(request)
        raw = response.text
        last_errors: list[str] = []

        for attempt in range(self.config.repair_rounds + 1):
            try:
                plan = normalize_plan_dependencies(parse_plan(raw))
            except PlanParseError as exc:
                last_errors = [str(exc)]
            else:
                validation = validate_plan(plan, self.config)
                if validation.valid:
                    return plan
                last_errors = validation.errors

            if attempt < self.config.repair_rounds:
                repair = await self.llm.complete(
                    LLMRequest(system=PLANNER_SYSTEM, prompt=plan_repair_prompt(raw, last_errors))
                )
                raw = repair.text

        raise PlannerError("Planner produced invalid DAG: " + "; ".join(last_errors))

    async def _plan_structured_azure(self, question: str) -> DagPlan:
        validation_errors: list[str] = []

        def _call(errors: list[str]) -> dict[str, Any]:
            endpoint = _env_or_value(self.llm_config.api_base_env, self.llm_config.api_base)
            api_version = _env_or_value(
                self.llm_config.api_version_env, self.llm_config.api_version
            )
            api_key = os.getenv(self.llm_config.api_key_env or "")
            model = _env_or_value(self.llm_config.model_env, self.llm_config.model)
            if not endpoint or not api_version or not api_key or not model:
                raise PlannerError("Azure OpenAI planner config is incomplete.")

            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=model.removeprefix("azure/"),
                temperature=0.0,
                messages=[
                    {"role": "system", "content": STRUCTURED_PLANNER_SYSTEM},
                    {
                        "role": "user",
                        "content": structured_planner_prompt(
                            question,
                            self.config.max_nodes,
                            self.config.max_depth,
                            errors,
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dag_plan",
                        "strict": True,
                        "schema": STRUCTURED_PLAN_SCHEMA,
                    },
                },
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)

        for attempt in range(self.config.repair_rounds + 1):
            try:
                current_errors = list(validation_errors)
                data, _retry_count = await retry_llm_call(
                    lambda errors=current_errors: asyncio.to_thread(_call, errors),
                    self.llm_config,
                )
                plan = normalize_plan_dependencies(_structured_data_to_plan(data))
            except Exception as exc:
                raise PlannerError(f"Structured Azure planner failed: {exc}") from exc

            validation = validate_plan(plan, self.config)
            if validation.valid:
                return plan
            validation_errors = validation.errors
            if attempt >= self.config.repair_rounds:
                break

        raise PlannerError("Planner produced invalid DAG: " + "; ".join(validation_errors))

    async def _plan_structured_cluster(self, question: str) -> DagPlan:
        validation_errors: list[str] = []
        api_base = _env_or_value(self.llm_config.api_base_env, self.llm_config.api_base)
        api_key = os.getenv(self.llm_config.api_key_env or "")
        model = _env_or_value(self.llm_config.model_env, self.llm_config.model)
        if not api_base or not api_key or not model:
            raise PlannerError("Cluster planner config is incomplete.")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=self.llm_config.request_timeout_seconds,
            max_retries=0,
        )

        for attempt in range(self.config.repair_rounds + 1):
            try:
                current_errors = list(validation_errors)
                response, _retry_count = await retry_llm_call(
                    lambda errors=current_errors: client.chat.completions.create(
                        model=model,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": STRUCTURED_PLANNER_SYSTEM},
                            {
                                "role": "user",
                                "content": structured_planner_prompt(
                                    question,
                                    self.config.max_nodes,
                                    self.config.max_depth,
                                    errors,
                                ),
                            },
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "dag_plan",
                                "strict": True,
                                "schema": STRUCTURED_PLAN_SCHEMA,
                            },
                        },
                    ),
                    self.llm_config,
                )
                content = response.choices[0].message.content or "{}"
                plan = normalize_plan_dependencies(_structured_data_to_plan(json.loads(content)))
            except Exception as exc:
                raise PlannerError(f"Structured cluster planner failed: {exc}") from exc

            validation = validate_plan(plan, self.config)
            if validation.valid:
                return plan
            validation_errors = validation.errors
            if attempt >= self.config.repair_rounds:
                break

        raise PlannerError("Planner produced invalid DAG: " + "; ".join(validation_errors))


def _env_or_value(env_name: str | None, value: str | None) -> str | None:
    return (os.getenv(env_name) if env_name else None) or value


STRUCTURED_PLANNER_SYSTEM = """You create executable DAG plans for a QA agent.
Return only data matching the supplied JSON schema.
References must use exactly '<node_id>.<field>', for example 'q1.city'.
Do not invent nested references like 'q1.answer.city'.
Comparison and synthesis nodes must receive both the values being compared and
the labels/entities those values belong to. For example, do not compare only
year1/year2; also include university1/university2 so the final answer can name
the correct entity instead of returning a number.
For every node with dependencies, the executable prompt template must include
child values by using {dependencies}, placeholders from input_map such as
{left_value}, or direct child placeholders such as {q1.answer}.
Every node with dependencies must include input_map references that consume each
node listed in depends_on. Do not declare unused dependencies.
Never make a final comparison/synthesis node ask the original user question
without child outputs in the prompt.
Final comparison/synthesis nodes must include the original question in the prompt,
identify the requested answer type, and select an answer of that type from child
outputs or evidence spans. They must not return a bridge entity if the original
question asks for an attribute of that entity.
Prefer targeted lookup nodes that return only the facts needed to answer the
question. Do not create broad candidate-list or exhaustive enumeration nodes
when a more specific lookup can identify the required entity or relationship
directly.
For bridge questions, ask lookup nodes to find the bridge-specific entity or
attribute from the supplied evidence rather than enumerating all possible
candidates from world knowledge.
Preserve the resolved bridge entity across hops: after a node identifies a
person, work, event, organization, place, or series, later nodes must ask about
that exact value and ignore unrelated entities in distractor documents.
In questions phrased like "what region/place of the country where X is located
is Y", X usually resolves only the country or scope. The answer target is the
region/place containing Y, not the region/place containing X. Carry both
candidates if needed and make the final node choose the target entity named
after "is".
Preserve temporal boundary wording from the original question. Questions using
"before", "after", "later than", "earlier than", "since", or "until" usually
ask for a threshold or boundary value. Do not rewrite them into "latest",
"earliest", "current", or "stopped using" questions unless the original wording
explicitly asks for that endpoint.
Preserve quoted title wording. If the user asks who wrote a quoted work or
title, identify the writer of that quoted text; do not reinterpret the question
as asking who composed, performed, or created an entity mentioned inside the
title unless the original wording says so.
"""


def structured_planner_prompt(
    question: str,
    max_nodes: int,
    max_depth: int,
    previous_errors: list[str] | None = None,
) -> str:
    repair_instruction = ""
    if previous_errors:
        repair_instruction = (
            "\nThe previous structured DAG was invalid. Fix these errors before returning:\n"
            + "\n".join(f"- {error}" for error in previous_errors)
            + "\n"
        )

    return f"""Create a DAG plan for this question:
{question}
{repair_instruction}

Constraints:
- max_nodes: {max_nodes}
- max_depth: {max_depth}
- final_node must be the node that returns the final answer.
- Independent fact lookup nodes should run in parallel.
- Every input_map reference must point to a field in a dependency node's output_fields.
- Every node listed in depends_on must be consumed by at least one input_map reference.
- Every prompt user_template may use placeholders like {{q1.city}} only for dependencies.
- For every node with depends_on, prompt.user_template must include child values through
  {{dependencies}}, input_map placeholders such as {{left_city}}, or direct placeholders
  such as {{q1.city}}.
- If a node compares attributes of entities, its prompt and input_map must include the entity
  names as well as the comparable attributes.
- Lookup nodes that feed comparisons should use explicit output field names, for example
  composer and birth_year, mountain and elevation_meters, city and latitude.
- Prefer targeted lookup nodes over broad candidate-list nodes. For example, ask which cast
  member of a given film played James Bond; do not ask for all James Bond actors unless the
  user's question requires that list.
- Evidence-backed lookup nodes should return only facts directly needed by downstream nodes,
  not every related fact visible in the context.
- Bridge nodes must preserve the resolved bridge value in downstream questions and prompts;
  never let a later hop answer about a different entity that only appears in a distractor context.
- For "what region/place of the country where X is located is Y" questions, use X only to
  identify the country or scope. The final answer is the region/place containing Y. If both
  X-region and Y-region are looked up, the final node must select the Y-region.
- Preserve temporal boundary wording. For questions with "before", "after", "later than",
  "earlier than", "since", or "until", return the boundary/threshold requested by the original
  question instead of converting it to a latest/earliest endpoint.
- Preserve quoted titles as answer targets. For questions like who wrote a quoted work/title,
  keep the quoted text as the work to look up rather than changing the task to composition or
  authorship of another entity mentioned inside that title.
- Use array fields for list-valued outputs and object fields for structured outputs.
- The final_node must always include an output field named answer. It may include additional
  fields, but answer must contain the concise final response to the user's question.
- The final_node output field must answer the user's question directly, not return an
  intermediate value like a year, date, latitude, or count unless that is what was asked.
- Final synthesis must identify the answer type requested by the original question before
  choosing the answer. Do not return a bridge entity when the original question asks for an
  attribute of that entity. Examples of answer types include country, region, event/date,
  era/decade, language, organization, person, place, yes/no, and number.
- Final synthesis must preserve exact modifiers from evidence or dependency values. Do not
  collapse qualified answers to a bare head noun, and do not substitute a city for a region,
  a modern country for a historical country name, or a bare year for a named election/event
  when the question asks for that more specific surface.
- The final node prompt must mention the original question and explicitly compare the current
  candidate answer with the requested answer type before returning answer.
"""


STRUCTURED_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["question", "nodes", "final_node"],
    "properties": {
        "question": {"type": "string"},
        "final_node": {"type": "string"},
        "nodes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "label",
                    "task_type",
                    "operation",
                    "question",
                    "depends_on",
                    "prompt",
                    "input_map",
                    "child_output_policy",
                    "output_fields",
                ],
                "properties": {
                    "id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_-]*$"},
                    "label": {"type": "string"},
                    "task_type": {
                        "type": "string",
                        "enum": [item.value for item in TaskType],
                    },
                    "operation": {
                        "type": "string",
                        "enum": [item.value for item in Operation],
                    },
                    "question": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "prompt": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["system", "user_template"],
                        "properties": {
                            "system": {"type": "string"},
                            "user_template": {"type": "string"},
                        },
                    },
                    "input_map": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "reference"],
                            "properties": {
                                "name": {"type": "string"},
                                "reference": {
                                    "type": "string",
                                    "pattern": "^[A-Za-z][A-Za-z0-9_-]*\\.[A-Za-z_][A-Za-z0-9_]*$",
                                },
                            },
                        },
                    },
                    "child_output_policy": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                    "output_fields": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "type", "description"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
                                },
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "string",
                                        "number",
                                        "integer",
                                        "boolean",
                                        "array",
                                        "object",
                                    ],
                                },
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}


def _structured_data_to_plan(data: dict[str, Any]) -> DagPlan:
    nodes = []
    for raw_node in data["nodes"]:
        output_schema = _output_fields_to_schema(
            raw_node["output_fields"],
            is_final=raw_node["id"] == data["final_node"],
        )
        nodes.append(
            DagNode(
                id=raw_node["id"],
                label=raw_node["label"],
                task_type=raw_node["task_type"],
                operation=raw_node["operation"],
                question=raw_node["question"],
                depends_on=raw_node["depends_on"],
                prompt=PromptSpec(**raw_node["prompt"]),
                input_map={
                    item["name"]: item["reference"].replace(".answer.", ".", 1)
                    for item in raw_node["input_map"]
                },
                child_output_policy=raw_node["child_output_policy"],
                output_schema=output_schema,
            )
        )
    return DagPlan(question=data["question"], nodes=nodes, final_node=data["final_node"])


def _output_fields_to_schema(fields: list[dict[str, str]], *, is_final: bool) -> dict[str, Any]:
    properties = {
        field["name"]: {
            "type": field["type"],
            "description": field["description"],
        }
        for field in fields
    }
    required = [field["name"] for field in fields]
    if is_final and "answer" not in properties:
        properties = {
            "answer": {
                "type": "string",
                "description": "Concise final answer to the original user question.",
            },
            **properties,
        }
        required = ["answer"]
    elif is_final and "answer" in required:
        required = ["answer", *[field for field in required if field != "answer"]]
    if is_final:
        if "answer_type" not in properties:
            properties["answer_type"] = {
                "type": "string",
                "description": (
                    "Answer type requested by the original question, such as country, region, "
                    "event/date, era/decade, language, organization, person, place, yes/no, "
                    "or number."
                ),
            }
        if "answer_source_span" not in properties:
            properties["answer_source_span"] = {
                "type": "string",
                "description": (
                    "Shortest dependency or evidence span that directly supports the final answer."
                ),
            }
        for field in ("answer_type", "answer_source_span"):
            if field not in required:
                required.append(field)
    return {
        "type": "object",
        "required": required,
        "properties": properties,
    }
