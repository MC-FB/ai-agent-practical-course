from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from openai import AzureOpenAI

from dagqa.config import LLMConfig
from dagqa.config import PlannerConfig
from dagqa.llm.base import LanguageModel
from dagqa.planning.parser import PlanParseError, parse_plan
from dagqa.planning.prompts import PLANNER_SYSTEM, plan_repair_prompt, planner_user_prompt
from dagqa.planning.validator import validate_plan
from dagqa.schemas import DagNode, DagPlan, LLMRequest, Operation, PromptSpec, TaskType


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

        request = LLMRequest(
            system=PLANNER_SYSTEM,
            prompt=planner_user_prompt(question, self.config.max_nodes, self.config.max_depth),
        )
        response = await self.llm.complete(request)
        raw = response.text
        last_errors: list[str] = []

        for attempt in range(self.config.repair_rounds + 1):
            try:
                plan = parse_plan(raw)
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
        started = time.perf_counter()

        def _call() -> dict[str, Any]:
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
            )
            response = client.chat.completions.create(
                model=model.removeprefix("azure/"),
                temperature=0.0,
                messages=[
                    {"role": "system", "content": STRUCTURED_PLANNER_SYSTEM},
                    {
                        "role": "user",
                        "content": structured_planner_prompt(
                            question, self.config.max_nodes, self.config.max_depth
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

        try:
            data = await asyncio.to_thread(_call)
            plan = _structured_data_to_plan(data)
        except Exception as exc:
            raise PlannerError(f"Structured Azure planner failed: {exc}") from exc

        validation = validate_plan(plan, self.config)
        if not validation.valid:
            raise PlannerError("Planner produced invalid DAG: " + "; ".join(validation.errors))
        return plan


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
"""


def structured_planner_prompt(question: str, max_nodes: int, max_depth: int) -> str:
    return f"""Create a DAG plan for this question:
{question}

Constraints:
- max_nodes: {max_nodes}
- max_depth: {max_depth}
- final_node must be the node that returns the final answer.
- Independent fact lookup nodes should run in parallel.
- Every input_map reference must point to a field in a dependency node's output_fields.
- Every prompt user_template may use placeholders like {{q1.city}} only for dependencies.
- If a node compares attributes of entities, its prompt and input_map must include the entity
  names as well as the comparable attributes.
- The final_node output field must answer the user's question directly, not return an
  intermediate value like a year, date, latitude, or count unless that is what was asked.
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
                                    "enum": ["string", "number", "integer", "boolean"],
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
        output_schema = {
            "type": "object",
            "properties": {
                field["name"]: {
                    "type": field["type"],
                    "description": field["description"],
                }
                for field in raw_node["output_fields"]
            },
        }
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
