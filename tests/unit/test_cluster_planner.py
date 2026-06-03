from __future__ import annotations

import json
from types import SimpleNamespace

from dagqa.config import LLMConfig, PlannerConfig
from dagqa.planning import planner
from dagqa.planning.parser import parse_plan
from tests.conftest import StubLLM
from tests.fixtures import PARALLEL_PLAN


async def test_cluster_planner_uses_structured_json_schema(monkeypatch) -> None:
    captured: dict = {}
    plan = parse_plan(PARALLEL_PLAN)
    plan_json = json.dumps(
        {
            "question": plan.question,
            "final_node": plan.final_node,
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "task_type": node.task_type,
                    "operation": node.operation,
                    "question": node.question,
                    "depends_on": node.depends_on,
                    "prompt": node.prompt.model_dump(mode="json"),
                    "input_map": [
                        {"name": name, "reference": reference}
                        for name, reference in node.input_map.items()
                    ],
                    "child_output_policy": node.child_output_policy,
                    "output_fields": [
                        {
                            "name": name,
                            "type": field["type"],
                            "description": field.get("description", name),
                        }
                        for name, field in node.output_schema["properties"].items()
                    ],
                }
                for node in plan.nodes
            ],
        }
    )

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=plan_json))]
            )

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=Completions())

    llm_config = LLMConfig(
        provider="cluster",
        model="google/gemma-4-31B-it",
        api_key_env="CLUSTER_API_KEY",
        api_base="http://cluster/inference",
    )
    monkeypatch.setenv("CLUSTER_API_KEY", "secret")
    monkeypatch.setattr(planner, "AsyncOpenAI", Client)

    result = await planner.Planner(StubLLM([]), PlannerConfig(), llm_config).plan(
        "Which person was born earlier?"
    )

    assert result.final_node == "q3"
    assert captured["client"] == {"api_key": "secret", "base_url": "http://cluster/inference"}
    assert captured["response_format"]["type"] == "json_schema"
