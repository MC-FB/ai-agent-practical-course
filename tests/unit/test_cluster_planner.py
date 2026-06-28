from __future__ import annotations

import json
from types import SimpleNamespace

from dagqa.config import LLMConfig, PlannerConfig
from dagqa.planning import planner
from dagqa.planning.parser import parse_plan
from tests.conftest import StubLLM
from tests.fixtures import PARALLEL_PLAN

EXPECTED_RETRY_CALL_COUNT = 2


def test_structured_planner_prompt_requires_multihop_bridge_chains() -> None:
    prompt = planner.structured_planner_prompt(
        "What was the language from which the last name Sylvester originates during the era "
        "of the person crowned new Roman emperor in 800 A.D. later known as?",
        max_nodes=10,
        max_depth=6,
    )

    assert "create a multi-node bridge" in prompt
    assert "one-node plan is only acceptable" in prompt
    assert "combine final answer selection into the" in prompt


def test_structured_planner_prompt_requires_bridge_reasoning_contract() -> None:
    prompt = planner.structured_planner_prompt(
        "The leader visiting where Steven Spielberg's grandparents are from met with whom "
        "on November 22?",
        max_nodes=10,
        max_depth=6,
    )

    assert "bridge_answer" in prompt
    assert "bridge_reasoning" in prompt
    assert "constraint_status" in prompt
    assert "ambiguous" in prompt


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

    result = await planner.Planner(
        StubLLM([]), PlannerConfig(planner_mode="structured"), llm_config
    ).plan("Which person was born earlier?")

    assert result.final_node == "q3"
    assert captured["client"] == {
        "api_key": "secret",
        "base_url": "http://cluster/inference",
        "timeout": 120.0,
        "max_retries": 0,
    }
    assert captured["response_format"]["type"] == "json_schema"


async def test_cluster_planner_retries_structured_request(monkeypatch) -> None:
    calls = 0
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

    class RetryableClusterError(Exception):
        status_code = 500

    class Completions:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RetryableClusterError("temporary cluster failure")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=plan_json))]
            )

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    llm_config = LLMConfig(
        provider="cluster",
        model="google/gemma-4-31B-it",
        api_key_env="CLUSTER_API_KEY",
        api_base="http://cluster/inference",
        max_retries=1,
        retry_initial_delay_seconds=0.001,
    )
    monkeypatch.setenv("CLUSTER_API_KEY", "secret")
    monkeypatch.setattr(planner, "AsyncOpenAI", Client)

    result = await planner.Planner(
        StubLLM([]), PlannerConfig(planner_mode="structured"), llm_config
    ).plan("Which person was born earlier?")

    assert result.final_node == "q3"
    assert calls == EXPECTED_RETRY_CALL_COUNT
