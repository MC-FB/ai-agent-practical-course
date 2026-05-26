from __future__ import annotations

from dagqa.client import DagQaClient
from tests.conftest import StubLLM
from tests.fixtures import PARALLEL_PLAN


async def test_client_ask_uses_planner_then_executor(app_config) -> None:
    expected_node_count = 3
    llm = StubLLM(
        [
            PARALLEL_PLAN,
            '{"answer": "10 December 1815"}',
            '{"answer": "23 June 1912"}',
            '{"answer": "Ada Lovelace", "reasoning": "1815 is earlier than 1912."}',
        ]
    )

    run = await DagQaClient(app_config, llm).ask("Which person was born earlier?")

    assert run.final_answer["answer"] == "Ada Lovelace"
    assert len(run.nodes) == expected_node_count
