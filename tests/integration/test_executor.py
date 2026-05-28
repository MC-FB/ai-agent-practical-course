from __future__ import annotations

from dagqa.graph.executor import DagExecutor
from dagqa.planning.parser import parse_plan
from dagqa.schemas import NodeStatus
from tests.conftest import StubLLM
from tests.fixtures import PARALLEL_PLAN


async def test_executor_runs_parallel_dependencies_then_parent(app_config) -> None:
    expected_call_count = 3
    llm = StubLLM(
        [
            '{"answer": "10 December 1815"}',
            '{"answer": "23 June 1912"}',
            '{"answer": "Ada Lovelace", "reasoning": "1815 is earlier than 1912."}',
        ]
    )
    run = await DagExecutor(llm, app_config).execute(parse_plan(PARALLEL_PLAN))

    assert run.status == NodeStatus.succeeded
    assert run.waves[0].node_ids == ["q1", "q2"]
    assert run.final_answer == {
        "answer": "Ada Lovelace",
        "reasoning": "1815 is earlier than 1912.",
    }
    assert len(llm.requests) == expected_call_count
    assert "10 December 1815" in llm.requests[2].prompt
