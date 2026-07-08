from __future__ import annotations

from dagqa.graph.executor import DagExecutor
from dagqa.planning.parser import parse_plan
from dagqa.schemas import ChatMessage, EvidenceDocument, NodeStatus
from tests.conftest import StubLLM
from tests.fixtures import PARALLEL_PLAN

FINAL_JSON = (
    '{"answer": "Ada Lovelace", '
    '"_evidence_citations": [{"document_id": "context-0", "title": "Ada Lovelace", '
    '"sentence_indices": [0], "fact": "Ada was born on 10 December 1815."}]}'
)


def _documents() -> list[EvidenceDocument]:
    return [
        EvidenceDocument(
            id="context-0",
            title="Ada Lovelace",
            text="Ada was born on 10 December 1815.",
        ),
        EvidenceDocument(
            id="context-1",
            title="Alan Turing",
            text="Turing was born on 23 June 1912.",
        ),
    ]


async def test_conversation_asks_one_sub_question_per_turn_with_history(app_config) -> None:
    expected_turns = 3
    llm = StubLLM(["10 December 1815", "23 June 1912", FINAL_JSON])

    run = await DagExecutor(llm, app_config).execute_least_to_most_conversation(
        parse_plan(PARALLEL_PLAN), _documents()
    )

    assert run.status == NodeStatus.succeeded
    assert run.final_answer == {"answer": "Ada Lovelace"}
    assert [trace.node_id for trace in run.nodes] == ["q1", "q2", "q3"]
    assert [wave.node_ids for wave in run.waves] == [["q1"], ["q2"], ["q3"]]
    assert len(llm.requests) == expected_turns
    assert llm.requests[0].history == []
    assert llm.requests[1].history == [
        ChatMessage(role="user", content=llm.requests[0].prompt),
        ChatMessage(role="assistant", content="10 December 1815"),
    ]
    assert llm.requests[2].history == [
        ChatMessage(role="user", content=llm.requests[0].prompt),
        ChatMessage(role="assistant", content="10 December 1815"),
        ChatMessage(role="user", content=llm.requests[1].prompt),
        ChatMessage(role="assistant", content="23 June 1912"),
    ]
    assert "Supporting evidence documents:" in llm.requests[0].system
    assert llm.requests[0].system == llm.requests[2].system
    assert "When was Ada Lovelace born?" in llm.requests[0].prompt
    assert "answer the original question" in llm.requests[2].prompt
    assert run.nodes[0].returned_value == {"answer": "10 December 1815"}
    assert run.nodes[2].evidence_citations[0].document_id == "context-0"


async def test_conversation_json_turn_format_parses_intermediate_answers(app_config) -> None:
    app_config.execution.ltm_conversation_turn_format = "json"
    llm = StubLLM(['{"answer": "10 December 1815"}', "23 June 1912", FINAL_JSON])

    run = await DagExecutor(llm, app_config).execute_least_to_most_conversation(
        parse_plan(PARALLEL_PLAN), _documents()
    )

    assert run.status == NodeStatus.succeeded
    assert 'Return JSON only: {"answer": "short answer"}' in llm.requests[0].prompt
    assert run.nodes[0].parsed_output == {"answer": "10 December 1815"}
    assert run.nodes[0].returned_value == {"answer": "10 December 1815"}
    # A non-JSON reply in json mode falls back to the raw text.
    assert run.nodes[1].returned_value == {"answer": "23 June 1912"}
    assert run.final_answer == {"answer": "Ada Lovelace"}


async def test_conversation_stops_after_failed_turn(app_config) -> None:
    llm = StubLLM(["10 December 1815"])

    run = await DagExecutor(llm, app_config).execute_least_to_most_conversation(
        parse_plan(PARALLEL_PLAN), _documents()
    )

    assert run.status == NodeStatus.failed
    assert run.final_answer is None
    assert [trace.node_id for trace in run.nodes] == ["q1", "q2"]
    assert run.nodes[0].status == NodeStatus.succeeded
    assert run.nodes[1].status == NodeStatus.failed
    assert run.nodes[1].error
