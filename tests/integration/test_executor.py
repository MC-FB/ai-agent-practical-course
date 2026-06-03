from __future__ import annotations

from dagqa.graph.executor import DagExecutor
from dagqa.planning.parser import parse_plan
from dagqa.schemas import EvidenceDocument, NodeStatus
from tests.conftest import StubLLM
from tests.fixtures import PARALLEL_PLAN


async def test_executor_runs_parallel_dependencies_then_parent(app_config) -> None:
    expected_call_count = 3
    llm = StubLLM(
        [
            (
                '{"answer": "10 December 1815", "_evidence_citations": '
                '[{"document_id": "context-0", "title": "Ada Lovelace", '
                '"sentence_indices": [0], "fact": "Ada was born in 1815."}]}'
            ),
            (
                '{"answer": "23 June 1912", "_evidence_citations": '
                '[{"document_id": "context-1", "title": "Alan Turing", '
                '"sentence_indices": [0], "fact": "Turing was born in 1912."}]}'
            ),
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


async def test_executor_injects_and_persists_evidence_for_factual_nodes(app_config) -> None:
    llm = StubLLM(
        [
            (
                '{"answer": "10 December 1815", "_evidence_citations": '
                '[{"document_id": "context-0", "title": "Ada Lovelace", '
                '"sentence_indices": [0], "fact": "Ada was born in 1815."}]}'
            ),
            (
                '{"answer": "23 June 1912", "_evidence_citations": '
                '[{"document_id": "context-1", "title": "Alan Turing", '
                '"sentence_indices": [0], "fact": "Turing was born in 1912."}]}'
            ),
            '{"answer": "Ada Lovelace", "reasoning": "1815 is earlier than 1912."}',
        ]
    )
    documents = [
        EvidenceDocument(id="context-0", title="Ada Lovelace", text="Ada was born in 1815."),
        EvidenceDocument(id="context-1", title="Alan Turing", text="Turing was born in 1912."),
    ]

    run = await DagExecutor(llm, app_config).execute(
        parse_plan(PARALLEL_PLAN),
        evidence_documents=documents,
    )

    assert run.nodes[0].supporting_evidence is not None
    assert run.nodes[0].supporting_evidence.documents == documents
    assert run.nodes[1].supporting_evidence is not None
    assert run.nodes[2].supporting_evidence is None
    assert run.nodes[0].evidence_citations[0].document_id == "context-0"
    assert "_evidence_citations" not in run.nodes[0].returned_value
    assert "Ada was born in 1815." in llm.requests[0].prompt
    assert "Turing was born in 1912." in llm.requests[1].prompt
    assert "Supporting evidence documents:" not in llm.requests[2].prompt
