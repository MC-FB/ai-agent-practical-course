from __future__ import annotations

import json

from dagqa.graph.executor import DagExecutor
from dagqa.planning.parser import parse_plan
from dagqa.schemas import EvidenceDocument, NodeStatus
from tests.conftest import StubLLM
from tests.fixtures import PARALLEL_PLAN


def _bridge_birth_response(
    *,
    answer: str,
    title: str,
    document_id: str,
    fact: str,
    person: str,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "bridge_answer": answer,
            "bridge_reasoning": f"{person} is the requested person and the cited span states it.",
            "bridge_source_span": fact,
            "constraint_status": "satisfied",
            "bridge_candidates": [
                {
                    "candidate": answer,
                    "status": "selected",
                    "supporting_span": fact,
                    "constraint_match": f"{person} birth date",
                }
            ],
            "_evidence_citations": [
                {
                    "document_id": document_id,
                    "title": title,
                    "sentence_indices": [0],
                    "fact": fact,
                }
            ],
        }
    )


async def test_executor_runs_parallel_dependencies_then_parent(app_config) -> None:
    expected_call_count = 3
    llm = StubLLM(
        [
            _bridge_birth_response(
                answer="10 December 1815",
                title="Ada Lovelace",
                document_id="context-0",
                fact="Ada was born in 1815.",
                person="Ada Lovelace",
            ),
            _bridge_birth_response(
                answer="23 June 1912",
                title="Alan Turing",
                document_id="context-1",
                fact="Turing was born in 1912.",
                person="Alan Turing",
            ),
            (
                '{"answer": "Ada Lovelace", "reasoning": "1815 is earlier than 1912.", '
                '"answer_type": "person", "answer_source_span": "Ada Lovelace: 10 December 1815", '
                '"_evidence_citations": [{"document_id": "context-0", "title": "Ada Lovelace", '
                '"sentence_indices": [0], "fact": "Ada was born in 1815."}]}'
            ),
        ]
    )
    run = await DagExecutor(llm, app_config).execute(parse_plan(PARALLEL_PLAN))

    assert run.status == NodeStatus.succeeded
    assert run.waves[0].node_ids == ["q1", "q2"]
    assert run.final_answer == {
        "answer": "Ada Lovelace",
        "reasoning": "1815 is earlier than 1912.",
        "answer_type": "person",
        "answer_source_span": "Ada Lovelace: 10 December 1815",
    }
    assert len(llm.requests) == expected_call_count
    assert "10 December 1815" in llm.requests[2].prompt


async def test_executor_injects_and_persists_evidence_for_factual_nodes(app_config) -> None:
    llm = StubLLM(
        [
            _bridge_birth_response(
                answer="10 December 1815",
                title="Ada Lovelace",
                document_id="context-0",
                fact="Ada was born in 1815.",
                person="Ada Lovelace",
            ),
            _bridge_birth_response(
                answer="23 June 1912",
                title="Alan Turing",
                document_id="context-1",
                fact="Turing was born in 1912.",
                person="Alan Turing",
            ),
            (
                '{"answer": "Ada Lovelace", "reasoning": "1815 is earlier than 1912.", '
                '"answer_type": "person", "answer_source_span": "Ada Lovelace: 10 December 1815", '
                '"_evidence_citations": [{"document_id": "context-0", "title": "Ada Lovelace", '
                '"sentence_indices": [0], "fact": "Ada was born in 1815."}]}'
            ),
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
    assert run.nodes[2].supporting_evidence is not None
    assert [document.id for document in run.nodes[2].supporting_evidence.documents] == [
        "context-0",
        "context-1",
    ]
    assert run.nodes[0].evidence_citations[0].document_id == "context-0"
    assert "_evidence_citations" not in run.nodes[0].returned_value
    assert "_evidence_citations" not in run.final_answer
    assert "Ada was born in 1815." in llm.requests[0].prompt
    assert "Turing was born in 1912." in llm.requests[1].prompt
    assert "Supporting evidence documents:" in llm.requests[2].prompt
