from __future__ import annotations

import json

from dagqa.client import DagQaClient
from dagqa.schemas import EvidenceDocument
from tests.conftest import StubLLM
from tests.fixtures import PARALLEL_PLAN


def _bridge_birth_response(
    *,
    answer: str,
    fact: str,
    person: str,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "reasoning": f"{person} birth date found in evidence: {fact}",
            "_evidence_citations": [
                {
                    "document_id": "context-0",
                    "title": "Birth dates",
                    "sentence_indices": [0],
                    "fact": fact,
                }
            ],
        }
    )


async def test_client_ask_uses_planner_then_executor(app_config) -> None:
    expected_node_count = 3
    llm = StubLLM(
        [
            PARALLEL_PLAN,
            _bridge_birth_response(
                answer="10 December 1815",
                fact="Ada was born in 1815.",
                person="Ada Lovelace",
            ),
            _bridge_birth_response(
                answer="23 June 1912",
                fact="Turing was born in 1912.",
                person="Alan Turing",
            ),
            (
                '{"answer": "Ada Lovelace", "reasoning": "1815 is earlier than 1912.", '
                '"answer_type": "person", "answer_source_span": "Ada Lovelace: 10 December 1815", '
                '"_evidence_citations": [{"document_id": "context-0", "title": "Birth dates", '
                '"sentence_indices": [0], "fact": "Ada was born in 1815."}]}'
            ),
        ]
    )

    run = await DagQaClient(app_config, llm).ask("Which person was born earlier?")

    assert run.final_answer["answer"] == "Ada Lovelace"
    assert len(run.nodes) == expected_node_count


async def test_client_ask_uses_least_to_most_with_evidence(app_config) -> None:
    llm = StubLLM(
        [
            PARALLEL_PLAN,
            # LtM path response
            '{"steps": "Step 1: Ada 1815. Step 2: Turing 1912.", "answer": "Ada Lovelace"}',
            # DAG path: q1
            _bridge_birth_response(
                answer="10 December 1815",
                fact="Ada was born in 1815.",
                person="Ada Lovelace",
            ),
            # DAG path: q2
            _bridge_birth_response(
                answer="23 June 1912",
                fact="Ada was born in 1815.",
                person="Alan Turing",
            ),
            # DAG path: q3
            '{"answer": "Ada Lovelace", "answer_type": "person", "answer_source_span": "Ada"}',
        ]
    )
    documents = [
        EvidenceDocument(id="context-0", title="Birth dates", text="Ada was born in 1815.")
    ]

    run = await DagQaClient(app_config, llm).ask(
        "Which person was born earlier?",
        evidence_documents=documents,
    )

    assert run.final_answer["answer"] == "Ada Lovelace"
