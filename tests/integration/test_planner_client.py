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


async def test_client_ask_passes_evidence_to_executor(app_config) -> None:
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
    documents = [
        EvidenceDocument(id="context-0", title="Birth dates", text="Ada was born in 1815.")
    ]

    run = await DagQaClient(app_config, llm).ask(
        "Which person was born earlier?",
        evidence_documents=documents,
    )

    assert run.nodes[0].supporting_evidence is not None
    assert run.nodes[0].supporting_evidence.documents == documents
