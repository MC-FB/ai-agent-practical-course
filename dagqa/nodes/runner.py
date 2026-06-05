from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from dagqa.config import ExecutionConfig, LLMConfig
from dagqa.evidence import select_evidence
from dagqa.graph.substitution import (
    MissingDependencyValue,
    resolve_input_map,
    resolve_question,
)
from dagqa.llm.base import LanguageModel
from dagqa.nodes.output_validation import parse_node_output, validate_node_output
from dagqa.nodes.prompts import node_output_schema, render_node_prompt, render_repair_prompt
from dagqa.schemas import (
    DagNode,
    EvidenceCitation,
    EvidenceDocument,
    EvidenceSelection,
    LLMRequest,
    NodeStatus,
    NodeTrace,
    ValidationResult,
)


class NodeRunner:
    def __init__(
        self,
        llm: LanguageModel,
        execution: ExecutionConfig,
        llm_config: LLMConfig,
    ) -> None:
        self.llm = llm
        self.execution = execution
        self.llm_config = llm_config

    async def run(
        self,
        node: DagNode,
        outputs: dict[str, dict[str, Any]],
        evidence_documents: list[EvidenceDocument] | None = None,
    ) -> NodeTrace:
        started_perf = time.perf_counter()
        started_at = datetime.now(UTC).isoformat()
        trace = NodeTrace(
            node_id=node.id,
            label=node.label,
            task_type=node.task_type,
            operation=node.operation,
            status=NodeStatus.running,
            started_at=started_at,
        )
        try:
            dependency_values = resolve_input_map(node.input_map, outputs)
            resolved_question = resolve_question(node.question, outputs)
            supporting_evidence = select_evidence(node, evidence_documents)
            prompt = render_node_prompt(
                node,
                resolved_question,
                dependency_values,
                outputs,
                supporting_evidence,
            )
            trace.dependency_values = dependency_values
            trace.resolved_question = resolved_question
            trace.supporting_evidence = supporting_evidence
            trace.rendered_prompt = prompt
            raw_response, retry_count = await self._call_llm(node.prompt.system, prompt)
            trace.raw_response = raw_response
            trace.llm_retry_count += retry_count
            parsed, validation = await self._parse_validate_repair(
                node,
                raw_response,
                supporting_evidence,
                trace,
            )
            trace.parsed_output = parsed
            if parsed is not None:
                trace.evidence_citations = [
                    EvidenceCitation.model_validate(citation)
                    for citation in parsed.get("_evidence_citations", [])
                ]
                trace.returned_value = {
                    key: value for key, value in parsed.items() if key != "_evidence_citations"
                }
            trace.validation = validation
            trace.repair_attempts = 0 if validation.valid else self.execution.node_repair_rounds
            trace.status = NodeStatus.succeeded if validation.valid else NodeStatus.failed
        except (MissingDependencyValue, ValueError) as exc:
            trace.status = NodeStatus.failed
            trace.validation = ValidationResult(valid=False, errors=[str(exc)])
            trace.error = str(exc)
        finally:
            trace.finished_at = datetime.now(UTC).isoformat()
            trace.duration_ms = (time.perf_counter() - started_perf) * 1000
        return trace

    async def _parse_validate_repair(
        self,
        node: DagNode,
        raw_response: str,
        supporting_evidence: EvidenceSelection | None,
        trace: NodeTrace,
    ) -> tuple[dict[str, Any] | None, ValidationResult]:
        current_raw = raw_response
        last_output: dict[str, Any] | None = None
        last_validation = ValidationResult(valid=False, errors=["No validation attempted."])
        schema = node_output_schema(node, supporting_evidence)

        for attempt in range(self.execution.node_repair_rounds + 1):
            try:
                last_output = parse_node_output(current_raw)
            except Exception as exc:
                last_validation = ValidationResult(valid=False, errors=[str(exc)])
            else:
                last_validation = validate_node_output(last_output, schema)
                if last_validation.valid:
                    return last_output, last_validation

            if attempt < self.execution.node_repair_rounds:
                current_raw, retry_count = await self._call_llm(
                    "Repair malformed node output.",
                    render_repair_prompt(current_raw, schema, last_validation.errors),
                )
                trace.llm_retry_count += retry_count

        return last_output, last_validation

    async def _call_llm(self, system: str, prompt: str) -> tuple[str, int]:
        response = await self.llm.complete(
            LLMRequest(system=system, prompt=prompt, temperature=self.llm_config.temperature)
        )
        retry_count = response.metadata.get("retry_count", 0)
        return response.text, int(retry_count) if isinstance(retry_count, int) else 0
