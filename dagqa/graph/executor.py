from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from dagqa.config import AppConfig
from dagqa.graph.scheduler import Scheduler
from dagqa.llm.base import LanguageModel
from dagqa.nodes.output_validation import parse_node_output
from dagqa.nodes.prompts import render_evidence_section
from dagqa.nodes.runner import NodeRunner
from dagqa.planning.normalizer import normalize_plan_dependencies
from dagqa.planning.validator import validate_plan
from dagqa.schemas import (
    DagPlan,
    EvidenceCitation,
    EvidenceDocument,
    EvidenceSelection,
    LLMRequest,
    NodeStatus,
    NodeTrace,
    Operation,
    RunTrace,
    SchedulerWave,
    TaskType,
)

logger = logging.getLogger(__name__)


def _parse_citations(parsed: dict) -> list[EvidenceCitation]:
    """Extract evidence citations from a parsed LLM response."""
    raw = parsed.get("_evidence_citations", [])
    if not isinstance(raw, list):
        return []
    citations = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            citations.append(EvidenceCitation.model_validate(item))
        except Exception:
            continue
    return citations


class ExecutionError(RuntimeError):
    pass


class DagExecutor:
    def __init__(self, llm: LanguageModel, config: AppConfig) -> None:
        self.llm = llm
        self.config = config
        self.scheduler = Scheduler()
        self.runner = NodeRunner(llm, config.execution, config.llm)

    async def execute(
        self,
        plan: DagPlan,
        evidence_documents: list[EvidenceDocument] | None = None,
    ) -> RunTrace:
        plan = normalize_plan_dependencies(plan)
        validation = validate_plan(plan, self.config.planner)
        if not validation.valid:
            raise ExecutionError("Invalid DAG: " + "; ".join(validation.errors))

        started = time.perf_counter()
        run_id = str(uuid4())
        waves = self.scheduler.build_waves(plan)
        by_id = {node.id: node for node in plan.nodes}
        outputs: dict[str, dict[str, Any]] = {}
        traces: list[NodeTrace] = []
        semaphore = asyncio.Semaphore(self.config.execution.max_parallel_nodes)

        async def run_node(node_id: str) -> NodeTrace:
            async with semaphore:
                return await asyncio.wait_for(
                    self.runner.run(
                        by_id[node_id],
                        outputs,
                        evidence_documents,
                        original_question=plan.question,
                        is_final=(node_id == plan.final_node),
                    ),
                    timeout=self.config.execution.node_timeout_seconds,
                )

        status = NodeStatus.succeeded
        for wave in waves:
            wave_traces = await asyncio.gather(
                *(run_node(node_id) for node_id in wave.node_ids),
                return_exceptions=True,
            )
            for node_id, result in zip(wave.node_ids, wave_traces, strict=True):
                if isinstance(result, Exception):
                    trace = self._exception_trace(by_id[node_id], result)
                else:
                    trace = result
                traces.append(trace)
                if trace.status == NodeStatus.succeeded and trace.returned_value is not None:
                    outputs[trace.node_id] = self._internal_output(trace)
                else:
                    status = NodeStatus.failed
                    if self.config.execution.fail_fast:
                        return self._run_trace(run_id, plan, waves, traces, None, status, started)

        final_trace = next((trace for trace in traces if trace.node_id == plan.final_node), None)
        final_answer = final_trace.returned_value if final_trace else None
        if final_answer is None:
            status = NodeStatus.failed
        return self._run_trace(run_id, plan, waves, traces, final_answer, status, started)

    def _internal_output(self, trace: NodeTrace) -> dict[str, Any]:
        output = dict(trace.returned_value or {})
        if trace.evidence_citations:
            output["_evidence_citations"] = [
                citation.model_dump(mode="json") for citation in trace.evidence_citations
            ]
        return output

    def _run_trace(
        self,
        run_id: str,
        plan: DagPlan,
        waves: list[SchedulerWave],
        traces: list[NodeTrace],
        final_answer: dict[str, Any] | None,
        status: NodeStatus,
        started: float,
    ) -> RunTrace:
        return RunTrace(
            run_id=run_id,
            question=plan.question,
            plan=plan,
            waves=waves,
            nodes=traces,
            final_answer=final_answer,
            status=status,
            total_duration_ms=(time.perf_counter() - started) * 1000,
            config=self.config.model_dump(mode="json"),
        )

    async def execute_least_to_most_only(
        self,
        plan: DagPlan,
        evidence_documents: list[EvidenceDocument],
    ) -> RunTrace:
        """Pure Least-to-Most execution without DAG fallback."""
        plan = normalize_plan_dependencies(plan)
        return await self._run_least_to_most(plan, evidence_documents)

    async def execute_least_to_most(
        self,
        plan: DagPlan,
        evidence_documents: list[EvidenceDocument],
    ) -> RunTrace:
        """Least-to-Most + Self-Consistency execution.

        Implements two research-grounded techniques:
        1. Least-to-Most (Zhou et al., 2022): compile the plan's decomposed
           sub-questions into a single structured prompt with all evidence,
           so the LLM answers step-by-step without error propagation.
        2. Self-Consistency (Wang et al., 2022): also run the standard
           node-by-node DAG execution; keep the answer that appears more
           grounded (longer exact-match overlap with evidence text).
        """
        plan = normalize_plan_dependencies(plan)
        started = time.perf_counter()
        run_id = str(uuid4())

        # Run both paths in parallel
        ltm_task = asyncio.create_task(self._run_least_to_most(plan, evidence_documents))
        dag_task = asyncio.create_task(self._run_dag_nodes(plan, evidence_documents))
        ltm_result, dag_result = await asyncio.gather(
            ltm_task,
            dag_task,
            return_exceptions=True,
        )

        # Pick the best answer using evidence grounding
        evidence_text = "\n".join(f"{d.title}\n{d.text}" for d in evidence_documents).casefold()

        ltm_answer = self._extract_answer_str(ltm_result)
        dag_answer = self._extract_answer_str(dag_result)

        ltm_score = self._grounding_score(ltm_answer, evidence_text)
        dag_score = self._grounding_score(dag_answer, evidence_text)

        logger.debug(
            "Self-Consistency: ltm=%r (%.2f) dag=%r (%.2f)",
            ltm_answer,
            ltm_score,
            dag_answer,
            dag_score,
        )

        if isinstance(dag_result, Exception) or dag_answer == "":
            chosen = ltm_result
        elif isinstance(ltm_result, Exception) or ltm_answer == "" or dag_score > ltm_score:
            chosen = dag_result
        else:
            chosen = ltm_result

        if isinstance(chosen, Exception):
            raise chosen
        # Merge: use the chosen RunTrace but adjust timing
        chosen.total_duration_ms = (time.perf_counter() - started) * 1000
        chosen.run_id = run_id
        return chosen

    async def _run_least_to_most(
        self,
        plan: DagPlan,
        evidence_documents: list[EvidenceDocument],
    ) -> RunTrace:
        started = time.perf_counter()
        run_id = str(uuid4())

        # Build ordered sub-questions from topological order
        waves = self.scheduler.build_waves(plan)
        ordered_questions: list[str] = []
        for wave in waves:
            for node_id in wave.node_ids:
                node = next(n for n in plan.nodes if n.id == node_id)
                if node.id != plan.final_node:
                    ordered_questions.append(node.question)

        evidence = EvidenceSelection(
            strategy="all_documents",
            total_available=len(evidence_documents),
            documents=evidence_documents,
        )
        evidence_section = render_evidence_section(evidence, require_citations=True)

        sub_q_block = "\n".join(f"  Step {i + 1}: {q}" for i, q in enumerate(ordered_questions))

        prompt = (
            f"Question: {plan.question}\n\n"
            f"Solve step by step. For each step, find the answer in the "
            f"evidence before moving to the next step.\n\n"
            f"{sub_q_block}\n"
            f"  Final step: Using the above answers, answer the original question.\n"
            f"{evidence_section}\n"
            f"Instructions:\n"
            f"- For each step, find the specific entity, value, or fact in the evidence.\n"
            f"- Copy exact names, numbers, and dates from the evidence.\n"
            f"- The final answer must be a short, specific value (a name, number, "
            f"place, date, or entity) — not a sentence or explanation.\n"
            f"- If the evidence does not contain the answer, still give your best "
            f"short answer based on what is available.\n"
            f"- Never return 'not determinable' or 'unsupported' — always return "
            f"a concrete answer.\n\n"
            f'Return JSON: {{"steps": "Step 1: ... Step 2: ...", '
            f'"answer": "short final answer", '
            f'"_evidence_citations": '
            f'[{{"document_id": "...", "title": "...", '
            f'"sentence_indices": [0], "fact": "..."}}]}}\n'
            f"Return JSON only."
        )

        trace = NodeTrace(
            node_id="ltm",
            label="Least-to-Most execution",
            task_type=TaskType.synthesis,
            operation=Operation.synthesize,
            status=NodeStatus.running,
            supporting_evidence=evidence,
        )
        trace.rendered_prompt = prompt

        try:
            response = await self.llm.complete(
                LLMRequest(
                    system=(
                        "Answer the question step by step using only the evidence provided. "
                        "Copy exact values from evidence documents. Return JSON."
                    ),
                    prompt=prompt,
                    temperature=0.0,
                )
            )
            trace.raw_response = response.text
            trace.llm_retry_count = int(response.metadata.get("retry_count", 0))
            parsed = parse_node_output(response.text)
            trace.parsed_output = parsed
            raw_answer = parsed.get("answer") or parsed.get("final_answer") or ""
            # Handle case where LLM returns a dict/list as the answer
            if isinstance(raw_answer, dict):
                raw_answer = (
                    raw_answer.get("answer") or raw_answer.get("final_answer") or str(raw_answer)
                )
            answer = str(raw_answer).strip()
            if answer:
                trace.returned_value = {"answer": answer}
                trace.status = NodeStatus.succeeded
            else:
                trace.status = NodeStatus.failed
            # Parse evidence citations
            trace.evidence_citations = _parse_citations(parsed)
        except Exception as exc:
            trace.status = NodeStatus.failed
            trace.error = str(exc)

        final_answer = trace.returned_value
        status = NodeStatus.succeeded if final_answer else NodeStatus.failed
        return RunTrace(
            run_id=run_id,
            question=plan.question,
            plan=plan,
            waves=[SchedulerWave(index=0, node_ids=["ltm"])],
            nodes=[trace],
            final_answer=final_answer,
            status=status,
            total_duration_ms=(time.perf_counter() - started) * 1000,
            config=self.config.model_dump(mode="json"),
        )

    async def _run_dag_nodes(
        self,
        plan: DagPlan,
        evidence_documents: list[EvidenceDocument],
    ) -> RunTrace:
        """Standard node-by-node DAG execution."""
        started = time.perf_counter()
        run_id = str(uuid4())
        waves = self.scheduler.build_waves(plan)
        by_id = {node.id: node for node in plan.nodes}
        outputs: dict[str, dict[str, Any]] = {}
        traces: list[NodeTrace] = []
        semaphore = asyncio.Semaphore(self.config.execution.max_parallel_nodes)

        async def run_node(node_id: str) -> NodeTrace:
            async with semaphore:
                return await asyncio.wait_for(
                    self.runner.run(
                        by_id[node_id],
                        outputs,
                        evidence_documents,
                        original_question=plan.question,
                        is_final=(node_id == plan.final_node),
                    ),
                    timeout=self.config.execution.node_timeout_seconds,
                )

        status = NodeStatus.succeeded
        for wave in waves:
            wave_traces = await asyncio.gather(
                *(run_node(node_id) for node_id in wave.node_ids),
                return_exceptions=True,
            )
            for node_id, result in zip(wave.node_ids, wave_traces, strict=True):
                if isinstance(result, Exception):
                    trace = self._exception_trace(by_id[node_id], result)
                else:
                    trace = result
                traces.append(trace)
                if trace.status == NodeStatus.succeeded and trace.returned_value is not None:
                    outputs[trace.node_id] = self._internal_output(trace)
                else:
                    status = NodeStatus.failed

        final_trace = next((t for t in traces if t.node_id == plan.final_node), None)
        final_answer = final_trace.returned_value if final_trace else None
        if final_answer is None:
            status = NodeStatus.failed
        return self._run_trace(run_id, plan, waves, traces, final_answer, status, started)

    @staticmethod
    def _extract_answer_str(result: Any) -> str:
        if isinstance(result, Exception):
            return ""
        if not isinstance(result, RunTrace):
            return ""
        if result.final_answer is None:
            return ""
        return str(result.final_answer.get("answer", "")).strip()

    _MAX_CONCISE_WORDS = 6
    _MAX_ANSWER_WORDS = 15

    @staticmethod
    def _grounding_score(answer: str, evidence_text: str) -> float:
        """Score how well an answer is grounded in evidence text.

        Prefers concise, evidence-grounded answers over verbose ones.
        Among exact matches, prefers answers with 2-6 words (specific
        entities/dates) over single-word matches (too vague) or very
        long matches (likely a sentence, not an extractive answer).
        """
        if not answer:
            return -1.0
        answer_lower = answer.casefold().strip()
        words = answer_lower.split()
        word_count = len(words)

        # Penalize overly verbose answers (likely a sentence, not an extraction)
        if word_count > DagExecutor._MAX_ANSWER_WORDS:
            return 0.1

        # Exact substring match in evidence = strong grounding
        if answer_lower in evidence_text:
            # 2-6 words (specific entities, dates) > 1 word (vague) > 7+ words
            if word_count <= 1:
                score = 2.5
            elif word_count <= DagExecutor._MAX_CONCISE_WORDS:
                score = 3.0
            else:
                score = 2.8
            return score

        # Fallback: word overlap ratio
        answer_words = set(words)
        evidence_words = set(evidence_text.split())
        overlap = answer_words & evidence_words
        return len(overlap) / max(len(answer_words), 1)

    def _exception_trace(self, node: Any, exc: Exception) -> NodeTrace:
        return NodeTrace(
            node_id=node.id,
            label=node.label,
            task_type=node.task_type,
            operation=node.operation,
            status=NodeStatus.failed,
            error=str(exc),
        )
