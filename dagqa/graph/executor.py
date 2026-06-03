from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

from dagqa.config import AppConfig
from dagqa.graph.scheduler import Scheduler
from dagqa.llm.base import LanguageModel
from dagqa.nodes.runner import NodeRunner
from dagqa.planning.normalizer import normalize_plan_dependencies
from dagqa.planning.validator import validate_plan
from dagqa.schemas import DagPlan, EvidenceDocument, NodeStatus, NodeTrace, RunTrace, SchedulerWave


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
                    self.runner.run(by_id[node_id], outputs, evidence_documents),
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
                    outputs[trace.node_id] = trace.returned_value
                else:
                    status = NodeStatus.failed
                    if self.config.execution.fail_fast:
                        return self._run_trace(run_id, plan, waves, traces, None, status, started)

        final_answer = outputs.get(plan.final_node)
        if final_answer is None:
            status = NodeStatus.failed
        return self._run_trace(run_id, plan, waves, traces, final_answer, status, started)

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

    def _exception_trace(self, node: Any, exc: Exception) -> NodeTrace:
        return NodeTrace(
            node_id=node.id,
            label=node.label,
            task_type=node.task_type,
            operation=node.operation,
            status=NodeStatus.failed,
            error=str(exc),
        )
