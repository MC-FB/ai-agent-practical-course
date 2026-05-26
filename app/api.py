from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dagqa.client import DagQaClient
from dagqa.config import AppConfig
from dagqa.eval.benchmark import BenchmarkResult, benchmark_hotpotqa
from dagqa.graph.render import render_mermaid
from dagqa.graph.scheduler import Scheduler
from dagqa.nodes.runner import NodeRunner
from dagqa.planning.planner import PlannerError
from dagqa.planning.parser import parse_plan
from dagqa.schemas import DagPlan, NodeStatus, NodeTrace, RunTrace, SchedulerWave

router = APIRouter()
RUNS: dict[str, RunTrace] = {}
CLIENTS: dict[str, DagQaClient] = {}
LIVE_RUNS: dict[str, dict[str, Any]] = {}


class AskRequest(BaseModel):
    question: str
    config_overrides: dict[str, Any] | None = None


class ExecuteRequest(BaseModel):
    plan_yaml: str


class BenchmarkRequest(BaseModel):
    system: str = "dag_agent"
    limit: int = 10
    data_path: str | None = None


def load_config() -> AppConfig:
    path = Path(os.getenv("DAGQA_CONFIG", "configs/local.yaml"))
    config = AppConfig.from_file(path) if path.exists() else AppConfig()
    return config


def client() -> DagQaClient:
    path = os.getenv("DAGQA_CONFIG", "configs/local.yaml")
    if path not in CLIENTS:
        CLIENTS[path] = DagQaClient(load_config())
    return CLIENTS[path]


@router.get("/config")
def get_config() -> dict[str, Any]:
    return load_config().model_dump(mode="json")


@router.post("/config/validate")
def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    return AppConfig.model_validate(config).model_dump(mode="json")


@router.post("/plan")
async def plan(request: AskRequest) -> dict[str, Any]:
    dag = await client().plan(request.question)
    return dag.model_dump(mode="json")


@router.post("/execute")
async def execute(request: ExecuteRequest) -> dict[str, Any]:
    run = await client().execute(parse_plan(request.plan_yaml))
    RUNS[run.run_id] = run
    payload = run.model_dump(mode="json")
    payload["mermaid"] = render_mermaid(run.plan, run.nodes)
    return payload


@router.post("/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    try:
        run = await client().ask(request.question)
    except PlannerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    RUNS[run.run_id] = run
    payload = run.model_dump(mode="json")
    payload["mermaid"] = render_mermaid(run.plan, run.nodes)
    return payload


@router.post("/ask/live")
async def ask_live(request: AskRequest) -> dict[str, Any]:
    run_id = str(uuid4())
    LIVE_RUNS[run_id] = {
        "run_id": run_id,
        "question": request.question,
        "phase": "planning",
        "status": "running",
        "nodes": [],
        "waves": [],
        "final_answer": None,
        "mermaid": None,
        "error": None,
        "total_duration_ms": 0,
    }
    asyncio.create_task(_run_live(run_id, request.question))
    return LIVE_RUNS[run_id]


@router.get("/ask/live/{run_id}")
def get_live_run(run_id: str) -> dict[str, Any]:
    if run_id not in LIVE_RUNS:
        raise HTTPException(status_code=404, detail="Run not found.")
    return LIVE_RUNS[run_id]


async def _run_live(run_id: str, question: str) -> None:
    started = time.perf_counter()
    cfg = load_config()
    dag_client = client()
    plan: DagPlan | None = None
    waves: list[SchedulerWave] = []
    traces: list[NodeTrace] = []
    outputs: dict[str, dict[str, Any]] = {}
    runner = NodeRunner(dag_client.llm, cfg.execution, cfg.llm)

    def update(
        phase: str,
        status: str = "running",
        error: str | None = None,
        final_answer: dict[str, Any] | None = None,
    ) -> None:
        LIVE_RUNS[run_id] = {
            "run_id": run_id,
            "question": question,
            "phase": phase,
            "status": status,
            "nodes": [trace.model_dump(mode="json") for trace in traces],
            "waves": [wave.model_dump(mode="json") for wave in waves],
            "final_answer": final_answer,
            "mermaid": render_mermaid(plan, traces) if plan is not None else None,
            "error": error,
            "total_duration_ms": (time.perf_counter() - started) * 1000,
        }

    try:
        plan = await dag_client.plan(question)
        waves = Scheduler().build_waves(plan)
        update("executing")

        by_id = {node.id: node for node in plan.nodes}
        semaphore = asyncio.Semaphore(cfg.execution.max_parallel_nodes)

        async def run_node(node_id: str) -> NodeTrace:
            async with semaphore:
                return await asyncio.wait_for(
                    runner.run(by_id[node_id], outputs),
                    timeout=cfg.execution.node_timeout_seconds,
                )

        status = NodeStatus.succeeded
        for wave in waves:
            running_traces = [
                NodeTrace(
                    node_id=node_id,
                    label=by_id[node_id].label,
                    task_type=by_id[node_id].task_type,
                    operation=by_id[node_id].operation,
                    status=NodeStatus.running,
                )
                for node_id in wave.node_ids
            ]
            traces.extend(running_traces)
            update("executing")

            results = await asyncio.gather(
                *(run_node(node_id) for node_id in wave.node_ids),
                return_exceptions=True,
            )
            for node_id, result in zip(wave.node_ids, results, strict=True):
                index = next(
                    idx
                    for idx, trace in enumerate(traces)
                    if trace.node_id == node_id and trace.status == NodeStatus.running
                )
                if isinstance(result, Exception):
                    trace = NodeTrace(
                        node_id=node_id,
                        label=by_id[node_id].label,
                        task_type=by_id[node_id].task_type,
                        operation=by_id[node_id].operation,
                        status=NodeStatus.failed,
                        error=str(result),
                    )
                else:
                    trace = result
                traces[index] = trace
                if trace.status == NodeStatus.succeeded and trace.returned_value is not None:
                    outputs[trace.node_id] = trace.returned_value
                else:
                    status = NodeStatus.failed
            update("executing", status=status.value)

        final_answer = outputs.get(plan.final_node)
        if final_answer is None:
            status = NodeStatus.failed
        run_trace = RunTrace(
            run_id=run_id,
            question=question,
            plan=plan,
            waves=waves,
            nodes=traces,
            final_answer=final_answer,
            status=status,
            total_duration_ms=(time.perf_counter() - started) * 1000,
            config=cfg.model_dump(mode="json"),
        )
        RUNS[run_id] = run_trace
        update("complete", status=status.value, final_answer=final_answer)
    except PlannerError as exc:
        update("error", status="failed", error=str(exc))
    except Exception as exc:
        update("error", status="failed", error=str(exc))


@router.post("/benchmarks/hotpotqa")
async def hotpotqa(request: BenchmarkRequest) -> BenchmarkResult:
    return await benchmark_hotpotqa(
        client(),
        system=request.system,  # type: ignore[arg-type]
        limit=request.limit,
        path=request.data_path,
    )


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = RUNS[run_id]
    return run.model_dump(mode="json")


@router.get("/runs/{run_id}/trace")
def get_trace(run_id: str) -> dict[str, Any]:
    run = RUNS[run_id]
    return {"run_id": run_id, "nodes": [node.model_dump(mode="json") for node in run.nodes]}
