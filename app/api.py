from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dagqa.client import DagQaClient
from dagqa.config import AppConfig
from dagqa.eval.benchmark import (
    BenchmarkRecord,
    BenchmarkResult,
    _aggregate,
    _run_example,
    _sample_examples,
    benchmark_hotpotqa,
)
from dagqa.eval.hotpot_loader import count_hotpot_examples, load_hotpot_examples
from dagqa.graph.render import render_mermaid
from dagqa.graph.scheduler import Scheduler
from dagqa.nodes.runner import NodeRunner
from dagqa.planning.parser import parse_plan
from dagqa.planning.planner import PlannerError
from dagqa.schemas import DagPlan, NodeStatus, NodeTrace, RunTrace, SchedulerWave

router = APIRouter()
RUNS: dict[str, RunTrace] = {}
CLIENTS: dict[str, DagQaClient] = {}
LIVE_RUNS: dict[str, dict[str, Any]] = {}
LIVE_BENCHMARKS: dict[str, dict[str, Any]] = {}


class AskRequest(BaseModel):
    question: str
    config_overrides: dict[str, Any] | None = None


class ExecuteRequest(BaseModel):
    plan_yaml: str


class BenchmarkRequest(BaseModel):
    system: str = "dag_agent"
    limit: int = 10
    seed: int | None = None
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
            "plan": plan.model_dump(mode="json") if plan is not None else None,
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


@router.post("/benchmarks/hotpotqa", tags=["Benchmarks"], summary="Run HotpotQA benchmark")
async def hotpotqa(request: BenchmarkRequest) -> BenchmarkResult:
    result = await benchmark_hotpotqa(
        client(),
        system=request.system,  # type: ignore[arg-type]
        limit=request.limit,
        seed=request.seed,
        path=request.data_path,
    )
    output_path = _save_benchmark_result(result)
    result.output_path = str(output_path)
    output_path.write_text(result.model_dump_json(indent=2))
    return result


@router.get(
    "/benchmarks/hotpotqa",
    tags=["Benchmarks"],
    summary="Get HotpotQA benchmark metadata",
)
def hotpotqa_info(data_path: str | None = None) -> dict[str, Any]:
    return hotpotqa_meta(data_path)


@router.post(
    "/benchmarks/hotpotqa/live",
    tags=["Benchmarks"],
    summary="Start live HotpotQA benchmark",
)
async def hotpotqa_live(request: BenchmarkRequest) -> dict[str, Any]:
    run_id = str(uuid4())
    seed = request.seed if request.seed is not None else int(time.time_ns() % 2_147_483_647)
    LIVE_BENCHMARKS[run_id] = {
        "run_id": run_id,
        "phase": "running",
        "status": "running",
        "system": request.system,
        "limit": request.limit,
        "seed": seed,
        "dataset": "hotpotqa",
        "split": load_config().benchmark.split,
        "dataset_size": count_hotpot_examples(request.data_path),
        "model": load_config().llm.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": 0,
        "total": request.limit,
        "current_question": None,
        "total_runtime_ms": 0,
        "metrics": {},
        "records": [],
        "output_path": None,
        "error": None,
    }
    asyncio.create_task(_run_live_benchmark(run_id, request, seed))
    return LIVE_BENCHMARKS[run_id]


@router.get(
    "/benchmarks/hotpotqa/live/{run_id}",
    tags=["Benchmarks"],
    summary="Get live HotpotQA benchmark status",
)
def get_live_benchmark(run_id: str) -> dict[str, Any]:
    if run_id not in LIVE_BENCHMARKS:
        raise HTTPException(status_code=404, detail="Benchmark run not found.")
    return LIVE_BENCHMARKS[run_id]


async def _run_live_benchmark(run_id: str, request: BenchmarkRequest, seed: int) -> None:
    started = time.perf_counter()
    records = []
    dag_client = client()
    cfg = load_config()
    created_at = LIVE_BENCHMARKS[run_id]["created_at"]
    total_examples = request.limit
    dataset_size = 0

    def update(
        phase: str = "running",
        status: str = "running",
        current_question: str | None = None,
        error: str | None = None,
        output_path: str | None = None,
    ) -> None:
        metrics = _aggregate(records)
        total_runtime_ms = (time.perf_counter() - started) * 1000
        if records:
            metrics["total_runtime_ms"] = total_runtime_ms
        LIVE_BENCHMARKS[run_id] = {
            "run_id": run_id,
            "phase": phase,
            "status": status,
            "system": request.system,
            "limit": request.limit,
            "seed": seed,
            "dataset": "hotpotqa",
            "split": cfg.benchmark.split,
            "dataset_size": dataset_size or None,
            "model": dag_client.config.llm.model,
            "created_at": created_at,
            "completed": len(records),
            "total": total_examples,
            "current_question": current_question,
            "total_runtime_ms": total_runtime_ms,
            "metrics": metrics,
            "records": [record.model_dump(mode="json") for record in records],
            "output_path": output_path,
            "error": error,
        }

    try:
        if request.limit <= 0 and request.data_path is None:
            dataset_size = count_hotpot_examples()
            examples = []
        else:
            all_examples = await asyncio.to_thread(load_hotpot_examples, request.data_path)
            dataset_size = len(all_examples)
            examples = _sample_examples(all_examples, request.limit, seed)
        total_examples = len(examples)
        update(current_question=examples[0].question if examples else None)
        for example in examples:
            update(current_question=example.question)
            records.append(await _run_example(dag_client, example, request.system))
            next_index = len(records)
            next_question = examples[next_index].question if next_index < len(examples) else None
            update(current_question=next_question)

        total_runtime_ms = (time.perf_counter() - started) * 1000
        metrics = _aggregate(records)
        metrics["total_runtime_ms"] = total_runtime_ms
        result = BenchmarkResult(
            run_id=run_id,
            system=request.system,
            limit=request.limit,
            model=dag_client.config.llm.model,
            dataset="hotpotqa",
            split=cfg.benchmark.split,
            dataset_size=dataset_size,
            seed=seed,
            created_at=created_at,
            total_runtime_ms=total_runtime_ms,
            records=records,
            metrics=metrics,
        )
        output_path = _save_benchmark_result(result)
        result.output_path = str(output_path)
        output_path.write_text(result.model_dump_json(indent=2))
        update("complete", status="succeeded", output_path=str(output_path))
    except Exception as exc:
        update("error", status="failed", error=str(exc))


@router.get(
    "/benchmarks/hotpotqa/meta",
    tags=["Benchmarks"],
    summary="Get HotpotQA benchmark metadata",
)
def hotpotqa_meta(data_path: str | None = None) -> dict[str, Any]:
    cfg = load_config()
    return {
        "dataset": "hotpotqa",
        "split": cfg.benchmark.split,
        "total_examples": count_hotpot_examples(data_path),
        "default_limit": cfg.benchmark.default_limit,
        "model": cfg.llm.model,
        "system": "dag_agent",
        "baselines": [
            {
                "label": "HotpotQA distractor validation size",
                "metric": "examples",
                "value": count_hotpot_examples(data_path),
                "source": "Hugging Face hotpotqa/hotpot_qa dataset card",
            }
        ],
    }


@router.get("/benchmarks/results", tags=["Benchmarks"], summary="List benchmark results")
def list_benchmark_results() -> dict[str, Any]:
    output_dir = _benchmark_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    result_paths = sorted(
        output_dir.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in result_paths:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        normalized = _normalize_benchmark_payload(data, path)
        items.append(
            {
                "run_id": normalized.get("run_id") or path.stem,
                "created_at": normalized.get("created_at"),
                "dataset": normalized.get("dataset"),
                "split": normalized.get("split"),
                "system": normalized.get("system"),
                "model": normalized.get("model"),
                "limit": normalized.get("limit"),
                "seed": normalized.get("seed"),
                "metrics": normalized.get("metrics", {}),
                "path": str(path),
            }
        )
    return {"results": items}


@router.get(
    "/benchmarks/results/{run_id}",
    tags=["Benchmarks"],
    summary="Get benchmark result",
)
def get_benchmark_result(run_id: str) -> dict[str, Any]:
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=400, detail="Invalid run id.")
    output_dir = _benchmark_output_dir()
    matches = list(output_dir.glob(f"*{run_id}*.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="Benchmark result not found.")
    path = matches[0]
    return _normalize_benchmark_payload(json.loads(path.read_text()), path)


@router.post("/benchmarks/results/{run_id}/repair")
def repair_benchmark_result(run_id: str) -> dict[str, Any]:
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=400, detail="Invalid run id.")
    output_dir = _benchmark_output_dir()
    matches = list(output_dir.glob(f"*{run_id}*.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="Benchmark result not found.")
    path = matches[0]
    raw = json.loads(path.read_text())
    repaired = _repair_benchmark_payload(raw, path)   # detects missing, calculates, writes to disk
    return _normalize_benchmark_payload(repaired, path)  # normalizes the now-repaired data for response

def _benchmark_output_dir() -> Path:
    return Path(load_config().benchmark.output_dir)


def _save_benchmark_result(result: BenchmarkResult) -> Path:
    output_dir = _benchmark_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = result.created_at.replace(":", "").replace("-", "")
    filename = f"hotpotqa-{timestamp}-{result.run_id}.json"
    return output_dir / filename


def _normalize_benchmark_payload(data: dict[str, Any], path: Path) -> dict[str, Any]:
    metrics = data.get("metrics") or data.get("summary") or {}
    records = [_normalize_benchmark_record(record) for record in data.get("records", [])]
    dataset_size = data.get("dataset_size") or data.get("total_examples")
    if dataset_size is None:
        try:
            dataset_size = count_hotpot_examples()
        except Exception:
            dataset_size = len(records)
    return {
        **data,
        "run_id": data.get("run_id") or path.stem,
        "dataset": data.get("dataset") or "hotpotqa",
        "split": data.get("split") or "validation",
        "dataset_size": dataset_size,
        "system": data.get("system") or "dag_agent",
        "model": data.get("model") or "",
        "seed": data.get("seed") or 0,
        "limit": data.get("limit") or len(records),
        "created_at": data.get("created_at"),
        "output_path": data.get("output_path") or str(path),
        "total_runtime_ms": data.get("total_runtime_ms") or metrics.get("total_runtime_ms") or 0,
        "metrics": metrics,
        "records": records,
    }


def _repair_benchmark_payload(data: dict[str, Any], path: Path) -> dict[str, Any]:
    needs_save = False

    # Patch per-record cosine_sim if missing
    for record in data.get("records", []):
        if "cosine_sim" not in record:
            from dagqa.eval.metrics import cosine_sim as _cosine_sim
            record["cosine_sim"] = _cosine_sim(
                record.get("prediction", " "),
                record.get("gold_answer", " "),
            )
            needs_save = True

    # Patch aggregate metrics if missing
    metrics = dict(data.get("metrics") or {})
    if "cosine_sim" not in metrics:
        records = [BenchmarkRecord.model_validate(r) for r in data.get("records", [])]
        metrics.update(_aggregate(records))
        data["metrics"] = metrics
        needs_save = True

    if needs_save:
        path.write_text(json.dumps(data, indent=2))

    return data

def _normalize_benchmark_record(record: dict[str, Any]) -> dict[str, Any]:
    structure = record.get("structure") or {}
    run_trace = _normalize_run_trace(record.get("run_trace"))
    return {
        "id": record.get("id") or str(record.get("index") or ""),
        "question": record.get("question") or "",
        "gold_answer": record.get("gold_answer") or "",
        "prediction": record.get("prediction") or "",
        "exact_match": record.get("exact_match") or 0,
        "f1": record.get("f1") or 0,
        "cosine_sim": record.get("cosine_sim") if record.get("cosine_sim") is not None else 0,
        "latency_ms": record.get("latency_ms") or 0,
        "llm_call_count": record.get("llm_call_count"),
        "node_count": record.get("node_count"),
        "graph_depth": record.get("graph_depth") or record.get("wave_count"),
        "structural_failure": record.get("structural_failure")
        if record.get("structural_failure") is not None
        else not bool(structure.get("valid", True)),
        "structural_valid": record.get("structural_valid")
        if record.get("structural_valid") is not None
        else structure.get("valid"),
        "structural_issues": record.get("structural_issues") or structure.get("problems", []),
        "run_trace": run_trace,
        "error": record.get("error"),
    }


def _normalize_run_trace(run_trace: Any) -> dict[str, Any] | None:
    if not isinstance(run_trace, dict):
        return None
    normalized = dict(run_trace)
    if not normalized.get("mermaid") and normalized.get("plan") and normalized.get("nodes"):
        try:
            plan = DagPlan.model_validate(normalized["plan"])
            traces = [NodeTrace.model_validate(node) for node in normalized.get("nodes", [])]
            normalized["mermaid"] = render_mermaid(plan, traces)
        except Exception:
            normalized["mermaid"] = None
    return normalized


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = RUNS[run_id]
    return run.model_dump(mode="json")


@router.get("/runs/{run_id}/trace")
def get_trace(run_id: str) -> dict[str, Any]:
    run = RUNS[run_id]
    return {"run_id": run_id, "nodes": [node.model_dump(mode="json") for node in run.nodes]}
