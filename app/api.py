from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dagqa.client import DagQaClient
from dagqa.config import AppConfig, LLMConfig
from dagqa.eval.benchmark import (
    BenchmarkRecord,
    BenchmarkResult,
    _aggregate,
    _run_examples,
    _sample_examples,
    benchmark_hotpotqa,
)
from dagqa.eval.hotpot_loader import count_hotpot_examples, load_hotpot_examples
from dagqa.eval.metrics import cosine_sim
from dagqa.graph.render import render_mermaid
from dagqa.graph.scheduler import Scheduler
from dagqa.nodes.runner import NodeRunner
from dagqa.planning.parser import parse_plan
from dagqa.planning.planner import PlannerError
from dagqa.schemas import DagPlan, NodeStatus, NodeTrace, RunTrace, SchedulerWave

router = APIRouter()
RUNS: dict[str, RunTrace] = {}
LIVE_RUNS: dict[str, dict[str, Any]] = {}
LIVE_BENCHMARKS: dict[str, dict[str, Any]] = {}

CLUSTER_API_BASE = "http://atknoll32.air.cit.tum.de:3000/inference"
CLUSTER_MODELS_URL = "http://atknoll32.air.cit.tum.de:3000/models"
DEFAULT_CLUSTER_MODEL = "openai/gpt-oss-120b"
PAIRED_SYSTEM_COUNT = 2


class LLMSelection(BaseModel):
    provider: Literal["azure_openai", "cluster"]
    model: str


class AskRequest(BaseModel):
    question: str
    llm: LLMSelection | None = None


class ExecuteRequest(BaseModel):
    plan_yaml: str


class BenchmarkRequest(BaseModel):
    system: Literal["dag_agent", "direct_llm"] = "dag_agent"
    systems: list[Literal["dag_agent", "direct_llm"]] | None = None
    limit: int = 10
    seed: int | None = None
    data_path: str | None = None
    llm: LLMSelection | None = None


def load_config() -> AppConfig:
    path = Path(os.getenv("DAGQA_CONFIG", "configs/local.yaml"))
    config = AppConfig.from_file(path) if path.exists() else AppConfig()
    return config


def _resolved_model(config: LLMConfig) -> str:
    return (os.getenv(config.model_env) if config.model_env else None) or config.model


def _azure_llm_config() -> LLMConfig:
    active = load_config().llm
    if active.provider == "azure_openai":
        return active
    path = Path("configs/azure-openai.yaml")
    if path.exists():
        return AppConfig.from_file(path).llm
    raise HTTPException(status_code=503, detail="Azure OpenAI config is not available.")


def config_for_selection(selection: LLMSelection | None = None) -> AppConfig:
    config = load_config()
    if selection is None:
        resolved = _resolved_model(config.llm)
        return config.model_copy(
            update={"llm": config.llm.model_copy(update={"model": resolved, "model_env": None})}
        )

    model = selection.model.strip()
    if not model:
        raise HTTPException(status_code=422, detail="A model must be selected.")
    if selection.provider == "azure_openai":
        azure = _azure_llm_config()
        expected = _resolved_model(azure)
        if model not in {expected, azure.model}:
            raise HTTPException(status_code=422, detail="Unknown Azure OpenAI model.")
        llm = azure.model_copy(update={"model": expected, "model_env": None})
    else:
        llm = LLMConfig(
            provider="cluster",
            model=model,
            temperature=config.llm.temperature,
            api_key_env="CLUSTER_API_KEY",
            api_base=os.getenv("CLUSTER_API_BASE", CLUSTER_API_BASE),
        )
    return config.model_copy(update={"llm": llm})


def client(selection: LLMSelection | None = None) -> DagQaClient:
    return DagQaClient(config_for_selection(selection))


async def _cluster_models() -> list[str]:
    api_key = os.getenv("CLUSTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="CLUSTER_API_KEY is not configured.")
    url = os.getenv("CLUSTER_MODELS_URL", CLUSTER_MODELS_URL)
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.get(url, headers={"Authorization": f"Bearer {api_key}"})
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load cluster models: {exc}",
        ) from exc
    data = response.json().get("data", [])
    models = [item if isinstance(item, str) else item.get("id") for item in data]
    return sorted(model for model in models if isinstance(model, str) and model)


@router.get("/llm/models")
async def list_llm_models() -> dict[str, Any]:
    azure = _azure_llm_config()
    azure_model = _resolved_model(azure)
    options = [
        {
            "provider": "azure_openai",
            "model": azure_model,
            "label": f"Azure · {azure_model.removeprefix('azure/')}",
        }
    ]
    cluster_error = None
    try:
        cluster_models = await _cluster_models()
    except HTTPException as exc:
        cluster_models = []
        cluster_error = exc.detail
    options.extend(
        {"provider": "cluster", "model": model, "label": f"Chair cluster · {model}"}
        for model in cluster_models
    )
    default = (
        {"provider": "cluster", "model": DEFAULT_CLUSTER_MODEL}
        if DEFAULT_CLUSTER_MODEL in cluster_models
        else {"provider": "azure_openai", "model": azure_model}
    )
    return {
        "models": options,
        "default": default,
        "cluster_error": cluster_error,
    }


@router.get("/config")
def get_config() -> dict[str, Any]:
    return load_config().model_dump(mode="json")


@router.post("/config/validate")
def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    return AppConfig.model_validate(config).model_dump(mode="json")


@router.post("/plan")
async def plan(request: AskRequest) -> dict[str, Any]:
    dag = await client(request.llm).plan(request.question)
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
        run = await client(request.llm).ask(request.question)
    except PlannerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    RUNS[run.run_id] = run
    payload = run.model_dump(mode="json")
    payload["mermaid"] = render_mermaid(run.plan, run.nodes)
    return payload


@router.post("/ask/live")
async def ask_live(request: AskRequest) -> dict[str, Any]:
    run_id = str(uuid4())
    cfg = config_for_selection(request.llm)
    dag_client = DagQaClient(cfg)
    LIVE_RUNS[run_id] = {
        "run_id": run_id,
        "question": request.question,
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
        "phase": "planning",
        "status": "running",
        "nodes": [],
        "waves": [],
        "final_answer": None,
        "mermaid": None,
        "error": None,
        "total_duration_ms": 0,
    }
    asyncio.create_task(_run_live(run_id, request.question, dag_client, cfg))
    return LIVE_RUNS[run_id]


@router.get("/ask/live/{run_id}")
def get_live_run(run_id: str) -> dict[str, Any]:
    if run_id not in LIVE_RUNS:
        raise HTTPException(status_code=404, detail="Run not found.")
    return LIVE_RUNS[run_id]


async def _run_live(
    run_id: str,
    question: str,
    dag_client: DagQaClient,
    cfg: AppConfig,
) -> None:
    started = time.perf_counter()
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
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
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
    systems = _benchmark_systems(request)
    if len(systems) != 1:
        raise HTTPException(status_code=422, detail="Use the live endpoint for paired benchmarks.")
    dag_client = client(request.llm)
    result = await benchmark_hotpotqa(
        dag_client,
        system=systems[0],
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
    systems = _benchmark_systems(request)
    comparison_group_id = str(uuid4()) if len(systems) == PAIRED_SYSTEM_COUNT else None
    cfg = config_for_selection(request.llm)
    dag_client = DagQaClient(cfg)
    seed = request.seed if request.seed is not None else int(time.time_ns() % 2_147_483_647)
    LIVE_BENCHMARKS[run_id] = {
        "run_id": run_id,
        "phase": "running",
        "status": "running",
        "system": systems[0],
        "systems": systems,
        "current_system": systems[0],
        "comparison_group_id": comparison_group_id,
        "comparison_run_ids": [],
        "comparison_results": [],
        "limit": request.limit,
        "seed": seed,
        "max_parallel_examples": cfg.benchmark.max_parallel_examples,
        "dataset": "hotpotqa",
        "split": cfg.benchmark.split,
        "dataset_size": count_hotpot_examples(request.data_path),
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": 0,
        "total": request.limit * len(systems),
        "current_question": None,
        "total_runtime_ms": 0,
        "metrics": {},
        "records": [],
        "output_path": None,
        "error": None,
    }
    asyncio.create_task(
        _run_live_benchmark(
            run_id,
            request,
            seed,
            dag_client,
            cfg,
            systems,
            comparison_group_id,
        )
    )
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


async def _run_live_benchmark(
    run_id: str,
    request: BenchmarkRequest,
    seed: int,
    dag_client: DagQaClient,
    cfg: AppConfig,
    systems: list[Literal["dag_agent", "direct_llm"]] | None = None,
    comparison_group_id: str | None = None,
) -> None:
    started = time.perf_counter()
    records: list[BenchmarkRecord] = []
    systems = systems or _benchmark_systems(request)
    completed_results: list[BenchmarkResult] = []
    current_system = systems[0]
    created_at = LIVE_BENCHMARKS[run_id]["created_at"]
    total_examples = request.limit * len(systems)
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
            "system": current_system,
            "systems": systems,
            "current_system": current_system,
            "comparison_group_id": comparison_group_id,
            "comparison_run_ids": [result.run_id for result in completed_results],
            "comparison_results": [result.model_dump(mode="json") for result in completed_results],
            "limit": request.limit,
            "seed": seed,
            "max_parallel_examples": cfg.benchmark.max_parallel_examples,
            "dataset": "hotpotqa",
            "split": cfg.benchmark.split,
            "dataset_size": dataset_size or None,
            "provider": dag_client.config.llm.provider,
            "model": dag_client.config.llm.model,
            "created_at": created_at,
            "completed": sum(len(result.records) for result in completed_results) + len(records),
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
            all_examples = []
        else:
            all_examples = await asyncio.to_thread(load_hotpot_examples, request.data_path)
            dataset_size = len(all_examples)
        sampled_examples = _sample_examples(all_examples, request.limit, seed)
        per_system_total = len(sampled_examples)
        total_examples = per_system_total * len(systems)

        for system_index, system in enumerate(systems):
            system_started = time.perf_counter()
            current_system = system
            records = []
            update(
                current_question=(
                    f"Running {system.replace('_', ' ')} with up to "
                    f"{min(cfg.benchmark.max_parallel_examples, per_system_total)} examples "
                    "in parallel."
                    if sampled_examples
                    else None
                )
            )

            def record_completed(
                record: BenchmarkRecord,
                records_ref: list[BenchmarkRecord] = records,
                system_name: str = system,
            ) -> None:
                records_ref.append(record)
                remaining = per_system_total - len(records_ref)
                update(
                    current_question=(
                        f"{remaining} {system_name.replace('_', ' ')} examples remaining."
                        if remaining
                        else None
                    )
                )

            ordered_records = await _run_examples(
                dag_client,
                sampled_examples,
                system,
                cfg.benchmark.max_parallel_examples,
                record_completed,
            )
            records = ordered_records

            system_runtime_ms = (time.perf_counter() - system_started) * 1000
            metrics = _aggregate(records)
            metrics["total_runtime_ms"] = system_runtime_ms
            result = BenchmarkResult(
                run_id=run_id if len(systems) == 1 else str(uuid4()),
                comparison_group_id=comparison_group_id,
                system=system,
                limit=request.limit,
                provider=dag_client.config.llm.provider,
                model=dag_client.config.llm.model,
                dataset="hotpotqa",
                split=cfg.benchmark.split,
                dataset_size=dataset_size,
                seed=seed,
                max_parallel_examples=cfg.benchmark.max_parallel_examples,
                created_at=created_at,
                total_runtime_ms=system_runtime_ms,
                records=records,
                metrics=metrics,
            )
            output_path = _save_benchmark_result(result)
            result.output_path = str(output_path)
            output_path.write_text(result.model_dump_json(indent=2))
            completed_results.append(result)
            records = []
            if system_index + 1 < len(systems):
                current_system = systems[system_index + 1]
                update(current_question=f"Starting {current_system.replace('_', ' ')}.")

        update(
            "complete",
            status="succeeded",
            output_path=completed_results[-1].output_path if completed_results else None,
        )
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
        "provider": cfg.llm.provider,
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


def _benchmark_systems(
    request: BenchmarkRequest,
) -> list[Literal["dag_agent", "direct_llm"]]:
    systems = request.systems or [request.system]
    resolved = [system for system in ("dag_agent", "direct_llm") if system in systems]
    if not resolved:
        raise HTTPException(status_code=422, detail="Select at least one benchmark system.")
    return resolved


@router.get("/benchmarks/results", tags=["Benchmarks"], summary="List benchmark results")
def list_benchmark_results() -> dict[str, Any]:
    output_dir = _benchmark_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for path in output_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        normalized = _normalize_benchmark_payload(data, path)
        items.append(
            {
                "run_id": normalized.get("run_id") or path.stem,
                "comparison_group_id": normalized.get("comparison_group_id"),
                "created_at": normalized.get("created_at"),
                "dataset": normalized.get("dataset"),
                "split": normalized.get("split"),
                "system": normalized.get("system"),
                "provider": normalized.get("provider"),
                "model": normalized.get("model"),
                "limit": normalized.get("limit"),
                "seed": normalized.get("seed"),
                "metrics": normalized.get("metrics", {}),
                "path": str(path),
            }
        )
    items.sort(
        key=lambda item: (item.get("created_at") or "", item.get("run_id") or ""),
        reverse=True,
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
    repaired = _repair_benchmark_payload(raw, path)
    return _normalize_benchmark_payload(repaired, path)


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
        "provider": data.get("provider"),
        "model": data.get("model") or "",
        "seed": data.get("seed") or 0,
        "limit": data.get("limit") or len(records),
        "max_parallel_examples": data.get("max_parallel_examples") or 1,
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
            record["cosine_sim"] = cosine_sim(
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
        "gold_supporting_facts": record.get("gold_supporting_facts") or [],
        "evidence_citation_count": record.get("evidence_citation_count"),
        "correct_evidence_citation_count": record.get("correct_evidence_citation_count"),
        "wrong_evidence_citation_count": record.get("wrong_evidence_citation_count"),
        "wrong_supporting_text_rate": record.get("wrong_supporting_text_rate"),
        "gold_supporting_fact_recall": record.get("gold_supporting_fact_recall"),
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
