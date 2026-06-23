from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import UTC, datetime
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
    benchmark_dataset,
)
from dagqa.eval.datasets import (
    DATASETS,
    count_benchmark_examples,
    get_benchmark_dataset,
    load_benchmark_examples,
)
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
LIVE_BENCHMARK_TASKS: dict[str, asyncio.Task[None]] = {}
LIVE_BENCHMARK_STOPS: dict[str, asyncio.Event] = {}

CLUSTER_API_BASE = "http://atknoll32.air.cit.tum.de:3000/inference"
CLUSTER_MODELS_URL = "http://atknoll32.air.cit.tum.de:3000/models"
PREFERRED_CLUSTER_MODELS = (
    "Qwen/Qwen3.5-122B-A10B",
    "mistralai/Mistral-Medium-3.5-128B",
    "google/gemma-4-31B-it",
)
DEFAULT_CLUSTER_MODEL = PREFERRED_CLUSTER_MODELS[0]
CLUSTER_MODEL_MAX_PARALLEL_EXAMPLES = {
    "google/gemma-4-31B-it": 2,
}
PAIRED_SYSTEM_COUNT = 2
MUSIQUE_BENCHMARK_MAX_DEPTH = 6
BenchmarkDatasetId = Literal["hotpotqa", "musique"]


class LLMSelection(BaseModel):
    provider: Literal["azure_openai", "cluster"]
    model: str


class AskRequest(BaseModel):
    question: str
    llm: LLMSelection | None = None


class ExecuteRequest(BaseModel):
    plan_yaml: str


class BenchmarkRequest(BaseModel):
    name: str | None = None
    system: Literal["dag_agent", "direct_llm"] = "dag_agent"
    systems: list[Literal["dag_agent", "direct_llm"]] | None = None
    limit: int = 10
    seed: int | None = None
    dataset: BenchmarkDatasetId = "hotpotqa"
    subset: str = "validation"
    data_path: str | None = None
    llm: LLMSelection | None = None


class BenchmarkResumeRequest(BaseModel):
    llm: LLMSelection | None = None


class BenchmarkPreflightCheck(BaseModel):
    name: str
    ok: bool
    detail: str


class BenchmarkPreflightResult(BaseModel):
    ok: bool
    checks: list[BenchmarkPreflightCheck]


class MarkedBenchmarkRow(BaseModel):
    key: str
    record_id: str
    focus_run_id: str
    reference_run_id: str
    focus_run_name: str | None = None
    reference_run_name: str | None = None
    focus_system: str
    reference_system: str
    model: str
    created_at: str
    seed: int
    question: str
    gold_answer: str
    focus_prediction: str
    reference_prediction: str
    focus_cosine_sim: float
    reference_cosine_sim: float
    focus_gold_recall: float | None = None
    reference_gold_recall: float | None = None


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


def _benchmark_subset_path(request: BenchmarkRequest) -> str | None:
    if request.data_path:
        return request.data_path
    return get_benchmark_dataset(request.dataset).subset(request.subset).path


def _benchmark_subset_label(subset: str, dataset: str = "hotpotqa") -> str:
    try:
        return get_benchmark_dataset(dataset).subset(subset).label
    except ValueError:
        return subset


def _benchmark_default_limit(
    config: AppConfig,
    dataset: str,
    subset: str,
    data_path: str | None = None,
) -> int:
    dataset_info = get_benchmark_dataset(dataset)
    if data_path is not None or subset != dataset_info.default_subset:
        return count_benchmark_examples(dataset, subset, data_path)
    return config.benchmark.default_limit


def _with_model_parallelism(config: AppConfig, model: str) -> AppConfig:
    max_parallel = CLUSTER_MODEL_MAX_PARALLEL_EXAMPLES.get(model)
    if max_parallel is None:
        return config
    return config.model_copy(
        update={
            "benchmark": config.benchmark.model_copy(update={"max_parallel_examples": max_parallel})
        }
    )


def config_for_selection(selection: LLMSelection | None = None) -> AppConfig:
    config = load_config()
    if selection is None:
        resolved = _resolved_model(config.llm)
        selected = config.model_copy(
            update={"llm": config.llm.model_copy(update={"model": resolved, "model_env": None})}
        )
        return _with_model_parallelism(selected, resolved)

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
            request_timeout_seconds=config.llm.request_timeout_seconds,
            max_retries=config.llm.max_retries,
            retry_initial_delay_seconds=config.llm.retry_initial_delay_seconds,
            retry_max_delay_seconds=config.llm.retry_max_delay_seconds,
        )
    return _with_model_parallelism(config.model_copy(update={"llm": llm}), llm.model)


def client(selection: LLMSelection | None = None) -> DagQaClient:
    return DagQaClient(config_for_selection(selection))


async def validated_config_for_selection(selection: LLMSelection | None = None) -> AppConfig:
    config = load_config()
    if selection is None:
        resolved = _resolved_model(config.llm)
        if config.llm.provider != "cluster":
            return config.model_copy(
                update={"llm": config.llm.model_copy(update={"model": resolved, "model_env": None})}
            )
        cluster_models = await _cluster_models()
        selected_model = (
            resolved if resolved in cluster_models else _preferred_cluster_model(cluster_models)
        )
        selected = config.model_copy(
            update={
                "llm": config.llm.model_copy(update={"model": selected_model, "model_env": None})
            }
        )
        return _with_model_parallelism(selected, selected_model)

    if selection.provider != "cluster":
        return config_for_selection(selection)

    model = selection.model.strip()
    if not model:
        raise HTTPException(status_code=422, detail="A model must be selected.")
    cluster_models = await _cluster_models()
    if model not in cluster_models:
        available = ", ".join(cluster_models) if cluster_models else "none"
        raise HTTPException(
            status_code=422,
            detail=f"Selected cluster model is not available. Current cluster models: {available}.",
        )
    return config_for_selection(selection)


async def validated_client(selection: LLMSelection | None = None) -> DagQaClient:
    return DagQaClient(await validated_config_for_selection(selection))


def _config_for_benchmark_dataset(config: AppConfig, dataset: str) -> AppConfig:
    if dataset != "musique" or config.planner.max_depth >= MUSIQUE_BENCHMARK_MAX_DEPTH:
        return config
    return config.model_copy(
        update={
            "planner": config.planner.model_copy(update={"max_depth": MUSIQUE_BENCHMARK_MAX_DEPTH})
        }
    )


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


def _preferred_cluster_model(cluster_models: list[str]) -> str:
    if not cluster_models:
        raise HTTPException(status_code=503, detail="Cluster model catalog is empty.")
    available = set(cluster_models)
    for model in PREFERRED_CLUSTER_MODELS:
        if model in available:
            return model
    return cluster_models[0]


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
        {"provider": "cluster", "model": _preferred_cluster_model(cluster_models)}
        if cluster_models
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
    dag = await (await validated_client(request.llm)).plan(request.question)
    return dag.model_dump(mode="json")


@router.post("/execute")
async def execute(request: ExecuteRequest) -> dict[str, Any]:
    run = await (await validated_client()).execute(parse_plan(request.plan_yaml))
    RUNS[run.run_id] = run
    payload = run.model_dump(mode="json")
    payload["mermaid"] = render_mermaid(run.plan, run.nodes)
    return payload


@router.post("/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    try:
        run = await (await validated_client(request.llm)).ask(request.question)
    except PlannerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    RUNS[run.run_id] = run
    payload = run.model_dump(mode="json")
    payload["mermaid"] = render_mermaid(run.plan, run.nodes)
    return payload


@router.post("/ask/live")
async def ask_live(request: AskRequest) -> dict[str, Any]:
    run_id = str(uuid4())
    cfg = await validated_config_for_selection(request.llm)
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
    cfg = _config_for_benchmark_dataset(
        await validated_config_for_selection(request.llm),
        request.dataset,
    )
    dag_client = DagQaClient(cfg)
    data_path = _benchmark_subset_path(request)
    result = await benchmark_dataset(
        dag_client,
        dataset=request.dataset,
        system=systems[0],
        limit=request.limit,
        seed=request.seed,
        path=data_path,
        subset=request.subset,
        name=_benchmark_name(request),
    )
    result = _write_benchmark_result(result)
    return result


@router.get(
    "/benchmarks/hotpotqa",
    tags=["Benchmarks"],
    summary="Get HotpotQA benchmark metadata",
)
def hotpotqa_info(
    data_path: str | None = None,
    subset: str = "validation",
    dataset: BenchmarkDatasetId = "hotpotqa",
) -> dict[str, Any]:
    return hotpotqa_meta(data_path, subset, dataset)


@router.post(
    "/benchmarks/hotpotqa/live",
    tags=["Benchmarks"],
    summary="Start live HotpotQA benchmark",
)
async def hotpotqa_live(request: BenchmarkRequest) -> dict[str, Any]:
    run_id = str(uuid4())
    systems = _benchmark_systems(request)
    comparison_group_id = str(uuid4()) if len(systems) == PAIRED_SYSTEM_COUNT else None
    cfg = _config_for_benchmark_dataset(
        await validated_config_for_selection(request.llm),
        request.dataset,
    )
    dag_client = DagQaClient(cfg)
    seed = request.seed if request.seed is not None else int(time.time_ns() % 2_147_483_647)
    data_path = _benchmark_subset_path(request)
    dataset_info = get_benchmark_dataset(request.dataset)
    dataset_size = count_benchmark_examples(request.dataset, request.subset, data_path)
    state = {
        "run_id": run_id,
        "name": _benchmark_name(request),
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
        "dataset": dataset_info.id,
        "dataset_label": dataset_info.label,
        "split": dataset_info.split,
        "subset": request.subset,
        "subset_label": _benchmark_subset_label(request.subset, request.dataset),
        "dataset_size": dataset_size,
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
        "request": request.model_dump(mode="json"),
        "completed_result_payloads": [],
        "stop_requested": False,
        "resumable": True,
    }
    _store_live_benchmark(state)
    stop_event = asyncio.Event()
    LIVE_BENCHMARK_STOPS[run_id] = stop_event
    LIVE_BENCHMARK_TASKS[run_id] = asyncio.create_task(
        _run_live_benchmark(
            run_id,
            request,
            seed,
            dag_client,
            cfg,
            systems,
            comparison_group_id,
            stop_event,
        )
    )
    return LIVE_BENCHMARKS[run_id]


@router.get(
    "/benchmarks/hotpotqa/live",
    tags=["Benchmarks"],
    summary="List unfinished live HotpotQA benchmarks",
)
def list_live_benchmarks() -> dict[str, Any]:
    live_dir = _live_benchmark_dir()
    live_dir.mkdir(parents=True, exist_ok=True)
    run_ids = {path.stem for path in live_dir.glob("*.json") if path.is_file()}
    run_ids.update(LIVE_BENCHMARKS.keys())

    runs = []
    for run_id in run_ids:
        try:
            state = get_live_benchmark(run_id)
        except Exception:
            continue
        if state.get("phase") == "complete":
            continue
        runs.append(_live_benchmark_summary(state))

    runs.sort(
        key=lambda item: (item.get("created_at") or "", item.get("run_id") or ""),
        reverse=True,
    )
    return {"runs": runs}


@router.post(
    "/benchmarks/hotpotqa/preflight",
    tags=["Benchmarks"],
    summary="Validate HotpotQA benchmark prerequisites",
)
async def hotpotqa_preflight(request: BenchmarkRequest) -> BenchmarkPreflightResult:
    checks: list[BenchmarkPreflightCheck] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(BenchmarkPreflightCheck(name=name, ok=ok, detail=detail))

    try:
        systems = _benchmark_systems(request)
        add("systems", True, f"Selected systems: {', '.join(systems)}.")
    except Exception as exc:
        add("systems", False, str(exc))

    if request.limit > 0:
        add("limit", True, f"Requested {request.limit} examples.")
    else:
        add("limit", False, "Benchmark limit must be greater than zero.")

    try:
        data_path = _benchmark_subset_path(request)
        dataset_size = await asyncio.to_thread(
            count_benchmark_examples,
            request.dataset,
            request.subset,
            data_path,
        )
        if request.limit > dataset_size:
            add(
                "dataset",
                False,
                f"Requested {request.limit} examples but dataset has {dataset_size}.",
            )
        else:
            dataset_label = get_benchmark_dataset(request.dataset).label
            add("dataset", True, f"{dataset_label} is readable with {dataset_size} examples.")
    except Exception as exc:
        add("dataset", False, f"Dataset could not be loaded: {exc}")

    try:
        cfg = _config_for_benchmark_dataset(
            await validated_config_for_selection(request.llm),
            request.dataset,
        )
        add(
            "configuration",
            True,
            (
                f"{cfg.llm.provider}/{cfg.llm.model}, timeout "
                f"{cfg.llm.request_timeout_seconds:g}s, retries {cfg.llm.max_retries}, "
                f"planner depth {cfg.planner.max_depth}."
            ),
        )
    except Exception as exc:
        cfg = None
        add("configuration", False, str(exc))

    output_dir = _benchmark_output_dir()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / f".preflight-{uuid4()}.tmp"
        probe.write_text("ok")
        probe.unlink()
        free_gb = shutil.disk_usage(output_dir).free / (1024**3)
        add("storage", free_gb >= 1.0, f"Output dir writable; {free_gb:.1f} GiB free.")
    except Exception as exc:
        add("storage", False, f"Output dir is not writable: {exc}")

    if cfg is not None and request.llm is not None and request.llm.provider == "cluster":
        try:
            models = await _cluster_models()
            add(
                "model_catalog",
                request.llm.model in models,
                (
                    "Selected cluster model is available."
                    if request.llm.model in models
                    else "Selected cluster model was not returned by the model catalog."
                ),
            )
        except Exception as exc:
            add("model_catalog", False, f"Cluster model catalog unavailable: {exc}")

    active = [
        run_id
        for run_id, state in LIVE_BENCHMARKS.items()
        if state.get("phase") in {"running", "stopping"}
    ]
    add(
        "active_runs",
        not active,
        "No active benchmark is running." if not active else f"Active benchmark: {active[0]}.",
    )

    return BenchmarkPreflightResult(ok=all(check.ok for check in checks), checks=checks)


@router.get(
    "/benchmarks/hotpotqa/live/{run_id}",
    tags=["Benchmarks"],
    summary="Get live HotpotQA benchmark status",
)
def get_live_benchmark(run_id: str) -> dict[str, Any]:
    if run_id not in LIVE_BENCHMARKS:
        persisted = _load_live_benchmark(run_id)
        if persisted is not None:
            LIVE_BENCHMARKS[run_id] = persisted
    if run_id not in LIVE_BENCHMARKS:
        raise HTTPException(status_code=404, detail="Benchmark run not found.")
    state = LIVE_BENCHMARKS[run_id]
    task = LIVE_BENCHMARK_TASKS.get(run_id)
    if state.get("phase") in {"running", "stopping"} and (task is None or task.done()):
        state = {
            **state,
            "phase": "stopped",
            "status": "stopped",
            "error": "Benchmark runner is not active. Resume to continue from the checkpoint.",
            "stop_requested": False,
            "resumable": True,
        }
        _store_live_benchmark(state)
    return LIVE_BENCHMARKS[run_id]


@router.post(
    "/benchmarks/hotpotqa/live/{run_id}/stop",
    tags=["Benchmarks"],
    summary="Stop a live HotpotQA benchmark",
)
async def stop_live_benchmark(run_id: str) -> dict[str, Any]:
    state = get_live_benchmark(run_id)
    if state.get("phase") not in {"running", "stopping"}:
        return state
    stop_event = LIVE_BENCHMARK_STOPS.get(run_id)
    if stop_event is not None:
        stop_event.set()
    state = {**state, "phase": "stopping", "status": "stopping", "stop_requested": True}
    _store_live_benchmark(state)
    task = LIVE_BENCHMARK_TASKS.get(run_id)
    if task is not None and not task.done():
        task.cancel()
    return state


@router.post(
    "/benchmarks/hotpotqa/live/{run_id}/resume",
    tags=["Benchmarks"],
    summary="Resume a stopped or failed live HotpotQA benchmark",
)
async def resume_live_benchmark(
    run_id: str,
    request: BenchmarkResumeRequest | None = None,
) -> dict[str, Any]:
    state = get_live_benchmark(run_id)
    task = LIVE_BENCHMARK_TASKS.get(run_id)
    if task is not None and not task.done():
        return state
    if state.get("phase") == "complete":
        return state

    raw_request = state.get("request")
    if not isinstance(raw_request, dict):
        raise HTTPException(status_code=409, detail="Benchmark state is missing resume metadata.")
    benchmark_request = BenchmarkRequest.model_validate(raw_request)
    if request is not None and request.llm is not None:
        benchmark_request.llm = request.llm
    systems = _benchmark_systems(benchmark_request)
    cfg = _config_for_benchmark_dataset(
        await validated_config_for_selection(benchmark_request.llm),
        benchmark_request.dataset,
    )
    dag_client = DagQaClient(cfg)
    stop_event = asyncio.Event()
    LIVE_BENCHMARK_STOPS[run_id] = stop_event
    state = {
        **state,
        "phase": "running",
        "status": "running",
        "error": None,
        "stop_requested": False,
        "resumable": True,
    }
    _store_live_benchmark(state)
    LIVE_BENCHMARK_TASKS[run_id] = asyncio.create_task(
        _run_live_benchmark(
            run_id,
            benchmark_request,
            int(state.get("seed") or benchmark_request.seed or 0),
            dag_client,
            cfg,
            systems,
            state.get("comparison_group_id"),
            stop_event,
        )
    )
    return LIVE_BENCHMARKS[run_id]


async def _run_live_benchmark(  # noqa: PLR0912, PLR0915
    run_id: str,
    request: BenchmarkRequest,
    seed: int,
    dag_client: DagQaClient,
    cfg: AppConfig,
    systems: list[Literal["dag_agent", "direct_llm"]] | None = None,
    comparison_group_id: str | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    prior_state = LIVE_BENCHMARKS.get(run_id) or _load_live_benchmark(run_id) or {}
    elapsed_before_ms = float(prior_state.get("total_runtime_ms") or 0)
    started = time.perf_counter()
    records: list[BenchmarkRecord] = []
    systems = systems or _benchmark_systems(request)
    completed_results: list[BenchmarkResult] = [
        BenchmarkResult.model_validate(result)
        for result in prior_state.get("completed_result_payloads", [])
        if isinstance(result, dict)
    ]
    current_system = systems[0]
    created_at = prior_state.get("created_at") or LIVE_BENCHMARKS[run_id]["created_at"]
    benchmark_name = _benchmark_name(request)
    total_examples = request.limit * len(systems)
    dataset_size = 0
    data_path = _benchmark_subset_path(request)
    dataset_info = get_benchmark_dataset(request.dataset)
    last_record_completed_at = prior_state.get("last_record_completed_at")

    def update(
        phase: str = "running",
        status: str = "running",
        current_question: str | None = None,
        error: str | None = None,
        output_path: str | None = None,
    ) -> None:
        metrics = _aggregate(records)
        total_runtime_ms = elapsed_before_ms + (time.perf_counter() - started) * 1000
        if records:
            metrics["total_runtime_ms"] = total_runtime_ms
        prior_metrics = prior_state.get("metrics") if isinstance(prior_state, dict) else None
        completed_count = sum(len(result.records) for result in completed_results) + len(records)
        estimate = _live_benchmark_estimate(
            systems=systems,
            current_system=current_system,
            per_system_total=request.limit,
            current_records=records,
            completed_results=completed_results,
            elapsed_ms=total_runtime_ms,
            max_parallel_examples=cfg.benchmark.max_parallel_examples,
            historical_metrics=prior_metrics,
        )
        state = {
            "run_id": run_id,
            "name": benchmark_name,
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
            "dataset": dataset_info.id,
            "dataset_label": dataset_info.label,
            "split": dataset_info.split,
            "subset": request.subset,
            "subset_label": _benchmark_subset_label(request.subset, request.dataset),
            "dataset_size": dataset_size or None,
            "provider": dag_client.config.llm.provider,
            "model": dag_client.config.llm.model,
            "created_at": created_at,
            "completed": completed_count,
            "total": total_examples,
            "current_question": current_question,
            "total_runtime_ms": total_runtime_ms,
            "estimate": estimate,
            "metrics": metrics,
            "records": [record.model_dump(mode="json") for record in records],
            "output_path": output_path,
            "error": error,
            "last_record_completed_at": last_record_completed_at,
            "request": request.model_dump(mode="json"),
            "completed_result_payloads": [
                result.model_dump(mode="json") for result in completed_results
            ],
            "stop_requested": stop_event.is_set() if stop_event is not None else False,
            "resumable": phase in {"running", "stopping", "stopped", "error"},
        }
        _store_live_benchmark(state)

    try:
        if request.limit <= 0 and data_path is None:
            dataset_size = count_benchmark_examples(request.dataset, request.subset)
            all_examples = []
        else:
            all_examples = await asyncio.to_thread(
                load_benchmark_examples,
                request.dataset,
                request.subset,
                data_path,
            )
            dataset_size = len(all_examples)
        sampled_examples = _sample_examples(all_examples, request.limit, seed)
        per_system_total = len(sampled_examples)
        total_examples = per_system_total * len(systems)

        for system_index, system in enumerate(systems):
            if stop_event is not None and stop_event.is_set():
                break
            prior_result = next(
                (result for result in completed_results if result.system == system),
                None,
            )
            if prior_result is not None and len(prior_result.records) >= per_system_total:
                continue
            system_started = time.perf_counter()
            current_system = system
            if prior_result is not None:
                records = list(prior_result.records)
            elif prior_state.get("current_system") == system:
                records = [
                    BenchmarkRecord.model_validate(record)
                    for record in prior_state.get("records", [])
                    if isinstance(record, dict)
                ]
            else:
                records = []
            completed_ids = {record.id for record in records}
            remaining_examples = [
                example for example in sampled_examples if example.id not in completed_ids
            ]
            update(
                current_question=(
                    f"Running {system.replace('_', ' ')} with up to "
                    f"{min(cfg.benchmark.max_parallel_examples, len(remaining_examples))} examples "
                    "in parallel."
                    if remaining_examples
                    else None
                )
            )

            def record_completed(
                record: BenchmarkRecord,
                records_ref: list[BenchmarkRecord] = records,
                system_name: str = system,
            ) -> None:
                nonlocal last_record_completed_at
                records_ref.append(record)
                last_record_completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                _append_live_record(run_id, system_name, record)
                remaining = per_system_total - len(records_ref)
                stopping = stop_event is not None and stop_event.is_set()
                update(
                    phase="stopping" if stopping else "running",
                    status="stopping" if stopping else "running",
                    current_question=(
                        f"{remaining} {system_name.replace('_', ' ')} examples remaining."
                        if remaining
                        else None
                    ),
                )

            if stop_event is None:
                new_records = await _run_examples(
                    dag_client,
                    remaining_examples,
                    system,
                    cfg.benchmark.max_parallel_examples,
                    record_completed,
                )
            else:
                new_records = await _run_examples(
                    dag_client,
                    remaining_examples,
                    system,
                    cfg.benchmark.max_parallel_examples,
                    record_completed,
                    stop_event.is_set,
                )
            records_by_id = {record.id: record for record in records}
            records_by_id.update({record.id: record for record in new_records})
            records = [
                records_by_id[example.id]
                for example in sampled_examples
                if example.id in records_by_id
            ]

            system_runtime_ms = (time.perf_counter() - system_started) * 1000
            metrics = _aggregate(records)
            metrics["total_runtime_ms"] = system_runtime_ms
            result = BenchmarkResult(
                run_id=run_id if len(systems) == 1 else str(uuid4()),
                name=benchmark_name,
                comparison_group_id=comparison_group_id,
                system=system,
                limit=request.limit,
                provider=dag_client.config.llm.provider,
                model=dag_client.config.llm.model,
                dataset=dataset_info.id,
                split=dataset_info.split,
                subset=request.subset,
                dataset_size=dataset_size,
                seed=seed,
                max_parallel_examples=cfg.benchmark.max_parallel_examples,
                created_at=created_at,
                total_runtime_ms=system_runtime_ms,
                records=records,
                metrics=metrics,
            )
            result = _write_benchmark_result(result)
            completed_results = [
                existing for existing in completed_results if existing.system != system
            ]
            completed_results.append(result)
            records = []
            if len(result.records) < per_system_total:
                update(
                    "stopped",
                    status="stopped",
                    output_path=result.output_path,
                    current_question="Benchmark stopped. Partial results were saved.",
                )
                return
            if system_index + 1 < len(systems):
                current_system = systems[system_index + 1]
                update(current_question=f"Starting {current_system.replace('_', ' ')}.")

        update(
            "complete",
            status="succeeded",
            output_path=completed_results[-1].output_path if completed_results else None,
        )
    except asyncio.CancelledError:
        update("stopped", status="stopped", error="Benchmark task was cancelled.")
        raise
    except Exception as exc:
        update("error", status="failed", error=str(exc))
    finally:
        LIVE_BENCHMARK_STOPS.pop(run_id, None)
        task = LIVE_BENCHMARK_TASKS.get(run_id)
        if task is not None and task.done():
            LIVE_BENCHMARK_TASKS.pop(run_id, None)


@router.get(
    "/benchmarks/hotpotqa/meta",
    tags=["Benchmarks"],
    summary="Get HotpotQA benchmark metadata",
)
def hotpotqa_meta(
    data_path: str | None = None,
    subset: str = "validation",
    dataset: BenchmarkDatasetId = "hotpotqa",
) -> dict[str, Any]:
    cfg = load_config()
    dataset_info = get_benchmark_dataset(dataset)
    resolved_path = data_path or dataset_info.subset(subset).path
    total_examples = count_benchmark_examples(dataset, subset, resolved_path)
    default_limit = _benchmark_default_limit(cfg, dataset, subset, data_path)
    return {
        "dataset": dataset_info.id,
        "dataset_label": dataset_info.label,
        "split": dataset_info.split,
        "subset": subset,
        "subset_label": _benchmark_subset_label(subset, dataset),
        "subsets": [
            {"id": subset_id, "label": details.label}
            for subset_id, details in dataset_info.subsets.items()
        ],
        "datasets": [
            {
                "id": item.id,
                "label": item.label,
                "default_subset": item.default_subset,
            }
            for item in DATASETS.values()
        ],
        "total_examples": total_examples,
        "default_limit": default_limit,
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
        "system": "dag_agent",
        "baselines": [
            {
                "label": _benchmark_subset_label(subset, dataset),
                "metric": "examples",
                "value": total_examples,
                "source": dataset_info.source if resolved_path is None else str(resolved_path),
            }
        ],
    }


def _live_benchmark_estimate(
    *,
    systems: list[Literal["dag_agent", "direct_llm"]],
    current_system: Literal["dag_agent", "direct_llm"],
    per_system_total: int,
    current_records: list[BenchmarkRecord],
    completed_results: list[BenchmarkResult],
    elapsed_ms: float,
    max_parallel_examples: int,
    historical_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records_by_system: dict[str, list[BenchmarkRecord]] = {
        result.system: list(result.records) for result in completed_results
    }
    records_by_system[current_system] = current_records
    all_records = [
        record for system_records in records_by_system.values() for record in system_records
    ]
    records_with_calls = [
        record
        for record in all_records
        if record.llm_call_count is not None and record.llm_call_count > 0
    ]
    total_call_count = sum(record.llm_call_count or 0 for record in records_with_calls)
    observed_call_ms = (
        sum(record.latency_ms for record in records_with_calls) / total_call_count
        if total_call_count
        else None
    )
    if observed_call_ms is None and historical_metrics:
        historical_avg_latency = historical_metrics.get("avg_latency_ms")
        historical_avg_calls = historical_metrics.get("avg_llm_call_count")
        if (
            isinstance(historical_avg_latency, int | float)
            and isinstance(historical_avg_calls, int | float)
            and historical_avg_latency > 0
            and historical_avg_calls > 0
        ):
            observed_call_ms = historical_avg_latency / historical_avg_calls
    llm_call_ms = observed_call_ms or 15_000.0

    dag_node_counts = [
        record.node_count
        for record in records_by_system.get("dag_agent", [])
        if record.node_count is not None and record.node_count > 0
    ]
    avg_dag_node_count = sum(dag_node_counts) / len(dag_node_counts) if dag_node_counts else 3.0
    system_index = systems.index(current_system) if current_system in systems else 0
    parallelism = max(max_parallel_examples, 1)
    remaining_by_system: dict[str, dict[str, float]] = {}
    total_remaining_call_work = 0.0

    for index, system in enumerate(systems):
        if index < system_index:
            completed = per_system_total
        else:
            completed = min(len(records_by_system.get(system, [])), per_system_total)
        remaining_examples = max(per_system_total - completed, 0)
        calls_per_example = 1.0 if system == "direct_llm" else avg_dag_node_count + 1.0
        call_work = remaining_examples * calls_per_example
        total_remaining_call_work += call_work
        remaining_by_system[system] = {
            "completed": float(completed),
            "remaining_examples": float(remaining_examples),
            "calls_per_example": calls_per_example,
            "remaining_call_work": call_work,
            "remaining_ms": (call_work * llm_call_ms) / parallelism,
        }

    remaining_ms = (total_remaining_call_work * llm_call_ms) / parallelism
    return {
        "elapsed_ms": elapsed_ms,
        "remaining_ms": remaining_ms,
        "total_ms": elapsed_ms + remaining_ms,
        "avg_llm_call_ms": llm_call_ms,
        "observed_llm_call_count": float(total_call_count),
        "avg_dag_node_count": avg_dag_node_count,
        "parallelism": float(parallelism),
        "remaining_by_system": remaining_by_system,
    }


def _benchmark_systems(
    request: BenchmarkRequest,
) -> list[Literal["dag_agent", "direct_llm"]]:
    systems = request.systems or [request.system]
    resolved = [system for system in ("dag_agent", "direct_llm") if system in systems]
    if not resolved:
        raise HTTPException(status_code=422, detail="Select at least one benchmark system.")
    return resolved


def _benchmark_name(request: BenchmarkRequest) -> str | None:
    name = (request.name or "").strip()
    return name or None


@router.get("/benchmarks/results", tags=["Benchmarks"], summary="List benchmark results")
def list_benchmark_results() -> dict[str, Any]:
    output_dir = _benchmark_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for path in output_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        normalized = _normalize_benchmark_payload(data, path)
        completed = normalized.get("metrics", {}).get("example_count") or len(
            normalized.get("records", [])
        )
        limit = normalized.get("limit")
        item = {
            "run_id": normalized.get("run_id") or path.stem,
            "name": normalized.get("name"),
            "comparison_group_id": normalized.get("comparison_group_id"),
            "created_at": normalized.get("created_at"),
            "dataset": normalized.get("dataset"),
            "dataset_label": normalized.get("dataset_label"),
            "split": normalized.get("split"),
            "subset": normalized.get("subset") or "validation",
            "subset_label": _benchmark_subset_label(
                normalized.get("subset") or "validation",
                normalized.get("dataset") or "hotpotqa",
            ),
            "system": normalized.get("system"),
            "provider": normalized.get("provider"),
            "model": normalized.get("model"),
            "limit": limit,
            "completed": completed,
            "partial": bool(limit and completed < limit),
            "seed": normalized.get("seed"),
            "metrics": normalized.get("metrics", {}),
            "path": str(path),
        }
        candidates.append(item)

    complete_keys = {
        _result_completion_key(item)
        for item in candidates
        if not item.get("partial") and _result_completion_key(item) is not None
    }
    items = [
        item
        for item in candidates
        if not (
            item.get("partial")
            and (key := _result_completion_key(item)) is not None
            and key in complete_keys
        )
    ]
    items.sort(
        key=lambda item: (item.get("created_at") or "", item.get("run_id") or ""),
        reverse=True,
    )
    return {"results": items}


def _result_completion_key(item: dict[str, Any]) -> tuple[Any, ...] | None:
    comparison_group_id = item.get("comparison_group_id")
    if not comparison_group_id:
        return None
    return (
        comparison_group_id,
        item.get("system"),
        item.get("seed"),
        item.get("model"),
        item.get("limit"),
        item.get("dataset") or "hotpotqa",
        item.get("subset") or "validation",
    )


@router.get(
    "/benchmarks/results/{run_id}",
    tags=["Benchmarks"],
    summary="Get benchmark result",
)
def get_benchmark_result(run_id: str) -> dict[str, Any]:
    _validate_storage_id(run_id)
    output_dir = _benchmark_output_dir()
    matches = list(output_dir.glob(f"*{run_id}*.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="Benchmark result not found.")
    path = matches[0]
    return _normalize_benchmark_payload(json.loads(path.read_text()), path)


@router.get(
    "/benchmarks/marked-rows",
    tags=["Benchmarks"],
    summary="List marked benchmark comparison rows",
)
def list_marked_benchmark_rows() -> dict[str, Any]:
    return {"rows": [row.model_dump(mode="json") for row in _load_marked_benchmark_rows()]}


@router.post(
    "/benchmarks/marked-rows",
    tags=["Benchmarks"],
    summary="Save a marked benchmark comparison row",
)
def save_marked_benchmark_row(row: MarkedBenchmarkRow) -> dict[str, Any]:
    _validate_storage_id(row.key)
    rows = [existing for existing in _load_marked_benchmark_rows() if existing.key != row.key]
    rows.insert(0, row)
    _store_marked_benchmark_rows(rows)
    return {"rows": [saved.model_dump(mode="json") for saved in rows]}


@router.delete(
    "/benchmarks/marked-rows/{row_key}",
    tags=["Benchmarks"],
    summary="Remove a marked benchmark comparison row",
)
def delete_marked_benchmark_row(row_key: str) -> dict[str, Any]:
    _validate_storage_id(row_key)
    rows = [row for row in _load_marked_benchmark_rows() if row.key != row_key]
    _store_marked_benchmark_rows(rows)
    return {"rows": [row.model_dump(mode="json") for row in rows]}


@router.post("/benchmarks/results/{run_id}/repair")
def repair_benchmark_result(run_id: str) -> dict[str, Any]:
    _validate_storage_id(run_id)
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


def _live_benchmark_dir() -> Path:
    return _benchmark_output_dir() / ".live"


def _marked_benchmark_rows_path() -> Path:
    return _benchmark_output_dir() / ".marked" / "rows.json"


def _load_marked_benchmark_rows() -> list[MarkedBenchmarkRow]:
    path = _marked_benchmark_rows_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    raw_rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(raw_rows, list):
        return []
    rows = []
    for raw_row in raw_rows:
        try:
            rows.append(MarkedBenchmarkRow.model_validate(raw_row))
        except Exception:
            continue
    return rows


def _store_marked_benchmark_rows(rows: list[MarkedBenchmarkRow]) -> None:
    _atomic_write_json(
        _marked_benchmark_rows_path(),
        {"rows": [row.model_dump(mode="json") for row in rows]},
    )


def _save_benchmark_result(result: BenchmarkResult) -> Path:
    output_dir = _benchmark_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = result.created_at.replace(":", "").replace("-", "")
    prefix = get_benchmark_dataset(result.dataset).result_prefix
    filename = f"{prefix}-{timestamp}-{result.run_id}.json"
    return output_dir / filename


def _write_benchmark_result(result: BenchmarkResult) -> BenchmarkResult:
    output_path = _save_benchmark_result(result)
    result.output_path = str(output_path)
    _atomic_write_json(output_path, result.model_dump(mode="json"))
    _cleanup_partial_benchmark_results(result, output_path)
    return result


def _cleanup_partial_benchmark_results(result: BenchmarkResult, output_path: Path) -> None:
    if len(result.records) < result.limit:
        return

    output_dir = _benchmark_output_dir()
    for path in output_dir.glob("*.json"):
        if path == output_path:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if not _is_superseded_partial_benchmark_result(data, result):
            continue
        try:
            path.unlink()
        except OSError:
            continue


def _is_superseded_partial_benchmark_result(
    data: dict[str, Any],
    result: BenchmarkResult,
) -> bool:
    return all(
        [
            data.get("system") == result.system,
            data.get("comparison_group_id") == result.comparison_group_id,
            data.get("seed") == result.seed,
            data.get("model") == result.model,
            data.get("limit") == result.limit,
            (data.get("dataset") or "hotpotqa") == result.dataset,
            (data.get("subset") or "validation") == result.subset,
            len(data.get("records", [])) < result.limit,
        ]
    )


def _live_benchmark_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": state.get("run_id"),
        "name": state.get("name"),
        "phase": state.get("phase"),
        "status": state.get("status"),
        "systems": state.get("systems") or [],
        "current_system": state.get("current_system"),
        "completed": state.get("completed") or 0,
        "total": state.get("total") or 0,
        "current_question": state.get("current_question"),
        "limit": state.get("limit"),
        "seed": state.get("seed"),
        "provider": state.get("provider"),
        "model": state.get("model"),
        "dataset": state.get("dataset") or "hotpotqa",
        "dataset_label": state.get("dataset_label"),
        "subset": state.get("subset") or "validation",
        "subset_label": state.get("subset_label")
        or _benchmark_subset_label(
            state.get("subset") or "validation",
            state.get("dataset") or "hotpotqa",
        ),
        "created_at": state.get("created_at"),
        "total_runtime_ms": state.get("total_runtime_ms") or 0,
        "estimate": state.get("estimate"),
        "last_record_completed_at": state.get("last_record_completed_at"),
        "error": state.get("error"),
        "resumable": state.get("phase") in {"stopped", "error"},
    }


def _live_benchmark_path(run_id: str) -> Path:
    _validate_storage_id(run_id)
    return _live_benchmark_dir() / f"{run_id}.json"


def _load_live_benchmark(run_id: str) -> dict[str, Any] | None:
    path = _live_benchmark_path(run_id)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
    except Exception:
        return None
    return _hydrate_live_benchmark_state(state)


def _store_live_benchmark(state: dict[str, Any]) -> None:
    run_id = str(state.get("run_id") or "")
    if not run_id:
        raise ValueError("Live benchmark state is missing run_id.")
    _validate_storage_id(run_id)
    LIVE_BENCHMARKS[run_id] = state
    _atomic_write_json(_live_benchmark_path(run_id), _checkpoint_live_benchmark_state(state))


def _checkpoint_live_benchmark_state(state: dict[str, Any]) -> dict[str, Any]:
    checkpoint = dict(state)
    records = checkpoint.get("records")
    if isinstance(records, list) and records:
        checkpoint["record_ids"] = [
            record.get("id") for record in records if isinstance(record, dict)
        ]
        checkpoint["records_path"] = str(
            _live_records_path(str(checkpoint["run_id"]), str(checkpoint["current_system"]))
        )
    checkpoint["records"] = []
    checkpoint["comparison_results"] = []
    checkpoint["completed_result_payloads"] = [
        {
            "run_id": result.get("run_id"),
            "name": result.get("name"),
            "system": result.get("system"),
            "output_path": result.get("output_path"),
        }
        for result in checkpoint.get("completed_result_payloads", [])
        if isinstance(result, dict)
    ]
    return checkpoint


def _hydrate_live_benchmark_state(state: dict[str, Any]) -> dict[str, Any]:
    hydrated = dict(state)
    current_system = hydrated.get("current_system")
    if isinstance(current_system, str):
        hydrated["records"] = [
            record.model_dump(mode="json")
            for record in _read_live_records(str(hydrated["run_id"]), current_system)
        ]
    completed_results = []
    for summary in hydrated.get("completed_result_payloads", []):
        if not isinstance(summary, dict):
            continue
        output_path = summary.get("output_path")
        if not isinstance(output_path, str):
            continue
        path = Path(output_path)
        if not path.exists():
            continue
        try:
            completed_results.append(BenchmarkResult.model_validate_json(path.read_text()))
        except Exception:
            continue
    hydrated["completed_result_payloads"] = [
        result.model_dump(mode="json") for result in completed_results
    ]
    hydrated["comparison_results"] = [
        result.model_dump(mode="json") for result in completed_results
    ]
    hydrated["comparison_run_ids"] = [result.run_id for result in completed_results]
    if completed_results and hydrated.get("phase") in {"stopped", "complete"}:
        hydrated["output_path"] = completed_results[-1].output_path
    return hydrated


def _live_records_path(run_id: str, system: str) -> Path:
    _validate_storage_id(run_id)
    return _live_benchmark_dir() / f"{run_id}-{system}.records.jsonl"


def _append_live_record(run_id: str, system: str, record: BenchmarkRecord) -> None:
    path = _live_records_path(run_id, system)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(record.model_dump_json() + "\n")


def _read_live_records(run_id: str, system: str) -> list[BenchmarkRecord]:
    path = _live_records_path(run_id, system)
    if not path.exists():
        return []
    records_by_id = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = BenchmarkRecord.model_validate_json(line)
        except Exception:
            continue
        records_by_id[record.id] = record
    return list(records_by_id.values())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def _validate_storage_id(value: str) -> None:
    if "/" in value or "\\" in value or ".." in value or not value:
        raise HTTPException(status_code=400, detail="Invalid run id.")


def _normalize_benchmark_payload(data: dict[str, Any], path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise HTTPException(status_code=404, detail="Benchmark result not found.")
    metrics = data.get("metrics") or data.get("summary") or {}
    records = [_normalize_benchmark_record(record) for record in data.get("records", [])]
    dataset_size = data.get("dataset_size") or data.get("total_examples")
    dataset = data.get("dataset") or "hotpotqa"
    subset = data.get("subset") or "validation"
    if dataset_size is None:
        try:
            dataset_size = count_benchmark_examples(dataset, subset)
        except Exception:
            dataset_size = len(records)
    return {
        **data,
        "run_id": data.get("run_id") or path.stem,
        "dataset": dataset,
        "dataset_label": data.get("dataset_label") or get_benchmark_dataset(dataset).label,
        "split": data.get("split") or "validation",
        "subset": subset,
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
        _atomic_write_json(path, data)

    return data


def _normalize_benchmark_record(record: dict[str, Any]) -> dict[str, Any]:
    structure = record.get("structure") or {}
    run_trace = _normalize_run_trace(record.get("run_trace"))
    return {
        "id": record.get("id") or str(record.get("index") or ""),
        "question": record.get("question") or "",
        "gold_answer": record.get("gold_answer") or "",
        "prediction": record.get("prediction") or "",
        "raw_prediction": record.get("raw_prediction"),
        "exact_match": record.get("exact_match") or 0,
        "f1": record.get("f1") or 0,
        "cosine_sim": record.get("cosine_sim") if record.get("cosine_sim") is not None else 0,
        "latency_ms": record.get("latency_ms") or 0,
        "llm_call_count": record.get("llm_call_count"),
        "llm_retry_count": record.get("llm_retry_count"),
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
