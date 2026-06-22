from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from dagqa.client import DagQaClient
from dagqa.config import AppConfig
from dagqa.eval.benchmark import benchmark_dataset
from dagqa.eval.datasets import get_benchmark_dataset
from dagqa.planning.parser import parse_plan

app = typer.Typer(no_args_is_help=True)


def _client(config: Path) -> DagQaClient:
    return DagQaClient(AppConfig.from_file(config))


@app.command()
def ask(
    question: str,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/local.yaml"),
) -> None:
    run = asyncio.run(_client(config).ask(question))
    typer.echo(run.model_dump_json(indent=2))


@app.command()
def plan(
    question: str,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/local.yaml"),
) -> None:
    dag = asyncio.run(_client(config).plan(question))
    payload = dag.model_dump_json(indent=2)
    if output:
        output.write_text(payload)
    typer.echo(payload)


@app.command()
def execute(
    plan_file: Path,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/local.yaml"),
) -> None:
    dag = parse_plan(plan_file.read_text())
    run = asyncio.run(_client(config).execute(dag))
    payload = run.model_dump_json(indent=2)
    if output:
        output.write_text(payload)
    typer.echo(payload)


@app.command()
def benchmark(
    dataset: str = "hotpotqa",
    limit: int = 100,
    system: str = "dag_agent",
    seed: int | None = None,
    subset: Annotated[str | None, typer.Option("--subset")] = None,
    max_parallel_examples: Annotated[int | None, typer.Option("--max-parallel-examples")] = None,
    delay_seconds: Annotated[float, typer.Option("--delay-seconds")] = 0.0,
    data_path: Annotated[Path | None, typer.Option("--data-path")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/local.yaml"),
) -> None:
    try:
        dataset_info = get_benchmark_dataset(dataset)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    subset = subset or dataset_info.default_subset

    completed = 0
    output_handle = None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output.open("w")

    def on_complete(record) -> None:
        nonlocal completed
        completed += 1
        if output_handle is not None:
            output_handle.write(json.dumps(record.model_dump(mode="json")) + "\n")
            output_handle.flush()
        status = "ok" if not record.structural_failure and not record.error else "failed"
        detail = f"em={record.exact_match:.0f} f1={record.f1:.2f}"
        if record.error:
            detail += f" error={record.error.splitlines()[0][:120]}"
        typer.echo(
            f"[{completed}/{limit}] {status} {record.id} {detail}",
            err=True,
        )

    try:
        result = asyncio.run(
            benchmark_dataset(
                _client(config),
                dataset=dataset,
                system=system,
                limit=limit,
                seed=seed,
                path=data_path,
                subset=subset,
                max_parallel_examples=max_parallel_examples,
                on_complete=on_complete,
                delay_between_examples_seconds=delay_seconds,
            )
        )
        if output_handle is not None:
            output_handle.write(
                json.dumps({"metrics": result.metrics, "system": result.system}) + "\n"
            )
            output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()
    typer.echo(json.dumps(result.metrics, indent=2))


config_app = typer.Typer()
app.add_typer(config_app, name="config")


@config_app.command("validate")
def validate_config(path: Path) -> None:
    config = AppConfig.from_file(path)
    typer.echo(config.model_dump_json(indent=2))
