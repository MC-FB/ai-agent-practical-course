from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from dagqa.client import DagQaClient
from dagqa.config import AppConfig
from dagqa.eval.benchmark import benchmark_hotpotqa, write_jsonl
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
    data_path: Annotated[Path | None, typer.Option("--data-path")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/local.yaml"),
) -> None:
    if dataset != "hotpotqa":
        raise typer.BadParameter("Only hotpotqa is supported.")
    result = asyncio.run(
        benchmark_hotpotqa(_client(config), system=system, limit=limit, path=data_path)
    )
    if output:
        write_jsonl(result, output)
    typer.echo(json.dumps(result.metrics, indent=2))


config_app = typer.Typer()
app.add_typer(config_app, name="config")


@config_app.command("validate")
def validate_config(path: Path) -> None:
    config = AppConfig.from_file(path)
    typer.echo(config.model_dump_json(indent=2))
