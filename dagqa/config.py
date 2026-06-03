from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class LLMConfig(BaseModel):
    provider: Literal["gemini", "openrouter", "azure_openai", "cluster", "dspy"] = "gemini"
    model: str = "gemini/gemini-2.0-flash"
    model_env: str | None = None
    temperature: float = 0.0
    api_key_env: str | None = "GEMINI_API_KEY"
    api_base: str | None = None
    api_base_env: str | None = None
    api_version: str | None = None
    api_version_env: str | None = None


class PlannerConfig(BaseModel):
    max_nodes: int = Field(default=10, ge=1)
    max_depth: int = Field(default=3, ge=1)
    repair_rounds: int = Field(default=1, ge=0)


class ExecutionConfig(BaseModel):
    max_parallel_nodes: int = Field(default=4, ge=1)
    node_timeout_seconds: float = Field(default=120.0, gt=0)
    node_repair_rounds: int = Field(default=1, ge=0)
    fail_fast: bool = False


class ValidationConfig(BaseModel):
    structural_only: bool = True


class BenchmarkConfig(BaseModel):
    dataset: str = "hotpotqa"
    split: str = "validation"
    default_limit: int = Field(default=100, ge=1)
    output_dir: str = "runs/benchmarks"


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> AppConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
