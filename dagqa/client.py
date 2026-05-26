from __future__ import annotations

from pathlib import Path

from dagqa.config import AppConfig
from dagqa.graph.executor import DagExecutor
from dagqa.llm.base import LanguageModel
from dagqa.llm.factory import build_llm
from dagqa.planning.planner import Planner
from dagqa.schemas import DagPlan, RunTrace


class DagQaClient:
    def __init__(self, config: AppConfig, llm: LanguageModel | None = None) -> None:
        self.config = config
        self.llm = llm or build_llm(config.llm)
        self.planner = Planner(self.llm, config.planner, config.llm)
        self.executor = DagExecutor(self.llm, config)

    @classmethod
    def from_config(cls, path: str | Path) -> DagQaClient:
        return cls(AppConfig.from_file(path))

    async def plan(self, question: str) -> DagPlan:
        return await self.planner.plan(question)

    async def execute(self, plan: DagPlan) -> RunTrace:
        return await self.executor.execute(plan)

    async def ask(self, question: str) -> RunTrace:
        plan = await self.plan(question)
        return await self.execute(plan)
