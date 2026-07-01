from __future__ import annotations

from pathlib import Path

from dagqa.config import AppConfig
from dagqa.graph.executor import DagExecutor
from dagqa.llm.base import LanguageModel
from dagqa.llm.factory import build_llm
from dagqa.planning.planner import Planner
from dagqa.schemas import DagPlan, EvidenceDocument, RunTrace


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

    async def execute(
        self,
        plan: DagPlan,
        evidence_documents: list[EvidenceDocument] | None = None,
    ) -> RunTrace:
        return await self.executor.execute(plan, evidence_documents)

    async def ask(
        self,
        question: str,
        evidence_documents: list[EvidenceDocument] | None = None,
    ) -> RunTrace:
        """Combined LtM + DAG with self-consistency selection."""
        plan = await self.plan(question)
        if (
            evidence_documents
            and self.config.planner.planner_mode == "simple"
            and len(plan.nodes) > 1
        ):
            return await self.executor.execute_least_to_most(plan, evidence_documents)
        return await self.execute(plan, evidence_documents)

    async def ask_multi_hop(
        self,
        question: str,
        evidence_documents: list[EvidenceDocument] | None = None,
    ) -> RunTrace:
        """Pure node-by-node DAG execution (no LtM)."""
        plan = await self.plan(question)
        return await self.execute(plan, evidence_documents)

    async def ask_least_to_most(
        self,
        question: str,
        evidence_documents: list[EvidenceDocument] | None = None,
    ) -> RunTrace:
        """Pure Least-to-Most execution (no DAG fallback)."""
        plan = await self.plan(question)
        if evidence_documents and len(plan.nodes) > 1:
            return await self.executor.execute_least_to_most_only(plan, evidence_documents)
        return await self.execute(plan, evidence_documents)
