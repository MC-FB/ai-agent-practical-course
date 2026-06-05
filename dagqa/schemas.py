from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Operation(StrEnum):
    answer = "answer"
    transform = "transform"
    compare = "compare"
    synthesize = "synthesize"


class TaskType(StrEnum):
    fact_lookup = "fact_lookup"
    entity_resolution = "entity_resolution"
    date_lookup = "date_lookup"
    comparison = "comparison"
    calculation = "calculation"
    classification = "classification"
    synthesis = "synthesis"


class NodeStatus(StrEnum):
    pending = "pending"
    ready = "ready"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


class PromptSpec(BaseModel):
    system: str
    user_template: str


class DagNode(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    label: str
    task_type: TaskType
    question: str
    operation: Operation
    depends_on: list[str] = Field(default_factory=list)
    prompt: PromptSpec
    input_map: dict[str, str] = Field(default_factory=dict)
    child_output_policy: str | None = None
    output_schema: dict[str, Any]


class DagPlan(BaseModel):
    question: str
    nodes: list[DagNode]
    final_node: str


class LLMRequest(BaseModel):
    system: str
    prompt: str
    temperature: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class EvidenceDocument(BaseModel):
    id: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceSelection(BaseModel):
    strategy: str
    total_available: int
    documents: list[EvidenceDocument] = Field(default_factory=list)


class EvidenceCitation(BaseModel):
    document_id: str
    title: str
    sentence_indices: list[int] = Field(default_factory=list)
    fact: str


class GoldSupportingFact(BaseModel):
    title: str
    sentence_index: int


class EvidenceCitationEvaluation(BaseModel):
    citation: EvidenceCitation
    matches_gold: bool
    matched_gold_facts: list[GoldSupportingFact] = Field(default_factory=list)


class NodeTrace(BaseModel):
    node_id: str
    label: str
    task_type: TaskType
    operation: Operation
    status: NodeStatus
    dependency_values: dict[str, Any] = Field(default_factory=dict)
    resolved_question: str | None = None
    rendered_prompt: str | None = None
    raw_response: str | None = None
    parsed_output: dict[str, Any] | None = None
    returned_value: dict[str, Any] | None = None
    validation: ValidationResult = Field(default_factory=lambda: ValidationResult(valid=True))
    supporting_evidence: EvidenceSelection | None = None
    evidence_citations: list[EvidenceCitation] = Field(default_factory=list)
    evidence_citation_evaluations: list[EvidenceCitationEvaluation] = Field(default_factory=list)
    repair_attempts: int = 0
    llm_retry_count: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float | None = None
    error: str | None = None


class SchedulerWave(BaseModel):
    index: int
    node_ids: list[str]


class RunTrace(BaseModel):
    run_id: str
    question: str
    plan: DagPlan
    waves: list[SchedulerWave]
    nodes: list[NodeTrace]
    final_answer: dict[str, Any] | None = None
    status: NodeStatus
    total_duration_ms: float
    config: dict[str, Any] = Field(default_factory=dict)
