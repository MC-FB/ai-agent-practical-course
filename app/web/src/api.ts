export type NodeTrace = {
  node_id: string;
  label: string;
  task_type: string;
  operation: string;
  status: string;
  dependency_values: Record<string, unknown>;
  resolved_question?: string;
  rendered_prompt?: string;
  raw_response?: string;
  parsed_output?: Record<string, unknown>;
  returned_value?: Record<string, unknown>;
  supporting_evidence?: EvidenceSelection | null;
  evidence_citations?: EvidenceCitation[];
  evidence_citation_evaluations?: EvidenceCitationEvaluation[];
  validation: { valid: boolean; errors: string[] };
  llm_retry_count?: number;
  duration_ms?: number;
  error?: string;
};

export type EvidenceDocument = {
  id: string;
  title: string;
  text: string;
  metadata: Record<string, unknown>;
};

export type EvidenceSelection = {
  strategy: string;
  total_available: number;
  documents: EvidenceDocument[];
};

export type EvidenceCitation = {
  document_id: string;
  title: string;
  sentence_indices: number[];
  fact: string;
};

export type GoldSupportingFact = {
  title: string;
  sentence_index: number;
};

export type EvidenceCitationEvaluation = {
  citation: EvidenceCitation;
  matches_gold: boolean;
  matched_gold_facts: GoldSupportingFact[];
};

export type DagNode = {
  id: string;
  label: string;
  task_type: string;
  operation: string;
  question: string;
  depends_on: string[];
  prompt: {
    system: string;
    user_template: string;
  };
  input_map: Record<string, string>;
  child_output_policy?: string | null;
  output_schema: Record<string, unknown>;
};

export type DagPlan = {
  question: string;
  nodes: DagNode[];
  final_node: string;
};

export type RunTrace = {
  run_id: string;
  question: string;
  plan?: DagPlan;
  final_answer?: Record<string, unknown> | null;
  status: string;
  total_duration_ms: number;
  mermaid?: string;
  nodes: NodeTrace[];
  waves: { index: number; node_ids: string[] }[];
};

export type LLMSelection = {
  provider: "azure_openai" | "cluster";
  model: string;
};

export type PlannerSelection = {
  max_nodes?: number;
  max_depth?: number;
};

export type AppConfigResponse = {
  planner: { max_nodes: number; max_depth: number };
  [key: string]: unknown;
};

export async function getAppConfig(): Promise<AppConfigResponse> {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function throwApiError(response: Response): Promise<never> {
  throw new ApiError(response.status, await response.text());
}

export type LLMModelOption = LLMSelection & {
  label: string;
};

export type LLMModelCatalog = {
  models: LLMModelOption[];
  default: LLMSelection;
  cluster_error?: string | null;
};

export type LiveRun = {
  run_id: string;
  question: string;
  provider: string;
  model: string;
  plan?: DagPlan | null;
  phase: "planning" | "executing" | "complete" | "error";
  status: string;
  final_answer?: Record<string, unknown> | null;
  total_duration_ms: number;
  mermaid?: string | null;
  nodes: NodeTrace[];
  waves: { index: number; node_ids: string[] }[];
  error?: string | null;
};

export async function ask(
  question: string,
  llm: LLMSelection,
  planner?: PlannerSelection,
): Promise<RunTrace> {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, llm, planner }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getLLMModels(): Promise<LLMModelCatalog> {
  const response = await fetch("/api/llm/models");
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function startLiveAsk(
  question: string,
  llm: LLMSelection,
  planner?: PlannerSelection,
): Promise<LiveRun> {
  const response = await fetch("/api/ask/live", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, llm, planner }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getLiveAsk(runId: string): Promise<LiveRun> {
  const response = await fetch(`/api/ask/live/${runId}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export type HotpotBenchmarkMeta = {
  dataset: string;
  split: string;
  total_examples: number;
  default_limit: number;
  provider?: string | null;
  model: string;
  system: string;
  baselines: Array<Record<string, unknown>>;
};

export type HotpotBenchmarkRecord = {
  id: string;
  question: string;
  gold_answer: string;
  prediction: string;
  exact_match: number;
  f1: number;
  cosine_sim: number;
  latency_ms: number;
  llm_call_count?: number | null;
  llm_retry_count?: number | null;
  node_count?: number | null;
  graph_depth?: number | null;
  structural_valid?: boolean | null;
  structural_issues?: string[];
  structural_failure: boolean;
  gold_supporting_facts?: GoldSupportingFact[];
  evidence_citation_count?: number | null;
  correct_evidence_citation_count?: number | null;
  wrong_evidence_citation_count?: number | null;
  wrong_supporting_text_rate?: number | null;
  gold_supporting_fact_recall?: number | null;
  run_trace?: RunTrace | null;
  error?: string | null;
};

export type HotpotBenchmarkResult = {
  run_id: string;
  name?: string | null;
  comparison_group_id?: string | null;
  system: string;
  limit: number;
  provider?: string | null;
  model: string;
  dataset: string;
  split: string;
  dataset_size?: number | null;
  seed: number;
  max_parallel_examples: number;
  max_nodes?: number | null;
  max_depth?: number | null;
  created_at: string;
  output_path?: string | null;
  total_runtime_ms: number;
  records: HotpotBenchmarkRecord[];
  metrics: Record<string, number>;
};

export type LiveBenchmark = HotpotBenchmarkResult & {
  phase: "running" | "stopping" | "stopped" | "complete" | "error";
  status: string;
  completed: number;
  total: number;
  current_question?: string | null;
  error?: string | null;
  systems?: string[];
  current_system?: string;
  comparison_run_ids?: string[];
  comparison_results?: HotpotBenchmarkResult[];
};

export type BenchmarkPreflightResult = {
  ok: boolean;
  checks: { name: string; ok: boolean; detail: string }[];
};

export type SavedBenchmarkSummary = {
  run_id: string;
  name?: string | null;
  comparison_group_id?: string | null;
  created_at?: string | null;
  dataset?: string | null;
  split?: string | null;
  system?: string | null;
  provider?: string | null;
  model?: string | null;
  limit?: number | null;
  seed?: number | null;
  metrics: Record<string, number>;
  path: string;
};

export async function getHotpotBenchmarkMeta(): Promise<HotpotBenchmarkMeta> {
  const response = await fetch("/api/benchmarks/hotpotqa/meta");
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function benchmark(
  limit: number,
  system: string,
  llm: LLMSelection,
  seed?: number,
  name?: string,
): Promise<HotpotBenchmarkResult> {
  const response = await fetch("/api/benchmarks/hotpotqa", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit, system, seed, name, llm }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function startLiveBenchmark(
  limit: number,
  systems: string[],
  llm: LLMSelection,
  seed?: number,
  name?: string,
  planner?: PlannerSelection,
): Promise<LiveBenchmark> {
  const response = await fetch("/api/benchmarks/hotpotqa/live", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit, systems, seed, name, llm, planner }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function preflightBenchmark(
  limit: number,
  systems: string[],
  llm: LLMSelection,
  seed?: number,
  name?: string,
  planner?: PlannerSelection,
): Promise<BenchmarkPreflightResult> {
  const response = await fetch("/api/benchmarks/hotpotqa/preflight", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit, systems, seed, name, llm, planner }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getLiveBenchmark(runId: string): Promise<LiveBenchmark> {
  const response = await fetch(`/api/benchmarks/hotpotqa/live/${runId}`);
  if (!response.ok) await throwApiError(response);
  return response.json();
}

export async function stopLiveBenchmark(runId: string): Promise<LiveBenchmark> {
  const response = await fetch(`/api/benchmarks/hotpotqa/live/${runId}/stop`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function resumeLiveBenchmark(runId: string, llm: LLMSelection): Promise<LiveBenchmark> {
  const response = await fetch(`/api/benchmarks/hotpotqa/live/${runId}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ llm }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function listBenchmarkResults(): Promise<{ results: SavedBenchmarkSummary[] }> {
  const response = await fetch("/api/benchmarks/results");
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getBenchmarkResult(runId: string): Promise<HotpotBenchmarkResult> {
  const response = await fetch(`/api/benchmarks/results/${runId}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function repairBenchmarkResult(runId: string): Promise<HotpotBenchmarkResult> {
  const response = await fetch(`/api/benchmarks/results/${runId}/repair`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
