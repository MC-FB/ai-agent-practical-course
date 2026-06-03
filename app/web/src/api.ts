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
  validation: { valid: boolean; errors: string[] };
  duration_ms?: number;
  error?: string;
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

export type LiveRun = {
  run_id: string;
  question: string;
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

export async function ask(question: string): Promise<RunTrace> {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function startLiveAsk(question: string): Promise<LiveRun> {
  const response = await fetch("/api/ask/live", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
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
  node_count?: number | null;
  graph_depth?: number | null;
  structural_valid?: boolean | null;
  structural_issues?: string[];
  structural_failure: boolean;
  run_trace?: RunTrace | null;
  error?: string | null;
};

export type HotpotBenchmarkResult = {
  run_id: string;
  system: string;
  limit: number;
  model: string;
  dataset: string;
  split: string;
  dataset_size?: number | null;
  seed: number;
  created_at: string;
  output_path?: string | null;
  total_runtime_ms: number;
  records: HotpotBenchmarkRecord[];
  metrics: Record<string, number>;
};

export type LiveBenchmark = HotpotBenchmarkResult & {
  phase: "running" | "complete" | "error";
  status: string;
  completed: number;
  total: number;
  current_question?: string | null;
  error?: string | null;
};

export type SavedBenchmarkSummary = {
  run_id: string;
  created_at?: string | null;
  dataset?: string | null;
  split?: string | null;
  system?: string | null;
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
  seed?: number,
): Promise<HotpotBenchmarkResult> {
  const response = await fetch("/api/benchmarks/hotpotqa", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit, system, seed }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function startLiveBenchmark(
  limit: number,
  system: string,
  seed?: number,
): Promise<LiveBenchmark> {
  const response = await fetch("/api/benchmarks/hotpotqa/live", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit, system, seed }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getLiveBenchmark(runId: string): Promise<LiveBenchmark> {
  const response = await fetch(`/api/benchmarks/hotpotqa/live/${runId}`);
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
