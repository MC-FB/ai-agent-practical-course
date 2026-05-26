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

export type RunTrace = {
  run_id: string;
  question: string;
  final_answer?: Record<string, unknown>;
  status: string;
  total_duration_ms: number;
  mermaid?: string;
  nodes: NodeTrace[];
  waves: { index: number; node_ids: string[] }[];
};

export type LiveRun = {
  run_id: string;
  question: string;
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

export async function benchmark(limit: number, system: string) {
  const response = await fetch("/api/benchmarks/hotpotqa", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit, system }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
