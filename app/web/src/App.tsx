import {
  Activity,
  ArrowLeft,
  BarChart3,
  BookOpen,
  CheckCircle2,
  Database,
  GitBranch,
  History,
  Loader2,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  RotateCcw,
  Send,
  Square,
  Timer,
} from "lucide-react";
import mermaid from "mermaid";
import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ApiError,
  getBenchmarkResult,
  getHotpotBenchmarkMeta,
  getLLMModels,
  getLiveBenchmark,
  getLiveAsk,
  listBenchmarkResults,
  preflightBenchmark,
  repairBenchmarkResult,
  resumeLiveBenchmark,
  startLiveBenchmark,
  startLiveAsk,
  stopLiveBenchmark,
  type DagNode,
  type HotpotBenchmarkMeta,
  type HotpotBenchmarkRecord,
  type HotpotBenchmarkResult,
  type LiveBenchmark,
  type LiveRun,
  type LLMModelCatalog,
  type LLMSelection,
  type NodeTrace,
  type RunTrace,
  type SavedBenchmarkSummary,
} from "./api";
import "./index.css";

mermaid.initialize({ startOnLoad: false, theme: "default", securityLevel: "loose" });

const SAMPLE_QUESTIONS = [
  "Which person was born earlier: Ada Lovelace or Alan Turing?",
  "Which city is farther north: the birthplace of Albert Einstein or the birthplace of Marie Curie?",
  "Who lived longer: the author of Pride and Prejudice or the author of Frankenstein?",
  "Which film was released earlier: the film that won Best Picture in 1994 or the film that won Best Picture in 2001?",
  "Which country has the larger population: the birthplace country of Nikola Tesla or the birthplace country of Isaac Newton?",
  "Which river is longer: the river that flows through Paris or the river that flows through London?",
  "Which university is older: the university attended by Barack Obama or the university attended by Bill Clinton?",
  "Which composer was born first: the composer of The Four Seasons or the composer of The Magic Flute?",
  "Which mountain is taller: the highest mountain in Japan or the highest mountain in Germany?",
  "Which company was founded earlier: the company that created the iPhone or the company that created Windows?",
];

const ACTIVE_BENCHMARK_STORAGE_KEY = "dagqa.activeBenchmarkRunId";
const LIVE_BENCHMARK_POLL_RETRY_LIMIT = 5;

type Tab = "chat" | "dataset" | "results";
type RunPhase = "idle" | "planning" | "executing" | "complete" | "error";
type AppRoute = {
  tab: Tab;
  runId?: string;
  recordId?: string;
};

function parseRoute(): AppRoute {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const tab: Tab = parts[0] === "dataset" || parts[0] === "results" ? parts[0] : "chat";
  return {
    tab,
    runId: tab === "results" ? parts[1] : undefined,
    recordId: tab === "results" ? parts[2] : tab === "dataset" ? parts[1] : undefined,
  };
}

function routeHash(route: AppRoute): string {
  const parts: string[] = [route.tab];
  if (route.tab === "results" && route.runId) parts.push(route.runId);
  if (route.recordId) parts.push(route.recordId);
  return `#/${parts.join("/")}`;
}

function navigate(route: AppRoute, replace = false) {
  const hash = routeHash(route);
  if (replace) {
    window.history.replaceState(null, "", hash);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else {
    window.location.hash = hash;
  }
}

function formatRawResponse(raw?: string) {
  if (!raw) return "";
  const fenceMatch = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = (fenceMatch ? fenceMatch[1] : raw).trim();
  try {
    return JSON.stringify(JSON.parse(body), null, 2);
  } catch {
    return body;
  }
}

function formatJson(value: unknown) {
  if (value === undefined || value === null) return "";
  return JSON.stringify(value, null, 2);
}

function schemaType(schema: unknown): string {
  if (!schema || typeof schema !== "object") return "unknown";
  const typed = schema as { type?: unknown; format?: unknown; items?: unknown };
  const type = Array.isArray(typed.type) ? typed.type.join(" | ") : typed.type;
  const suffix = typeof typed.format === "string" ? `:${typed.format}` : "";
  if (type === "array") return `array<${schemaType(typed.items)}>`;
  return typeof type === "string" ? `${type}${suffix}` : "unknown";
}

function expectedInputsForNode(planNode?: DagNode, planNodes: DagNode[] = []) {
  if (!planNode) return [];
  const byId = new Map(planNodes.map((node) => [node.id, node]));
  return Object.entries(planNode.input_map).map(([name, reference]) => {
    const [childId, field] = reference.split(".", 2);
    const child = byId.get(childId);
    const properties = child?.output_schema?.properties;
    const schema =
      properties && typeof properties === "object"
        ? (properties as Record<string, unknown>)[field]
        : undefined;
    return {
      name,
      reference,
      childLabel: child?.label ?? childId,
      type: schemaType(schema),
    };
  });
}

function formatPercent(value?: number) {
  if (value === undefined) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value?: number, digits = 0) {
  if (value === undefined || Number.isNaN(value)) return "-";
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatDuration(ms?: number) {
  if (ms === undefined) return "-";
  if (Math.abs(ms) < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

function formatSavedDateTime(value?: string | null) {
  if (!value) return "Unknown date";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const pad = (part: number) => String(part).padStart(2, "0");
  return `${pad(date.getDate())}.${pad(date.getMonth() + 1)}.${date.getFullYear()} ${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`;
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function GraphView({
  mermaidText,
  phase,
  onSelectNode,
}: {
  mermaidText?: string;
  phase: RunPhase;
  onSelectNode?: (nodeId: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const onSelectNodeRef = useRef(onSelectNode);

  useEffect(() => {
    onSelectNodeRef.current = onSelectNode;
  }, [onSelectNode]);

  useEffect(() => {
    const globalWindow = window as Window &
      typeof globalThis & {
        dagqaSelectGraphNode?: (nodeId: string) => void;
      };
    const handler = (nodeId: string) => {
      onSelectNodeRef.current?.(nodeId);
    };

    globalWindow.dagqaSelectGraphNode = handler;

    return () => {
      if (globalWindow.dagqaSelectGraphNode === handler) {
        delete globalWindow.dagqaSelectGraphNode;
      }
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function renderGraph() {
      if (!containerRef.current) {
        return;
      }
      if (!mermaidText) {
        containerRef.current.innerHTML = "";
        return;
      }

      try {
        const result = await mermaid.render("dag-graph", mermaidText);
        if (cancelled || !containerRef.current) {
          return;
        }
        containerRef.current.innerHTML = result.svg;
        result.bindFunctions?.(containerRef.current);
      } catch {
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = "";
        }
      }
    }

    void renderGraph();

    return () => {
      cancelled = true;
    };
  }, [mermaidText]);

  return (
    <section className="panel-section">
      <div className="section-header">
        <div>
          <h2>Reasoning Graph</h2>
          <span>
            {phase === "planning"
              ? "Creating DAG"
              : phase === "executing"
                ? "Executing planned node waves"
              : mermaidText
                ? "Generated DAG"
                : "No graph yet"}
          </span>
        </div>
        {phase === "planning" ? <Loader2 className="spin" size={18} /> : <GitBranch size={18} />}
      </div>
      <div className="graph-surface">
        {phase === "planning" ? (
          <div className="graph-loader">
            <Loader2 className="spin" size={26} />
            <strong>Building the reasoning graph</strong>
            <span>The selected model is creating a DAG plan. The graph will appear as soon as planning finishes.</span>
            <div className="progress-rail">
              <span />
            </div>
          </div>
        ) : mermaidText ? (
          <div ref={containerRef} />
        ) : (
          <div className="empty">Run a question to inspect the plan.</div>
        )}
      </div>
    </section>
  );
}

function TracePanel({
  nodes,
  selected,
  onSelect,
}: {
  nodes: NodeTrace[];
  selected?: string;
  onSelect: (node: NodeTrace) => void;
}) {
  return (
    <section className="panel-section">
      <div className="section-header">
        <div>
          <h2>Trace</h2>
          <span>{nodes.length ? `${nodes.length} nodes` : "No run loaded"}</span>
        </div>
        <Activity size={18} />
      </div>
      <div className="trace-list">
        {nodes.length === 0 ? (
          <div className="empty">Node outputs will appear here.</div>
        ) : (
          nodes.map((node) => (
            <button
              className={`trace-row ${selected === node.node_id ? "active" : ""}`}
              key={node.node_id}
              onClick={() => onSelect(node)}
            >
              <span className={`status-dot ${node.status}`} />
              <span>
                <strong>{node.label}</strong>
                <small>
                  {node.task_type} · {Math.round(node.duration_ms ?? 0)} ms
                </small>
              </span>
            </button>
          ))
        )}
      </div>
    </section>
  );
}

function NodeInspector({
  node,
  planNode,
  planNodes,
}: {
  node?: NodeTrace;
  planNode?: DagNode;
  planNodes: DagNode[];
}) {
  const hasInputMap = planNode?.input_map && Object.keys(planNode.input_map).length > 0;
  const expectedInputs = expectedInputsForNode(planNode, planNodes);
  const evidenceCitations = node?.evidence_citations ?? [];

  return (
    <section className="panel-section inspector-section">
      <div className="section-header">
        <div>
          <h2>Inspector</h2>
          <span>{node ? node.node_id : "Select a node"}</span>
        </div>
        {node?.validation.valid ? <CheckCircle2 size={18} /> : <GitBranch size={18} />}
      </div>
      {!node ? (
        <div className="empty">Select a trace row to inspect prompts and outputs.</div>
      ) : (
        <div className="inspector-grid">
          {planNode && (
            <div className="inspector-summary">
              <div>
                <label>Depends On</label>
                <div className="node-chip-row">
                  {planNode.depends_on.length ? (
                    planNode.depends_on.map((dependency) => (
                      <span className="node-chip" key={dependency}>
                        {dependency}
                      </span>
                    ))
                  ) : (
                    <span className="muted-value">None</span>
                  )}
                </div>
              </div>
              <div>
                <label>Operation</label>
                <span className="muted-value">{planNode.operation}</span>
              </div>
            </div>
          )}
          {planNode && (
            <div className="contract-panel">
              <div className="contract-heading">
                <label>Expected From Children</label>
                <span>{hasInputMap ? `${expectedInputs.length} values` : "No child values required"}</span>
              </div>
              {hasInputMap ? (
                <div className="mapping-list contract-list">
                  {expectedInputs.map(({ name, reference, childLabel, type }) => (
                    <div className="mapping-row" key={name}>
                      <div>
                        <code>{name}</code>
                        <small>{type}</small>
                      </div>
                      <span>-&gt;</span>
                      <div>
                        <code>{reference}</code>
                        <small>{childLabel}</small>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty compact-empty">No child inputs.</div>
              )}
            </div>
          )}
          {planNode?.child_output_policy && (
            <div>
              <label>Child Output Policy</label>
              <pre>{planNode.child_output_policy}</pre>
            </div>
          )}
          <div className="contract-panel">
            <div className="contract-heading">
              <label>Received Saturated Values</label>
              <span>{Object.keys(node.dependency_values).length} values</span>
            </div>
            <pre>{formatJson(node.dependency_values)}</pre>
          </div>
          <div>
            <label>Question</label>
            <pre>{node.resolved_question ?? ""}</pre>
          </div>
          <div className="contract-panel">
            <div className="contract-heading">
              <label>Supporting Evidence Given To Model</label>
              <span>
                {node.supporting_evidence
                  ? `${node.supporting_evidence.documents.length} of ${node.supporting_evidence.total_available} documents · ${node.supporting_evidence.strategy}`
                  : "No evidence supplied"}
              </span>
            </div>
            {node.supporting_evidence?.documents.length ? (
              <div className="evidence-list">
                {node.supporting_evidence.documents.map((document, index) => (
                  <details className="evidence-document" key={document.id}>
                    <summary>
                      <span>{index + 1}</span>
                      <strong>{document.title || "Untitled document"}</strong>
                      <code>{document.id}</code>
                    </summary>
                    <pre>{document.text}</pre>
                    {Object.keys(document.metadata).length > 0 && (
                      <div className="evidence-metadata">
                        <label>Metadata</label>
                        <pre>{formatJson(document.metadata)}</pre>
                      </div>
                    )}
                  </details>
                ))}
              </div>
            ) : (
              <div className="empty compact-empty">This node did not receive supporting evidence.</div>
            )}
          </div>
          <div className="contract-panel">
            <div className="contract-heading">
              <label>Evidence Facts Used By Model</label>
              <span>
                {evidenceCitations.length
                  ? `${evidenceCitations.length} citations`
                  : "No citations returned"}
              </span>
            </div>
            {evidenceCitations.length ? (
              <div className="citation-list">
                {evidenceCitations.map((citation, index) => {
                  const evaluation = node.evidence_citation_evaluations?.[index];
                  return (
                    <div className="citation-card" key={`${citation.document_id}-${index}`}>
                      <div className="citation-heading">
                        <strong>{citation.fact}</strong>
                        {evaluation && (
                          <span className={evaluation.matches_gold ? "citation-gold" : "citation-non-gold"}>
                            {evaluation.matches_gold ? "Matches gold" : "Non-gold citation"}
                          </span>
                        )}
                      </div>
                      <div className="citation-source">
                        <code>{citation.document_id}</code>
                        <span>{citation.title}</span>
                        <span>
                          sentences{" "}
                          {citation.sentence_indices.length
                            ? citation.sentence_indices.join(", ")
                            : "not specified"}
                        </span>
                      </div>
                      {evaluation?.matched_gold_facts.length ? (
                        <small>
                          Gold match:{" "}
                          {evaluation.matched_gold_facts
                            .map((fact) => `${fact.title} (${fact.sentence_index})`)
                            .join(", ")}
                        </small>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="empty compact-empty">The model did not report evidence facts.</div>
            )}
          </div>
          {planNode && (
            <div>
              <label>User Template</label>
              <pre>{planNode.prompt.user_template}</pre>
            </div>
          )}
          <div>
            <label>Rendered Prompt Sent To Model</label>
            <pre>{node.rendered_prompt ?? ""}</pre>
          </div>
          {planNode && (
            <div className="contract-panel">
              <div className="contract-heading">
                <label>Return Format Expected By Parent</label>
                <span>JSON Schema</span>
              </div>
              <pre>{formatJson(planNode.output_schema)}</pre>
            </div>
          )}
          <div className="contract-panel">
            <div className="contract-heading">
              <label>Returned To Parent</label>
              <span>{node.validation.valid ? "Valid" : "Invalid"}</span>
            </div>
            <pre>{formatJson(node.returned_value)}</pre>
          </div>
          {!node.validation.valid && (
            <div>
              <label>Validation Errors</label>
              <pre>{formatJson(node.validation.errors)}</pre>
            </div>
          )}
          <div>
            <label>Raw Response</label>
            <pre>{formatRawResponse(node.raw_response)}</pre>
          </div>
        </div>
      )}
    </section>
  );
}

function ChatView({ llm }: { llm: LLMSelection }) {
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [run, setRun] = useState<LiveRun | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>();
  const [phase, setPhase] = useState<RunPhase>("idle");
  const [error, setError] = useState("");
  const busy = phase === "planning" || phase === "executing";
  const selectedNode = run?.nodes.find((node) => node.node_id === selectedNodeId) ?? run?.nodes[0];
  const selectedPlanNode = run?.plan?.nodes.find((node) => node.id === selectedNode?.node_id);

  const metrics = useMemo(
    () => ({
      status: busy ? "In Progress" : run?.status ?? "idle",
      nodes: run?.nodes.length ?? 0,
      waves: run?.waves.length ?? 0,
      runtime: busy ? "running" : `${Math.round(run?.total_duration_ms ?? 0)} ms`,
    }),
    [busy, run],
  );

  async function runAsk() {
    if (!question.trim()) return;
    setPhase("planning");
    setError("");
    setRun(null);
    setSelectedNodeId(undefined);
    try {
      const started = await startLiveAsk(question.trim(), llm);
      setRun(started);
      setPhase(started.phase);

      let current = started;
      while (current.phase === "planning" || current.phase === "executing") {
        await new Promise((resolve) => window.setTimeout(resolve, 700));
        current = await getLiveAsk(started.run_id);
        setRun(current);
        setPhase(current.phase);
        if (!selectedNodeId && current.nodes.length > 0) {
          setSelectedNodeId(current.nodes[0].node_id);
        }
      }

      if (current.phase === "error") {
        setError(current.error ?? "Request failed");
      } else {
        setSelectedNodeId(current.nodes[0]?.node_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
      setPhase("error");
    }
  }

  return (
    <main className="workspace">
      <section className="chat-column">
        <div className="chat-header">
          <div>
            <h1>Chat</h1>
            <p>Ask a multi-hop question and inspect how the DAG agent decomposes it.</p>
          </div>
          <span className="model-pill">{run?.model ?? llm.model}</span>
        </div>

        <div className="sample-select">
          <label htmlFor="sample-question">Example question</label>
          <select
            id="sample-question"
            value={SAMPLE_QUESTIONS.includes(question) ? question : ""}
            onChange={(event) => setQuestion(event.target.value)}
          >
            <option value="" disabled>
              Select an example
            </option>
            {SAMPLE_QUESTIONS.map((sample) => (
              <option key={sample} value={sample}>
                {sample}
              </option>
            ))}
          </select>
        </div>

        <div className="composer">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Paste a question..."
          />
          <button className="primary-button" disabled={busy || !question.trim()} onClick={runAsk}>
            {busy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
            {busy ? "Running" : "Ask"}
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}

        {busy && (
          <div className="run-progress">
            <Loader2 className="spin" size={18} />
            <div>
              <strong>In Progress</strong>
              <span>
                {phase === "planning"
                  ? "Creating the DAG plan."
                  : "Executing nodes and updating the graph as each call runs."}
              </span>
            </div>
          </div>
        )}

        <section className="answer-panel">
          <div className="section-header">
            <div>
              <h2>Answer</h2>
              <span>{run ? run.status : "Waiting for a question"}</span>
            </div>
            <MessageSquare size={18} />
          </div>
          <div className="answer-body">
            {busy ? (
              <div className="empty">Waiting for the final answer.</div>
            ) : run?.final_answer ? (
              <pre>{JSON.stringify(run.final_answer, null, 2)}</pre>
            ) : (
              <div className="empty">The final answer will appear here.</div>
            )}
          </div>
        </section>
      </section>

      <aside className="run-sidebar">
        <div className="metric-grid">
          <div className="metric">
            <label>Status</label>
            <strong>{metrics.status}</strong>
          </div>
          <div className="metric">
            <label>Nodes</label>
            <strong>{busy ? "-" : metrics.nodes}</strong>
          </div>
          <div className="metric">
            <label>Waves</label>
            <strong>{busy ? "-" : metrics.waves}</strong>
          </div>
          <div className="metric">
            <label>Runtime</label>
            <strong>{metrics.runtime}</strong>
          </div>
        </div>
        <GraphView
          mermaidText={run?.mermaid ?? undefined}
          phase={phase}
          onSelectNode={(nodeId) => setSelectedNodeId(nodeId)}
        />
        <TracePanel
          nodes={run?.nodes ?? []}
          selected={selectedNode?.node_id}
          onSelect={(node) => setSelectedNodeId(node.node_id)}
        />
        <NodeInspector
          node={selectedNode}
          planNode={selectedPlanNode}
          planNodes={run?.plan?.nodes ?? []}
        />
      </aside>
    </main>
  );
}

function BenchmarkResultView({
  result,
  totalExamples,
  onSelectRecord,
}: {
  result?: HotpotBenchmarkResult;
  totalExamples?: number;
  onSelectRecord?: (record: HotpotBenchmarkRecord) => void;
}) {
  const metrics = result?.metrics;
  const errorRecords = result?.records.filter((record) => record.error) ?? [];
  const sampleTotal = result?.dataset_size ?? totalExamples;

  if (!result || !metrics) {
    return <div className="empty">Metrics will appear here.</div>;
  }

  return (
    <div className="benchmark-result-grid">
      <div className="benchmark-metrics">
        <div className="metric">
          <label>Runtime</label>
          <strong>{formatDuration(result.total_runtime_ms)}</strong>
        </div>
        <div className="metric">
          <label>LLM Calls</label>
          <strong>{formatNumber(metrics.total_llm_call_count)}</strong>
        </div>
        <div className="metric">
          <label>Success</label>
          <strong>{formatNumber(metrics.success_count)}</strong>
        </div>
        <div className="metric">
          <label>Failures</label>
          <strong>{formatNumber(metrics.failure_count)}</strong>
        </div>
        <div className="metric">
          <label>Errors</label>
          <strong>{formatNumber(metrics.error_count)}</strong>
        </div>
        <div className="metric">
          <label>Exact Match</label>
          <strong>{formatPercent(metrics.exact_match)}</strong>
        </div>
        <div className="metric">
          <label>F1</label>
          <strong>{formatPercent(metrics.f1)}</strong>
        </div>
        <div className="metric">
          <label>Cosine Similarity</label>
          <strong>{formatNumber(metrics.cosine_sim, 3)}</strong>
        </div>

        <div className="metric">
          <label>Avg Latency</label>
          <strong>{formatDuration(metrics.avg_latency_ms)}</strong>
        </div>
        <div className="metric">
          <label>Avg Nodes</label>
          <strong>{formatNumber(metrics.avg_node_count, 1)}</strong>
        </div>
        <div className="metric">
          <label>Avg Depth</label>
          <strong>{formatNumber(metrics.avg_graph_depth, 1)}</strong>
        </div>
        <div className="metric">
          <label>Non-Gold Citations</label>
          <strong>{formatPercent(metrics.wrong_supporting_text_rate)}</strong>
        </div>
        <div className="metric">
          <label>Gold Fact Recall</label>
          <strong>{formatPercent(metrics.avg_gold_supporting_fact_recall)}</strong>
        </div>
      </div>
      <div className="benchmark-context">
        <div>
          <label>Provider</label>
          <strong>{result.provider ?? "Unknown"}</strong>
        </div>
        <div>
          <label>Model</label>
          <strong>{result.model}</strong>
        </div>
        <div>
          <label>Sample</label>
          <strong>
            {formatNumber(result.records.length)} of {formatNumber(sampleTotal)}
          </strong>
        </div>
        <div>
          <label>Seed</label>
          <strong>{result.seed}</strong>
        </div>
        <div>
          <label>Saved File</label>
          <span>{result.output_path ?? "Not saved"}</span>
        </div>
        <div>
          <label>Baseline Note</label>
          <span>
            Public HotpotQA leaderboards usually evaluate context-grounded systems, while this run
            uses the configured answer pipeline directly. Compare EM/F1 only when the setup is the
            same.
          </span>
        </div>
      </div>
      <div className="record-table-wrap">
        <table className="record-table">
          <thead>
            <tr>
              <th>Question</th>
              <th>Gold</th>
              <th>Prediction</th>
              <th>EM</th>
              <th>Cos Sim</th>
              <th>Status</th>
              <th>Graph</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {result.records.map((record) => {
              const canInspect = Boolean(onSelectRecord && record.run_trace);
              return (
                <tr
                  className={canInspect ? "clickable-row" : ""}
                  key={record.id}
                  onClick={canInspect ? () => onSelectRecord?.(record) : undefined}
                  onKeyDown={
                    canInspect
                      ? (event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            onSelectRecord?.(record);
                          }
                        }
                      : undefined
                  }
                  role={canInspect ? "button" : undefined}
                  tabIndex={canInspect ? 0 : undefined}
                >
                  <td>{record.question}</td>
                  <td>{record.gold_answer}</td>
                  <td>{record.prediction}</td>
                  <td>{formatPercent(record.exact_match)}</td>
                  <td>{formatNumber(record.cosine_sim, 3)}</td>
                  <td>{record.error ? "error" : record.structural_failure ? "failed" : "ok"}</td>
                  <td>
                    {canInspect ? (
                      <button
                        className="record-action"
                        onClick={(event) => {
                          event.stopPropagation();
                          onSelectRecord?.(record);
                        }}
                        type="button"
                      >
                        Inspect
                      </button>
                    ) : (
                      <span className="muted-value">No trace</span>
                    )}
                  </td>
                  <td>
                    {record.wrong_supporting_text_rate === null ||
                    record.wrong_supporting_text_rate === undefined
                      ? "-"
                      : `${formatPercent(record.wrong_supporting_text_rate)} non-gold`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {errorRecords.length > 0 && (
        <div>
          <label>Errors</label>
          <pre>{formatJson(errorRecords.map((record) => ({ id: record.id, error: record.error })))}</pre>
        </div>
      )}
    </div>
  );
}

const COMPARISON_METRICS = [
  ["exact_match", "Exact Match", "percent", "higher"],
  ["f1", "F1", "percent", "higher"],
  ["cosine_sim", "Cosine Similarity", "number", "higher"],
  ["avg_latency_ms", "Avg Latency", "duration", "lower"],
  ["avg_gold_supporting_fact_recall", "Gold Fact Recall", "percent", "higher"],
  ["wrong_supporting_text_rate", "Non-Gold Citations", "percent", "lower"],
] as const;

function systemLabel(system: string) {
  return system === "dag_agent" ? "DAG agent" : "Single prompt";
}

function runTypeLabel(system?: string | null) {
  return system === "dag_agent" ? "Multi-node DAG" : "Single-node prompt";
}

function benchmarkOptionLabel(item: SavedBenchmarkSummary | HotpotBenchmarkResult) {
  const parts = [
    item.name?.trim(),
    formatSavedDateTime(item.created_at),
    runTypeLabel(item.system),
    item.model ?? "unknown model",
    `${item.limit} examples`,
    `seed ${item.seed}`,
  ].filter(Boolean);
  return parts.join(" · ");
}

function comparisonValue(value: number | undefined, format: string) {
  if (format === "percent") return formatPercent(value);
  if (format === "duration") return formatDuration(value);
  return formatNumber(value, 3);
}

function deltaClass(value: number) {
  if (Math.abs(value) < 0.000001) return "neutral";
  return value > 0 ? "improved" : "degraded";
}

function BenchmarkComparisonView({
  first,
  second,
  onSelectRecord,
}: {
  first: HotpotBenchmarkResult;
  second: HotpotBenchmarkResult;
  onSelectRecord?: (run: HotpotBenchmarkResult, record: HotpotBenchmarkRecord) => void;
}) {
  const mixedSystems = first.system !== second.system;
  const focus = mixedSystems && second.system === "dag_agent" ? second : first;
  const reference = focus === first ? second : first;
  const focusLabel = mixedSystems ? "Multi-node DAG" : "Run A";
  const referenceLabel = mixedSystems ? "Single-node prompt" : "Run B";
  const deltaLabel = mixedSystems ? "Multi-node impact" : "Run A impact";
  const sameSeed = focus.seed === reference.seed;
  const referenceById = new Map(reference.records.map((record) => [record.id, record]));
  const aligned = sameSeed
    ? focus.records
        .map((record) => [record, referenceById.get(record.id)] as const)
        .filter((pair): pair is readonly [HotpotBenchmarkRecord, HotpotBenchmarkRecord] => Boolean(pair[1]))
    : [];
  const metricDeltas = COMPARISON_METRICS.map(([key, label, format, preference]) => {
    const focusValue = focus.metrics[key];
    const referenceValue = reference.metrics[key];
    const rawDelta = focusValue - referenceValue;
    const performanceDelta = preference === "higher" ? rawDelta : -rawDelta;
    return { key, label, format, focusValue, referenceValue, rawDelta, performanceDelta };
  });
  const improvements = metricDeltas.filter((metric) => metric.performanceDelta > 0.000001).length;
  const degradations = metricDeltas.filter((metric) => metric.performanceDelta < -0.000001).length;

  return (
    <div className="comparison-view">
      <div className="comparison-heading">
        <div>
          <span className="comparison-eyebrow">Performance overview</span>
          <strong>{mixedSystems ? "Multi-node DAG vs Single-node prompt" : `${systemLabel(first.system)} comparison`}</strong>
          <span>
            {sameSeed
              ? `Aligned question comparison, seed ${focus.seed}`
              : `Aggregate only: seeds differ (${focus.seed} vs ${reference.seed})`}
          </span>
        </div>
        <div className="comparison-statuses">
          <span className="comparison-score improved">{improvements} improvements</span>
          <span className="comparison-score degraded">{degradations} degradations</span>
          <span className={sameSeed ? "comparison-badge aligned" : "comparison-badge"}>
            {sameSeed ? `${aligned.length} aligned` : "Different seeds"}
          </span>
        </div>
      </div>
      <div className="comparison-legend">
        <span><i className="focus-dot" />{focusLabel}</span>
        <span><i className="reference-dot" />{referenceLabel}</span>
        <strong>{deltaLabel}</strong>
      </div>
      <div className="comparison-metrics">
        {metricDeltas.map((metric) => {
          const direction = deltaClass(metric.performanceDelta);
          return (
            <div className={`comparison-metric ${direction}`} key={metric.key}>
              <div className="comparison-metric-header">
                <label>{metric.label}</label>
                <strong>
                  {metric.rawDelta > 0 ? "+" : ""}
                  {comparisonValue(metric.rawDelta, metric.format)}
                </strong>
              </div>
              <div className="comparison-metric-values">
                <span>
                  <small>{focusLabel}</small>
                  {comparisonValue(metric.focusValue, metric.format)}
                </span>
                <span>
                  <small>{referenceLabel}</small>
                  {comparisonValue(metric.referenceValue, metric.format)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      {sameSeed && (
        <div className="record-table-wrap">
          <table className="record-table comparison-table">
            <thead>
              <tr>
                <th>Question</th>
                <th>Gold</th>
                <th>{focusLabel}</th>
                <th>{referenceLabel}</th>
                <th>{deltaLabel}</th>
              </tr>
            </thead>
            <tbody>
              {aligned.map(([focusRecord, referenceRecord]) => {
                const rowDelta = focusRecord.f1 - referenceRecord.f1;
                return (
                <tr key={focusRecord.id}>
                  <td>{focusRecord.question}</td>
                  <td>{focusRecord.gold_answer}</td>
                  <td>
                    <button className="comparison-answer" onClick={() => onSelectRecord?.(focus, focusRecord)}>
                      {focusRecord.prediction || "-"} · {formatPercent(focusRecord.f1)}
                    </button>
                  </td>
                  <td>
                    <button className="comparison-answer" onClick={() => onSelectRecord?.(reference, referenceRecord)}>
                      {referenceRecord.prediction || "-"} · {formatPercent(referenceRecord.f1)}
                    </button>
                  </td>
                  <td><span className={`row-delta ${deltaClass(rowDelta)}`}>{rowDelta > 0 ? "+" : ""}{formatPercent(rowDelta)}</span></td>
                </tr>
              )})}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function DatasetView({
  recordId,
  onSelectRecord,
  llm,
}: {
  recordId?: string;
  onSelectRecord: (recordId?: string) => void;
  llm: LLMSelection;
}) {
  const [limit, setLimit] = useState(5);
  const [benchmarkName, setBenchmarkName] = useState("");
  const [systems, setSystems] = useState<string[]>(["dag_agent", "direct_llm"]);
  const [seedInput, setSeedInput] = useState("");
  const [meta, setMeta] = useState<HotpotBenchmarkMeta>();
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<HotpotBenchmarkResult>();
  const [comparisonResults, setComparisonResults] = useState<HotpotBenchmarkResult[]>([]);
  const [liveRun, setLiveRun] = useState<LiveBenchmark>();
  const [error, setError] = useState("");
  const maxExamples = meta?.total_examples && meta.total_examples > 1 ? meta.total_examples : 7405;
  const resolvedLimit = Math.min(limit, maxExamples);
  const progressPercent = liveRun?.total ? (liveRun.completed / liveRun.total) * 100 : 0;
  const benchmarkRunning = liveRun?.phase === "running" || liveRun?.phase === "stopping";
  const canStop = liveRun?.phase === "running" || liveRun?.phase === "stopping";
  const canResume = liveRun?.phase === "stopped" || liveRun?.phase === "error";

  function updateLimit(value: number) {
    if (!Number.isFinite(value)) return;
    setLimit(Math.min(Math.max(Math.trunc(value), 1), maxExamples));
  }

  useEffect(() => {
    getHotpotBenchmarkMeta()
      .then((nextMeta) => {
        setMeta(nextMeta);
        setLimit(Math.min(nextMeta.default_limit, nextMeta.total_examples));
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load benchmark metadata"));
  }, []);

  useEffect(() => {
    const runId = window.localStorage.getItem(ACTIVE_BENCHMARK_STORAGE_KEY);
    if (!runId) return;
    const activeRunId = runId;
    let cancelled = false;

    async function restoreBenchmark() {
      try {
        setBusy(true);
        let restored: LiveBenchmark | undefined;
        for (let attempt = 1; attempt <= LIVE_BENCHMARK_POLL_RETRY_LIMIT; attempt += 1) {
          try {
            restored = await getLiveBenchmark(activeRunId);
            break;
          } catch (err) {
            if (err instanceof ApiError && err.status === 404) {
              window.localStorage.removeItem(ACTIVE_BENCHMARK_STORAGE_KEY);
              return;
            }
            if (attempt === LIVE_BENCHMARK_POLL_RETRY_LIMIT) {
              throw err;
            }
            if (!cancelled) {
              setError("Reconnecting to the running benchmark...");
            }
            await sleep(1000);
          }
        }
        if (cancelled) return;
        if (!restored) return;
        setError("");
        await watchBenchmark(restored, () => cancelled);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not restore benchmark");
        }
      } finally {
        if (!cancelled) setBusy(false);
      }
    }

    restoreBenchmark();
    return () => {
      cancelled = true;
    };
  }, []);

  async function watchBenchmark(started: LiveBenchmark, isCancelled = () => false) {
    setLiveRun(started);
    setResult(started.comparison_results?.[0] ?? started);
    setComparisonResults(started.comparison_results ?? []);
    setSeedInput(String(started.seed));
    setBenchmarkName(started.name ?? "");
    window.localStorage.setItem(ACTIVE_BENCHMARK_STORAGE_KEY, started.run_id);

    let current = started;
    let pollFailures = 0;
    while (!isCancelled() && (current.phase === "running" || current.phase === "stopping")) {
      await sleep(1000);
      try {
        current = await getLiveBenchmark(started.run_id);
        pollFailures = 0;
        setError("");
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          window.localStorage.removeItem(ACTIVE_BENCHMARK_STORAGE_KEY);
          throw err;
        }
        pollFailures += 1;
        if (pollFailures >= LIVE_BENCHMARK_POLL_RETRY_LIMIT) {
          throw err;
        }
        setError("Connection interrupted. Reconnecting to the running benchmark...");
        continue;
      }
      setLiveRun(current);
      setComparisonResults(current.comparison_results ?? []);
      setResult(current.comparison_results?.[0] ?? current);
    }

    if (isCancelled()) return;
    if (current.phase === "complete") {
      window.localStorage.removeItem(ACTIVE_BENCHMARK_STORAGE_KEY);
    }
    if (current.phase === "error") {
      setError(current.error ?? "Benchmark failed");
    }
  }

  async function runDatasetBenchmark() {
    setBusy(true);
    setError("");
    setLiveRun(undefined);
    setResult(undefined);
    setComparisonResults([]);
    onSelectRecord(undefined);
    try {
      const parsedSeed = seedInput.trim() ? Number(seedInput) : undefined;
      if (parsedSeed !== undefined && !Number.isInteger(parsedSeed)) {
        throw new Error("Seed must be an integer.");
      }
      if (!systems.length) {
        throw new Error("Select at least one system.");
      }
      const name = benchmarkName.trim() || undefined;
      const preflight = await preflightBenchmark(resolvedLimit, systems, llm, parsedSeed, name);
      if (!preflight.ok) {
        const failed = preflight.checks.filter((check) => !check.ok);
        throw new Error(failed.map((check) => check.detail).join(" "));
      }
      const started = await startLiveBenchmark(resolvedLimit, systems, llm, parsedSeed, name);
      await watchBenchmark(started);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed");
    } finally {
      setBusy(false);
    }
  }

  async function stopBenchmark() {
    if (!liveRun) return;
    setError("");
    try {
      const stopped = await stopLiveBenchmark(liveRun.run_id);
      setLiveRun(stopped);
      setComparisonResults(stopped.comparison_results ?? []);
      setResult(stopped.comparison_results?.[0] ?? stopped);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not stop benchmark");
    }
  }

  async function resumeBenchmark() {
    if (!liveRun) return;
    setBusy(true);
    setError("");
    try {
      const resumed = await resumeLiveBenchmark(liveRun.run_id, llm);
      await watchBenchmark(resumed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not resume benchmark");
    } finally {
      setBusy(false);
    }
  }

  const detailRecord = result?.records.find((record) => record.id === recordId);
  if (detailRecord) {
    return <BenchmarkRecordDetail record={detailRecord} onBack={() => onSelectRecord(undefined)} />;
  }

  return (
    <main className="dataset-workspace">
      <section className="dataset-panel">
        <div className="chat-header">
          <div>
            <h1>Dataset</h1>
            <p>Run a seeded HotpotQA validation sample against one or both systems.</p>
          </div>
          <Database size={22} />
        </div>

        <div className="benchmark-form">
          <label className="benchmark-name-field">
            Benchmark name
            <input
              placeholder="Optional"
              type="text"
              value={benchmarkName}
              onChange={(event) => setBenchmarkName(event.target.value)}
            />
          </label>
          <fieldset className="system-checks">
            <legend>Systems</legend>
            {[
              ["dag_agent", "DAG agent"],
              ["direct_llm", "Single prompt"],
            ].map(([value, label]) => (
              <label key={value}>
                <input
                  checked={systems.includes(value)}
                  type="checkbox"
                  onChange={(event) =>
                    setSystems((current) =>
                      event.target.checked
                        ? [...current, value]
                        : current.filter((system) => system !== value),
                    )
                  }
                />
                {label}
              </label>
            ))}
          </fieldset>
          <div className="benchmark-slider">
            <div className="slider-header">
              <label htmlFor="benchmark-limit">Examples</label>
              <div className="limit-input-wrap">
                <input
                  aria-label="Benchmark example count"
                  min={1}
                  max={maxExamples}
                  type="number"
                  value={resolvedLimit}
                  onChange={(event) => updateLimit(Number(event.target.value))}
                />
                <span>/ {formatNumber(maxExamples)}</span>
              </div>
            </div>
            <div className="range-row">
              <span>1</span>
              <input
                id="benchmark-limit"
                min={1}
                max={maxExamples}
                type="range"
                value={resolvedLimit}
                onChange={(event) => updateLimit(Number(event.target.value))}
              />
              <span>{formatNumber(maxExamples)}</span>
            </div>
          </div>
          <label className="benchmark-seed-field">
            Seed
            <input
              inputMode="numeric"
              placeholder="Random"
              type="number"
              value={seedInput}
              onChange={(event) => setSeedInput(event.target.value)}
            />
          </label>
          <button
            className="primary-button benchmark-run-button"
            disabled={busy || benchmarkRunning}
            onClick={runDatasetBenchmark}
          >
            {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            {busy ? "Running" : "Start benchmark"}
          </button>
          {liveRun && (
            <div className="benchmark-actions">
              <button disabled={!canStop} onClick={stopBenchmark}>
                <Square size={16} />
                Stop
              </button>
              <button disabled={!canResume || busy} onClick={resumeBenchmark}>
                <RotateCcw size={16} />
                Resume
              </button>
            </div>
          )}
        </div>

        {error && <div className="error-box">{error}</div>}

        {liveRun && (benchmarkRunning || liveRun.phase === "stopped" || liveRun.phase === "error") && (
          <div className="benchmark-progress">
            <div className="progress-summary">
              <div>
                <strong>
                  {formatNumber(liveRun.completed)} / {formatNumber(liveRun.total)} examples
                </strong>
                <span>
                  {liveRun.current_system ? `${systemLabel(liveRun.current_system)}: ` : ""}
                  {liveRun.current_question ?? "Finalizing benchmark run."}
                </span>
              </div>
              <span>{progressPercent.toFixed(0)}%</span>
            </div>
            <div className="progress-rail static-progress">
              <span style={{ width: `${progressPercent}%` }} />
            </div>
          </div>
        )}

        <section className="answer-panel">
          <div className="section-header">
            <div>
              <h2>Benchmark Result</h2>
              <span>
                {result
                  ? `${result.dataset} ${result.split}, seed ${result.seed}`
                  : meta
                    ? `${formatNumber(meta.total_examples)} HotpotQA examples available`
                    : "Loading metadata"}
              </span>
            </div>
            <BarChart3 size={18} />
          </div>
          <div className="answer-body benchmark-output">
            {comparisonResults.length === 2 ? (
              <BenchmarkComparisonView
                first={comparisonResults[0]}
                second={comparisonResults[1]}
                onSelectRecord={(run, record) => {
                  setResult(run);
                  onSelectRecord(record.id);
                }}
              />
            ) : (
              <BenchmarkResultView
                result={result}
                totalExamples={meta?.total_examples}
                onSelectRecord={(record) => onSelectRecord(record.id)}
              />
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

function BenchmarkRecordDetail({
  record,
  onBack,
}: {
  record: HotpotBenchmarkRecord;
  onBack: () => void;
}) {
  const run = record.run_trace;
  const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>(run?.nodes[0]?.node_id);
  const selectedNode = run?.nodes.find((node) => node.node_id === selectedNodeId) ?? run?.nodes[0];
  const selectedPlanNode = run?.plan?.nodes.find((node) => node.id === selectedNode?.node_id);

  if (!run) {
    return (
      <main className="dataset-workspace">
        <section className="dataset-panel">
          <button className="secondary-button" onClick={onBack}>
            <ArrowLeft size={18} />
            Back
          </button>
          <div className="empty">This saved row does not contain a graph trace.</div>
        </section>
      </main>
    );
  }

  return (
    <main className="workspace detail-workspace">
      <section className="chat-column">
        <div className="chat-header">
          <div>
            <h1>Benchmark Detail</h1>
            <p>{record.question}</p>
          </div>
          <button className="secondary-button" onClick={onBack}>
            <ArrowLeft size={18} />
            Back
          </button>
        </div>

        <div className="metric-grid">
          <div className="metric">
            <label>Status</label>
            <strong>{run.status}</strong>
          </div>
          <div className="metric">
            <label>Gold</label>
            <strong>{record.gold_answer}</strong>
          </div>
          <div className="metric">
            <label>Prediction</label>
            <strong>{record.prediction || "-"}</strong>
          </div>
          <div className="metric">
            <label>F1</label>
            <strong>{formatPercent(record.f1)}</strong>
          </div>
          <div className="metric">
            <label>Non-Gold Citations</label>
            <strong>{formatPercent(record.wrong_supporting_text_rate ?? undefined)}</strong>
          </div>
          <div className="metric">
            <label>Gold Fact Recall</label>
            <strong>{formatPercent(record.gold_supporting_fact_recall ?? undefined)}</strong>
          </div>
        </div>

        <section className="answer-panel">
          <div className="section-header">
            <div>
              <h2>Gold Supporting Facts</h2>
              <span>{record.gold_supporting_facts?.length ?? 0} labeled sentences</span>
            </div>
            <BookOpen size={18} />
          </div>
          <div className="answer-body">
            <pre>{formatJson(record.gold_supporting_facts ?? [])}</pre>
          </div>
        </section>

        {record.structural_issues && record.structural_issues.length > 0 && (
          <section className="answer-panel">
            <div className="section-header">
              <div>
                <h2>Structural Issues</h2>
                <span>{record.structural_issues.length} issues</span>
              </div>
              <GitBranch size={18} />
            </div>
            <div className="answer-body">
              <pre>{formatJson(record.structural_issues)}</pre>
            </div>
          </section>
        )}

        <section className="answer-panel">
          <div className="section-header">
            <div>
              <h2>Final Answer</h2>
              <span>{run.final_answer ? "Returned by final node" : "No final answer"}</span>
            </div>
            <MessageSquare size={18} />
          </div>
          <div className="answer-body">
            {run.final_answer ? (
              <pre>{formatJson(run.final_answer)}</pre>
            ) : (
              <div className="empty">No final answer was returned.</div>
            )}
          </div>
        </section>
      </section>

      <aside className="run-sidebar">
        <GraphView
          mermaidText={run.mermaid ?? undefined}
          phase="complete"
          onSelectNode={(nodeId) => setSelectedNodeId(nodeId)}
        />
        <TracePanel
          nodes={run.nodes}
          selected={selectedNode?.node_id}
          onSelect={(node) => setSelectedNodeId(node.node_id)}
        />
        <NodeInspector
          node={selectedNode}
          planNode={selectedPlanNode}
          planNodes={run.plan?.nodes ?? []}
        />
      </aside>
    </main>
  );
}

function ResultsView({
  runId,
  recordId,
  onNavigate,
}: {
  runId?: string;
  recordId?: string;
  onNavigate: (runId?: string, recordId?: string) => void;
}) {
  const [items, setItems] = useState<SavedBenchmarkSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState(runId ?? "");
  const [comparisonRunId, setComparisonRunId] = useState("");
  const [selectedResult, setSelectedResult] = useState<HotpotBenchmarkResult>();
  const [comparisonResult, setComparisonResult] = useState<HotpotBenchmarkResult>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function refreshResults(targetRunId = runId) {
    setBusy(true);
    setError("");
    try {
      const response = await listBenchmarkResults();
      setItems(response.results);
      setSelectedRunId(targetRunId ?? "");
      if (targetRunId) {
        const loaded = await getBenchmarkResult(targetRunId);
        setSelectedResult(loaded);
        const paired = response.results.find(
          (item) =>
            item.run_id !== targetRunId &&
            item.comparison_group_id &&
            item.comparison_group_id === loaded.comparison_group_id,
        );
        if (paired) {
          setComparisonRunId(paired.run_id);
          setComparisonResult(await getBenchmarkResult(paired.run_id));
        } else {
          setComparisonRunId("");
          setComparisonResult(undefined);
        }
      } else {
        setSelectedResult(undefined);
        setComparisonResult(undefined);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load benchmark results");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void refreshResults();
  }, [runId]);

  function selectResult(runId: string) {
    onNavigate(runId);
  }

  async function selectComparison(runId: string) {
    setComparisonRunId(runId);
    setComparisonResult(runId ? await getBenchmarkResult(runId) : undefined);
  }

  const detailRecord = selectedResult?.records.find((record) => record.id === recordId);
  if (detailRecord) {
    return <BenchmarkRecordDetail record={detailRecord} onBack={() => onNavigate(selectedRunId)} />;
  }

  return (
    <main className="dataset-workspace">
      <section className="dataset-panel">
        <div className="chat-header">
          <div>
            <h1>Results</h1>
            <p>Open previously saved HotpotQA benchmark runs.</p>
          </div>
          <History size={22} />
        </div>

        <div className="results-browser">
          <label>
            Run A
            <select
              value={selectedRunId}
              onChange={(event) => void selectResult(event.target.value)}
            >
              <option value="" disabled>
                Select a run
              </option>
              {items.map((item) => (
                <option key={item.run_id} value={item.run_id}>
                  {benchmarkOptionLabel(item)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Run B
            <select
              value={comparisonRunId}
              onChange={(event) => void selectComparison(event.target.value)}
            >
              <option value="">None</option>
              {items.filter((item) => item.run_id !== selectedRunId).map((item) => (
                <option key={item.run_id} value={item.run_id}>
                  {benchmarkOptionLabel(item)}
                </option>
              ))}
            </select>
          </label>
          <button className="primary-button" disabled={busy} onClick={() => void refreshResults()}>
            {busy ? <Loader2 className="spin" size={18} /> : <History size={18} />}
            Refresh
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}

        <section className="answer-panel">
          <div className="section-header">
            <div>
              <h2>Saved Benchmark</h2>
              <span>
            {selectedResult
              ? `${selectedResult.dataset} ${selectedResult.split}, seed ${selectedResult.seed}`
              : busy
                ? "Loading saved runs"
                : `${formatNumber(items.length)} saved runs`}
              </span>
            </div>
            <BarChart3 size={18} />
          </div>
          <div className="answer-body benchmark-output">
            {selectedResult && comparisonResult ? (
              <BenchmarkComparisonView
                first={selectedResult}
                second={comparisonResult}
                onSelectRecord={(run, record) => onNavigate(run.run_id, record.id)}
              />
            ) : (
              <BenchmarkResultView
                result={selectedResult}
                onSelectRecord={(record) => onNavigate(selectedRunId, record.id)}
              />
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

function ModelPicker({
  catalog,
  selected,
  error,
  onChange,
}: {
  catalog?: LLMModelCatalog;
  selected?: LLMSelection;
  error: string;
  onChange: (selection: LLMSelection) => void;
}) {
  const selectedValue = selected ? JSON.stringify(selected) : "";

  return (
    <div className="model-picker">
      <label htmlFor="llm-model">Execution model</label>
      <select
        id="llm-model"
        value={selectedValue}
        disabled={!catalog?.models.length}
        onChange={(event) => onChange(JSON.parse(event.target.value) as LLMSelection)}
      >
        {!catalog?.models.length && <option value="">Loading models...</option>}
        {catalog?.models.map((option) => (
          <option
            key={`${option.provider}:${option.model}`}
            value={JSON.stringify({ provider: option.provider, model: option.model })}
          >
            {option.label}
          </option>
        ))}
      </select>
      {catalog?.cluster_error && <small>Cluster unavailable: {catalog.cluster_error}</small>}
      {error && <small>{error}</small>}
    </div>
  );
}

function App() {
  const [route, setRoute] = useState<AppRoute>(() => parseRoute());
  const [modelCatalog, setModelCatalog] = useState<LLMModelCatalog>();
  const [selectedLLM, setSelectedLLM] = useState<LLMSelection>();
  const [modelError, setModelError] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => window.localStorage.getItem("dagqa-sidebar-collapsed") === "true",
  );

  useEffect(() => {
    const handleHashChange = () => setRoute(parseRoute());
    window.addEventListener("hashchange", handleHashChange);
    if (!window.location.hash) navigate({ tab: "chat" }, true);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  useEffect(() => {
    getLLMModels()
      .then((catalog) => {
        setModelCatalog(catalog);
        setSelectedLLM((current) => current ?? catalog.default);
      })
      .catch((err) => setModelError(err instanceof Error ? err.message : "Could not load models"));
  }, []);

  function toggleSidebar() {
    setSidebarCollapsed((current) => {
      const next = !current;
      window.localStorage.setItem("dagqa-sidebar-collapsed", String(next));
      return next;
    });
  }

  return (
    <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">DQ</div>
          <div className="sidebar-brand-copy">
            <strong>DAG QA</strong>
            <span>Multi-hop evaluation</span>
          </div>
          <button
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="sidebar-toggle"
            onClick={toggleSidebar}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            type="button"
          >
            {sidebarCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </button>
        </div>
        <nav className="tab-list">
          <button
            className={route.tab === "chat" ? "active" : ""}
            onClick={() => navigate({ tab: "chat" })}
            title="Chat"
          >
            <MessageSquare size={18} />
            <span>Chat</span>
          </button>
          <button
            className={route.tab === "dataset" ? "active" : ""}
            onClick={() => navigate({ tab: "dataset" })}
            title="Dataset"
          >
            <Database size={18} />
            <span>Dataset</span>
          </button>
          <button
            className={route.tab === "results" ? "active" : ""}
            onClick={() => navigate({ tab: "results" })}
            title="Results"
          >
            <History size={18} />
            <span>Results</span>
          </button>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" title="API Docs">
            <BookOpen size={18} />
            <span>API Docs</span>
          </a>
        </nav>
        <div className="sidebar-controls">
          <ModelPicker
            catalog={modelCatalog}
            selected={selectedLLM}
            error={modelError}
            onChange={setSelectedLLM}
          />
          <div className="sidebar-footer">
            <Timer size={16} />
            <span>Each run keeps its selected model.</span>
          </div>
        </div>
      </aside>
      {!selectedLLM ? (
        <main className="dataset-workspace">
          <div className="empty">Loading available LLM models.</div>
        </main>
      ) : route.tab === "chat" ? (
        <ChatView llm={selectedLLM} />
      ) : route.tab === "dataset" ? (
        <DatasetView
          recordId={route.recordId}
          onSelectRecord={(recordId) => navigate({ tab: "dataset", recordId })}
          llm={selectedLLM}
        />
      ) : (
        <ResultsView
          runId={route.runId}
          recordId={route.recordId}
          onNavigate={(runId, recordId) => navigate({ tab: "results", runId, recordId })}
        />
      )}
    </div>
  );
}

export default App;

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
