import {
  Activity,
  ArrowLeft,
  BarChart3,
  Bookmark,
  BookmarkCheck,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ClipboardCheck,
  ClipboardPlus,
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
import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Checkbox } from "./components/ui/checkbox";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Progress } from "./components/ui/progress";
import {
  ApiError,
  addAnnotation,
  deleteAnnotation,
  getAnnotationTable,
  saveAnnotationBand,
  listAnnotations,
  getAppConfig,
  deleteMarkedBenchmarkRow,
  getBenchmarkMeta,
  getBenchmarkResult,
  getLLMModels,
  getLiveBenchmark,
  getLiveAsk,
  listMarkedBenchmarkRows,
  listBenchmarkResults,
  listLiveBenchmarks,
  preflightBenchmark,
  repairBenchmarkResult,
  resumeLiveBenchmark,
  saveMarkedBenchmarkRow,
  startLiveBenchmark,
  startLiveAsk,
  stopLiveBenchmark,
  type BenchmarkSubset,
  type ChatStrategy,
  type DagNode,
  type BenchmarkMeta,
  type AnnotationTable,
  type HotpotBenchmarkRecord,
  type HotpotBenchmarkResult,
  type LiveBenchmark,
  type LiveBenchmarkSummary,
  type LiveRun,
  type LLMModelCatalog,
  type LLMSelection,
  type MarkedComparisonRow,
  type GoldSupportingFact,
  type NodeTrace,
  type PlannerSelection,
  type RunTrace,
  type SavedBenchmarkSummary,
} from "./api";
import "./index.css";

mermaid.initialize({ startOnLoad: false, theme: "default", securityLevel: "loose" });

const SAMPLE_QUESTIONS = [
  "Which happened first: the founding of the company that created the iPhone, or the death of the author of the novel that inspired Blade Runner?",
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
const LEGACY_MARKED_RECORDS_STORAGE_KEY = "dagqa.markedBenchmarkRecords";
const LIVE_BENCHMARK_POLL_RETRY_LIMIT = 5;

function parseOptionalInteger(value: string): number | undefined {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  if (trimmed.toLowerCase() === "random") return undefined;
  if (!/^-?\d+$/.test(trimmed)) {
    return undefined;
  }
  const parsed = Number(trimmed);
  if (!Number.isSafeInteger(parsed)) {
    return undefined;
  }
  return parsed;
}

function formatSeedInput(seed: number | null | undefined): string {
  return typeof seed === "number" && Number.isSafeInteger(seed) ? String(seed) : "";
}

const TREE_PARAM_BOUNDS = {
  max_nodes: { min: 1, max: 30 },
  max_depth: { min: 1, max: 8 },
};
// Shown until GET /api/config resolves; mirrors the PlannerConfig field defaults.
const TREE_PARAM_FALLBACK = { max_nodes: 10, max_depth: 3 };

type PlannerDefaults = { max_nodes: number; max_depth: number };

// Only send the fields the user changed; unchanged fields fall back to server config.
function buildPlannerOverride(
  current: PlannerDefaults,
  defaults: PlannerDefaults | undefined,
): PlannerSelection | undefined {
  if (!defaults) return undefined;
  const override: PlannerSelection = {};
  if (current.max_nodes !== defaults.max_nodes) override.max_nodes = current.max_nodes;
  if (current.max_depth !== defaults.max_depth) override.max_depth = current.max_depth;
  return Object.keys(override).length ? override : undefined;
}

// Fetches the active planner config, seeds the sliders from it, and derives the
// override payload to send (undefined when sliders match the server defaults).
function usePlannerSettings() {
  const [defaults, setDefaults] = useState<PlannerDefaults>();
  const [maxNodes, setMaxNodes] = useState(TREE_PARAM_FALLBACK.max_nodes);
  const [maxDepth, setMaxDepth] = useState(TREE_PARAM_FALLBACK.max_depth);

  useEffect(() => {
    let cancelled = false;
    getAppConfig()
      .then((config) => {
        if (cancelled) return;
        const next = {
          max_nodes: config.planner.max_nodes,
          max_depth: config.planner.max_depth,
        };
        setDefaults(next);
        setMaxNodes(next.max_nodes);
        setMaxDepth(next.max_depth);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const override = useMemo(
    () => buildPlannerOverride({ max_nodes: maxNodes, max_depth: maxDepth }, defaults),
    [maxNodes, maxDepth, defaults],
  );

  return { maxNodes, setMaxNodes, maxDepth, setMaxDepth, override };
}

function TreeParamSlider({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  function update(next: number) {
    if (!Number.isFinite(next)) return;
    onChange(Math.min(Math.max(Math.trunc(next), min), max));
  }
  return (
    <div className="grid gap-2">
      <div className="flex items-center justify-between gap-3">
        <Label>{label}</Label>
        <Input
          aria-label={label}
          className="h-9 w-20 bg-white text-right"
          min={min}
          max={max}
          type="number"
          value={value}
          onChange={(event) => update(Number(event.target.value))}
        />
      </div>
      <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 text-xs font-medium text-slate-500">
        <span>{min}</span>
        <input
          className="h-7 w-full accent-emerald-800"
          min={min}
          max={max}
          type="range"
          value={value}
          onChange={(event) => update(Number(event.target.value))}
        />
        <span>{max}</span>
      </div>
    </div>
  );
}

// Collapsible "dropdown" box exposing the planner's max_nodes / max_depth sliders.
function TreeParametersControl({
  maxNodes,
  maxDepth,
  onMaxNodesChange,
  onMaxDepthChange,
}: {
  maxNodes: number;
  maxDepth: number;
  onMaxNodesChange: (value: number) => void;
  onMaxDepthChange: (value: number) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <section className="panel-section">
      <div
        className="section-header cursor-pointer select-none"
        onClick={() => setOpen((value) => !value)}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") setOpen((v) => !v); }}
        role="button"
        tabIndex={0}
      >
        <div>
          <h2>Tree parameters</h2>
          <span>{maxNodes} nodes · depth {maxDepth}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <GitBranch size={18} />
          <ChevronDown className={`transition-transform ${open ? "rotate-180" : ""}`} size={16} />
        </div>
      </div>
      {open && (
        <div className="grid gap-4 p-3">
          <TreeParamSlider
            label="Max nodes"
            max={TREE_PARAM_BOUNDS.max_nodes.max}
            min={TREE_PARAM_BOUNDS.max_nodes.min}
            onChange={onMaxNodesChange}
            value={maxNodes}
          />
          <TreeParamSlider
            label="Max depth"
            max={TREE_PARAM_BOUNDS.max_depth.max}
            min={TREE_PARAM_BOUNDS.max_depth.min}
            onChange={onMaxDepthChange}
            value={maxDepth}
          />
        </div>
      )}
    </section>
  );
}
const RECORD_TABLE_INITIAL_ROWS = 80;
const RECORD_TABLE_LOAD_ROWS = 120;

type Tab = "chat" | "dataset" | "results" | "marked" | "metrics";
type RunPhase = "idle" | "planning" | "executing" | "complete" | "error";
type AppRoute = {
  tab: Tab;
  runId?: string;
  recordId?: string;
  source?: "marked";
};

function parseRoute(): AppRoute {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const tab: Tab =
    parts[0] === "dataset" ||
    parts[0] === "results" ||
    parts[0] === "marked" ||
    parts[0] === "metrics"
      ? parts[0]
      : "chat";
  return {
    tab,
    runId: tab === "results" ? parts[1] : undefined,
    recordId: tab === "results" ? parts[2] : tab === "dataset" ? parts[1] : undefined,
    source: tab === "results" && parts[3] === "marked" ? "marked" : undefined,
  };
}

function routeHash(route: AppRoute): string {
  const parts: string[] = [route.tab];
  if (route.tab === "results" && route.runId) parts.push(route.runId);
  if (route.recordId) parts.push(route.recordId);
  if (route.source) parts.push(route.source);
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
  if (Math.abs(ms) < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  if (Math.abs(ms) < 3_600_000) return `${Math.round(ms / 60_000)} min`;
  const hours = Math.floor(ms / 3_600_000);
  const minutes = Math.round((ms % 3_600_000) / 60_000);
  return `${hours} h ${minutes} min`;
}

// Turn a snake_case metric key into a column header, e.g. "exact_match" -> "Exact Match".
function humanizeMetric(key: string) {
  return key
    .split("_")
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

// The per-record metric scores, falling back to the legacy named fields so runs saved before
// the metric_scores registry still render in the metric-scores table.
function legacyScores(record: HotpotBenchmarkRecord): Record<string, number> {
  return (
    record.metric_scores ?? {
      exact_match: record.exact_match,
      f1: record.f1,
      cosine_sim: record.cosine_sim,
    }
  );
}

// A toggleable column in the per-record results table.
type RecordColumn = {
  key: string;
  label: string;
  render: (record: HotpotBenchmarkRecord) => ReactNode;
};

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

function median(values: number[]) {
  if (!values.length) return undefined;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function useIncrementalRows(total: number, key: string) {
  const [visibleRows, setVisibleRows] = useState(RECORD_TABLE_INITIAL_ROWS);
  const sentinelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setVisibleRows(Math.min(RECORD_TABLE_INITIAL_ROWS, total));
  }, [key, total]);

  useEffect(() => {
    const node = sentinelRef.current;
    if (!node || visibleRows >= total) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisibleRows((current) => Math.min(current + RECORD_TABLE_LOAD_ROWS, total));
        }
      },
      { rootMargin: "240px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [total, visibleRows]);

  return {
    visibleRows: Math.min(visibleRows, total),
    sentinelRef,
  };
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
  goldAnswer,
  structuralIssues,
  goldSupportingFacts,
}: {
  node?: NodeTrace;
  planNode?: DagNode;
  planNodes: DagNode[];
  goldAnswer?: string;
  structuralIssues?: string[];
  goldSupportingFacts?: GoldSupportingFact[];
}) {
  const hasInputMap = planNode?.input_map && Object.keys(planNode.input_map).length > 0;
  const expectedInputs = expectedInputsForNode(planNode, planNodes);
  const evidenceCitations = node?.evidence_citations ?? [];
  const failureReasons = node ? failureReasonsForNode(node, structuralIssues ?? []) : [];
  const goldFactsByTitle = goldSupportingFactsByTitle(goldSupportingFacts ?? []);

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
          {failureReasons.length > 0 && (
            <div className="failure-panel">
              <div className="failure-heading">
                <GitBranch size={16} />
                <div>
                  <strong>Failure Reason</strong>
                  <span>{failureReasons.length} signal{failureReasons.length === 1 ? "" : "s"}</span>
                </div>
              </div>
              <ul>
                {failureReasons.map((reason, index) => (
                  <li key={`${reason}-${index}`}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
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
          {goldAnswer !== undefined && (
            <div>
              <label>Gold Answer</label>
              <pre>{goldAnswer || "-"}</pre>
            </div>
          )}
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
                {node.supporting_evidence.documents.map((document, index) => {
                  const goldSentenceIndices = goldFactsByTitle.get(document.title) ?? [];
                  const isGoldDocument = goldSentenceIndices.length > 0;
                  return (
                    <details
                      className={`evidence-document ${isGoldDocument ? "gold-document" : "distractor-document"}`}
                      key={document.id}
                    >
                      <summary>
                        <span>{index + 1}</span>
                        <strong>{document.title || "Untitled document"}</strong>
                        <div className="evidence-summary-meta">
                          <em className={isGoldDocument ? "gold-evidence-badge" : "distractor-evidence-badge"}>
                            {isGoldDocument ? "Gold evidence" : "Distractor"}
                          </em>
                          <code>{document.id}</code>
                        </div>
                      </summary>
                      {isGoldDocument && (
                        <div className="gold-sentence-note">
                          Expected gold sentence{goldSentenceIndices.length === 1 ? "" : "s"}:{" "}
                          {goldSentenceIndices.join(", ")}
                        </div>
                      )}
                      <pre>{document.text}</pre>
                      {Object.keys(document.metadata).length > 0 && (
                        <div className="evidence-metadata">
                          <label>Metadata</label>
                          <pre>{formatJson(document.metadata)}</pre>
                        </div>
                      )}
                    </details>
                  );
                })}
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
              <label>System Prompt</label>
              <pre>{planNode.prompt.system}</pre>
            </div>
          )}
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

function failureReasonsForNode(node: NodeTrace, structuralIssues: string[]): string[] {
  const reasons: string[] = [];
  const add = (reason?: string | null) => {
    const value = reason?.trim();
    if (value && !reasons.includes(value)) {
      reasons.push(value);
    }
  };

  if (node.status !== "succeeded") {
    add(`Node status is ${node.status}.`);
  }
  add(node.error);

  if (!node.validation.valid) {
    for (const error of node.validation.errors ?? []) {
      add(error);
    }
  }

  const directStructuralIssues = structuralIssues.filter((issue) =>
    issue.startsWith(`${node.node_id}:`),
  );
  for (const issue of directStructuralIssues) {
    add(issue.replace(`${node.node_id}:`, "").trim());
  }

  if (
    node.status !== "succeeded" &&
    !node.returned_value &&
    !reasons.some((reason) => reason.toLowerCase().includes("returned"))
  ) {
    add("The node did not return a value usable by downstream nodes.");
  }

  return reasons;
}

function goldSupportingFactsByTitle(facts: GoldSupportingFact[]): Map<string, number[]> {
  const byTitle = new Map<string, number[]>();
  for (const fact of facts) {
    const title = fact.title?.trim();
    if (!title) continue;
    const indices = byTitle.get(title) ?? [];
    if (!indices.includes(fact.sentence_index)) {
      indices.push(fact.sentence_index);
    }
    byTitle.set(title, indices.sort((left, right) => left - right));
  }
  return byTitle;
}

function ChatView({ llm }: { llm: LLMSelection }) {
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [strategy, setStrategy] = useState<ChatStrategy>("dag_multi_hop");
  const [run, setRun] = useState<LiveRun | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>();
  const [phase, setPhase] = useState<RunPhase>("idle");
  const [error, setError] = useState("");
  const { maxNodes, setMaxNodes, maxDepth, setMaxDepth, override: plannerOverride } =
    usePlannerSettings();
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
      const started = await startLiveAsk(question.trim(), llm, plannerOverride, strategy);
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

        <div className="sample-select">
          <label htmlFor="chat-strategy">Strategy</label>
          <select
            id="chat-strategy"
            value={strategy}
            disabled={busy}
            onChange={(event) => {
              setStrategy(event.target.value as ChatStrategy);
              setRun(null);
              setSelectedNodeId(undefined);
              setPhase("idle");
            }}
          >
            {CHAT_STRATEGIES.map((system) => (
              <option key={system} value={system}>
                {runTypeLabel(system)}
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

        <TreeParametersControl
          maxDepth={maxDepth}
          maxNodes={maxNodes}
          onMaxDepthChange={setMaxDepth}
          onMaxNodesChange={setMaxNodes}
        />

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
  const tableKey = result ? `${result.run_id}:${result.records.length}` : "empty";
  const { visibleRows, sentinelRef } = useIncrementalRows(result?.records.length ?? 0, tableKey);
  // Ordered union of metric keys across records, so a newly registered metric becomes a column
  // automatically without any frontend change.
  const metricKeys = useMemo(() => {
    const keys: string[] = [];
    for (const record of result?.records ?? []) {
      for (const key of Object.keys(legacyScores(record))) {
        if (!keys.includes(key)) keys.push(key);
      }
    }
    return keys;
  }, [result]);
  // Every metric (raw score) plus the agent-analysis columns, all optional. Question / Gold /
  // Prediction are rendered separately and always shown.
  const toggleableColumns = useMemo<RecordColumn[]>(() => {
    const metricColumns: RecordColumn[] = metricKeys.map((key) => ({
      key: `metric:${key}`,
      label: humanizeMetric(key),
      render: (record) => {
        const scores = legacyScores(record);
        return key in scores ? formatNumber(scores[key], 3) : "-";
      },
    }));
    const analysisColumns: RecordColumn[] = [
      {
        key: "status",
        label: "Status",
        render: (record) =>
          record.error ? "error" : record.structural_failure ? "failed" : "ok",
      },
      ...(onSelectRecord
        ? [
            {
              key: "inspect",
              label: "Inspect",
              render: (record: HotpotBenchmarkRecord) =>
                record.run_trace ? (
                  <button
                    className="record-action"
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectRecord(record);
                    }}
                    type="button"
                  >
                    Inspect
                  </button>
                ) : (
                  <span className="muted-value">No trace</span>
                ),
            },
          ]
        : []),
      {
        key: "evidence",
        label: "Evidence",
        render: (record) =>
          record.wrong_supporting_text_rate === null ||
          record.wrong_supporting_text_rate === undefined
            ? "-"
            : `${formatPercent(record.wrong_supporting_text_rate)} non-gold`,
      },
      {
        key: "gold_recall",
        label: "Gold Recall",
        render: (record) => formatPercent(record.gold_supporting_fact_recall ?? undefined),
      },
    ];
    return [...metricColumns, ...analysisColumns];
  }, [metricKeys, onSelectRecord]);
  // Session-only selection; defaults to the columns the table showed before it became configurable.
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(
    () =>
      new Set([
        "metric:exact_match",
        "metric:cosine_sim",
        "status",
        "inspect",
        "evidence",
        "gold_recall",
      ]),
  );
  const shownColumns = toggleableColumns.filter((column) => visibleColumns.has(column.key));
  // Metric-evaluation annotation set: keys of records added for human band annotation.
  const [annotationKeys, setAnnotationKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    listAnnotations()
      .then((data) => setAnnotationKeys(new Set(data.records.map((record) => record.key))))
      .catch(() => {});
  }, []);

  const toggleAnnotation = (record: HotpotBenchmarkRecord) => {
    if (!result) return;
    const key = `${result.run_id}:${record.id}`;
    const request = annotationKeys.has(key)
      ? deleteAnnotation(key)
      : addAnnotation({
          run_id: result.run_id,
          record_id: record.id,
          question: record.question,
          gold_answer: record.gold_answer,
          prediction: record.prediction,
        });
    request
      .then((data) => setAnnotationKeys(new Set(data.records.map((entry) => entry.key))))
      .catch(() => {});
  };

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
          <label>Tree params</label>
          <strong>
            {result.system === "dag_agent" &&
            result.max_nodes != null &&
            result.max_depth != null
              ? `max node : ${result.max_nodes}  max depth : ${result.max_depth}`
              : "—"}
          </strong>
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
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-sm font-semibold text-slate-600">
          Annotation set: {annotationKeys.size}
        </span>
        <span className="text-xs text-slate-500">
          Click the clipboard on a row to add it, then set its score band in the Metrics tab.
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <span className="text-sm font-semibold text-slate-600">Columns</span>
        {toggleableColumns.map((column) => (
          <label key={column.key} className="flex items-center gap-1.5 text-sm">
            <Checkbox
              checked={visibleColumns.has(column.key)}
              onCheckedChange={(checked) =>
                setVisibleColumns((previous) => {
                  const next = new Set(previous);
                  if (checked === true) {
                    next.add(column.key);
                  } else {
                    next.delete(column.key);
                  }
                  return next;
                })
              }
            />
            <span>{column.label}</span>
          </label>
        ))}
      </div>
      <div className="record-table-wrap">
        <table className="record-table">
          <thead>
            <tr>
              <th>Question</th>
              <th>Gold</th>
              <th>Prediction</th>
              {shownColumns.map((column) => (
                <th key={column.key}>{column.label}</th>
              ))}
              <th>Annotate</th>
            </tr>
          </thead>
          <tbody>
            {result.records.slice(0, visibleRows).map((record) => {
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
                  {shownColumns.map((column) => (
                    <td key={column.key}>{column.render(record)}</td>
                  ))}
                  <td>
                    <button
                      className={`mark-record-button ${
                        annotationKeys.has(`${result.run_id}:${record.id}`) ? "marked" : ""
                      }`}
                      onClick={(event) => {
                        event.stopPropagation();
                        toggleAnnotation(record);
                      }}
                      title={
                        annotationKeys.has(`${result.run_id}:${record.id}`)
                          ? "Remove from annotation set"
                          : "Add to annotation set"
                      }
                      type="button"
                    >
                      {annotationKeys.has(`${result.run_id}:${record.id}`) ? (
                        <ClipboardCheck size={16} />
                      ) : (
                        <ClipboardPlus size={16} />
                      )}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {visibleRows < result.records.length && (
          <div className="table-load-sentinel" ref={sentinelRef}>
            Showing {formatNumber(visibleRows)} of {formatNumber(result.records.length)}
          </div>
        )}
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

// Aggregate answer-metric keys shown as columns in the multi-model comparison table, in
// display order. Mirrors ANSWER_METRICS in dagqa/eval/metrics.py; keys absent from every
// run in the group are hidden, so pre-registry runs simply show fewer columns.
const ANSWER_METRIC_COLUMNS = [
  "exact_match",
  "cosine_sim",
  "context_cosine_sim",
  "mini_l6_cosine_sim",
  "mini_l6_context_cosine_sim",
  "multi_qa_mpnet_cosine_sim",
  "gte_base_cosine_sim",
  "bge_base_cosine_sim",
  "e5_base_cosine_sim",
  "bleu",
  "bigram_f1",
  "chrf",
  "rouge_l",
  "meteor",
  "f1",
] as const;

// Compact column headers; the cosine variants are named after their embedding model.
const ANSWER_METRIC_LABELS: Record<string, string> = {
  exact_match: "EM",
  f1: "F1",
  cosine_sim: "Cosine",
  context_cosine_sim: "Cosine (ctx)",
  mini_l6_cosine_sim: "MiniLM",
  mini_l6_context_cosine_sim: "MiniLM (ctx)",
  multi_qa_mpnet_cosine_sim: "Multi-QA",
  gte_base_cosine_sim: "GTE",
  bge_base_cosine_sim: "BGE",
  e5_base_cosine_sim: "E5",
  bleu: "BLEU",
  bigram_f1: "Bigram F1",
  chrf: "chrF",
  rouge_l: "ROUGE-L",
  meteor: "METEOR",
};

function answerMetricCell(key: string, value: number | undefined) {
  if (value === undefined) return "—";
  if (key === "exact_match") return `${(value * 100).toFixed(0)}%`;
  if (key === "f1") return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(3);
}

const SYSTEM_LABELS: Record<string, string> = {
  direct_llm: "Single prompt",
  dag_multi_hop: "Multi-hop DAG",
  dag_least_to_most: "Least-to-Most",
  dag_ltm_conversation: "LtM Conversation",
  dag_agent: "Combined (LtM+DAG)",
};

const CHAT_STRATEGIES: ChatStrategy[] = [
  "direct_llm",
  "dag_least_to_most",
  "dag_ltm_conversation",
  "dag_multi_hop",
];

function systemLabel(system: string) {
  return SYSTEM_LABELS[system] ?? system;
}

function runTypeLabel(system?: string | null) {
  return SYSTEM_LABELS[system ?? ""] ?? system ?? "Unknown";
}

function benchmarkOptionLabel(item: SavedBenchmarkSummary | HotpotBenchmarkResult) {
  const parts = [
    item.name?.trim(),
    "partial" in item && item.partial ? "partial" : undefined,
    item.dataset_label ?? item.dataset,
    formatSavedDateTime(item.created_at),
    item.subset_label ?? item.subset,
    runTypeLabel(item.system),
    item.model ?? "unknown model",
    "completed" in item && item.completed !== undefined && item.limit
      ? `${formatNumber(item.completed ?? undefined)} / ${formatNumber(item.limit)} examples`
      : `${item.limit} examples`,
    `seed ${item.seed}`,
  ].filter(Boolean);
  return parts.join(" · ");
}

function truncateMiddle(value: string, maxLength = 110) {
  if (value.length <= maxLength) return value;
  const keep = Math.floor((maxLength - 3) / 2);
  return `${value.slice(0, keep)}...${value.slice(value.length - keep)}`;
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

function comparisonAnswerText(record: HotpotBenchmarkRecord) {
  if (record.prediction) return record.prediction;
  if (record.error) return "error";
  if (record.structural_failure) return "failed";
  return "-";
}

function hasEmptyBenchmarkAnswer(record: HotpotBenchmarkRecord) {
  return !record.prediction.trim();
}

function average(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function markedComparisonKey(focusRunId: string, referenceRunId: string, recordId: string) {
  return `${focusRunId}:${referenceRunId}:${recordId}`;
}

function markedComparisonFromRow(
  focus: HotpotBenchmarkResult,
  reference: HotpotBenchmarkResult,
  focusRecord: HotpotBenchmarkRecord,
  referenceRecord: HotpotBenchmarkRecord,
): MarkedComparisonRow {
  return {
    key: markedComparisonKey(focus.run_id, reference.run_id, focusRecord.id),
    record_id: focusRecord.id,
    focus_run_id: focus.run_id,
    reference_run_id: reference.run_id,
    focus_run_name: focus.name,
    reference_run_name: reference.name,
    focus_system: focus.system,
    reference_system: reference.system,
    model: focus.model,
    created_at: focus.created_at,
    seed: focus.seed,
    question: focusRecord.question,
    gold_answer: focusRecord.gold_answer,
    focus_prediction: comparisonAnswerText(focusRecord),
    reference_prediction: comparisonAnswerText(referenceRecord),
    focus_cosine_sim: focusRecord.cosine_sim,
    reference_cosine_sim: referenceRecord.cosine_sim,
    focus_gold_recall: focusRecord.gold_supporting_fact_recall,
    reference_gold_recall: referenceRecord.gold_supporting_fact_recall,
  };
}

function loadLegacyMarkedRows(): MarkedComparisonRow[] {
  try {
    const raw = window.localStorage.getItem(LEGACY_MARKED_RECORDS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is MarkedComparisonRow =>
        item &&
        typeof item === "object" &&
        typeof item.key === "string" &&
        typeof item.record_id === "string" &&
        typeof item.focus_run_id === "string" &&
        typeof item.reference_run_id === "string" &&
        typeof item.focus_system === "string" &&
        typeof item.reference_system === "string" &&
        typeof item.model === "string" &&
        typeof item.created_at === "string" &&
        typeof item.seed === "number" &&
        typeof item.question === "string" &&
        typeof item.gold_answer === "string" &&
        typeof item.focus_prediction === "string" &&
        typeof item.reference_prediction === "string" &&
        typeof item.focus_cosine_sim === "number" &&
        typeof item.reference_cosine_sim === "number",
    );
  } catch {
    return [];
  }
}

function MarkButton({
  marked,
  onToggle,
}: {
  marked: boolean;
  onToggle: (event: MouseEvent<HTMLButtonElement>) => void;
}) {
  return (
    <button
      aria-label={marked ? "Remove saved row" : "Save row for later"}
      className={`mark-record-button ${marked ? "marked" : ""}`}
      onClick={onToggle}
      title={marked ? "Saved for supervisor review" : "Save for supervisor review"}
      type="button"
    >
      {marked ? <BookmarkCheck size={17} /> : <Bookmark size={17} />}
    </button>
  );
}

function aggregateBenchmarkRecords(records: HotpotBenchmarkRecord[]): Record<string, number> {
  const citationCount = records.reduce(
    (sum, record) => sum + (record.evidence_citation_count ?? 0),
    0,
  );
  const wrongCitationCount = records.reduce(
    (sum, record) => sum + (record.wrong_evidence_citation_count ?? 0),
    0,
  );
  return {
    exact_match: average(records.map((record) => record.exact_match)),
    f1: average(records.map((record) => record.f1)),
    cosine_sim: average(records.map((record) => record.cosine_sim)),
    avg_latency_ms: average(records.map((record) => record.latency_ms)),
    avg_gold_supporting_fact_recall: average(
      records
        .map((record) => record.gold_supporting_fact_recall)
        .filter((value): value is number => value !== null && value !== undefined),
    ),
    wrong_supporting_text_rate: citationCount ? wrongCitationCount / citationCount : 0,
  };
}

function MultiModelComparisonView({
  results,
  onSelectResult,
}: {
  results: HotpotBenchmarkResult[];
  onSelectResult: (run: HotpotBenchmarkResult) => void;
}) {
  const uniqueResults = [...results]
    .sort((left, right) => {
      const leftTime = Date.parse(left.created_at ?? "") || 0;
      const rightTime = Date.parse(right.created_at ?? "") || 0;
      return rightTime - leftTime;
    })
    .filter((run, index, sorted) => {
      const key = `${run.dataset}:${run.seed}:${run.model}:${run.system}`;
      return (
        sorted.findIndex((candidate) => {
          const candidateKey = `${candidate.dataset}:${candidate.seed}:${candidate.model}:${candidate.system}`;
          return candidateKey === key;
        }) === index
      );
    });

  const rows = uniqueResults.map((r) => ({
    result: r,
    model: r.model?.split("/").pop() ?? r.model ?? "unknown",
    system: SYSTEM_LABELS[r.system] ?? r.system,
    metrics: r.metrics as Record<string, number | undefined>,
    goldRecall: (r.metrics.avg_gold_supporting_fact_recall ?? 0) * 100,
    latency: (r.metrics.avg_latency_ms ?? 0) / 1000,
    records: r.records.length,
  }));

  // Only answer metrics present in at least one run become columns.
  const metricKeys = ANSWER_METRIC_COLUMNS.filter((key) =>
    rows.some((row) => row.metrics[key] !== undefined),
  );

  // Group by model to show DAG advantage
  const modelGroups = new Map<string, typeof rows>();
  for (const row of rows) {
    const group = modelGroups.get(row.model) ?? [];
    group.push(row);
    modelGroups.set(row.model, group);
  }
  const bestByModelMetric = new Map<string, Map<string, number>>();
  for (const [model, group] of modelGroups.entries()) {
    const bestByMetric = new Map<string, number>();
    for (const key of metricKeys) {
      const values = group
        .map((row) => row.metrics[key])
        .filter((value): value is number => value !== undefined);
      if (values.length > 0) bestByMetric.set(key, Math.max(...values));
    }
    bestByModelMetric.set(model, bestByMetric);
  }

  return (
    <div className="grid gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-bold text-slate-900">
          Multi-model comparison · {results[0]?.records.length ?? 0} examples · seed{" "}
          {results[0]?.seed}
        </h3>
        <span className="text-xs text-slate-500">{uniqueResults.length} runs</span>
      </div>
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase text-slate-500">
              <th className="px-4 py-2">Model</th>
              <th className="px-4 py-2">System</th>
              {metricKeys.map((key) => (
                <th className="whitespace-nowrap px-4 py-2 text-right" key={key}>
                  {ANSWER_METRIC_LABELS[key] ?? humanizeMetric(key)}
                </th>
              ))}
              <th className="px-4 py-2 text-right">Gold Recall</th>
              <th className="px-4 py-2 text-right">Latency</th>
              <th className="px-4 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {[...modelGroups.entries()].map(([model, group]) =>
              group.map((row, index) => {
                const dagRow = group.find((r) => r.result.system === "dag_agent");
                const spRow = group.find((r) => r.result.system === "direct_llm");
                const dagWins =
                  dagRow && spRow && (dagRow.metrics.cosine_sim ?? 0) > (spRow.metrics.cosine_sim ?? 0);
                return (
                  <tr
                    key={row.result.run_id}
                    className={[
                      "border-b border-slate-100 transition-colors hover:bg-slate-50 cursor-pointer",
                      row.result.system === "dag_agent" && dagWins
                        ? "bg-emerald-50/50"
                        : "",
                      index === 0 && modelGroups.size > 1
                        ? "border-t-2 border-t-slate-300"
                        : "",
                    ].join(" ")}
                    onClick={() => onSelectResult(row.result)}
                  >
                    <td className="px-4 py-2 font-medium text-slate-900">
                      {index === 0 ? model : ""}
                    </td>
                    <td className="px-4 py-2 text-slate-600">{row.system}</td>
                    {metricKeys.map((key) => {
                      const value = row.metrics[key];
                      const bestForModel = bestByModelMetric.get(model)?.get(key);
                      const isBest = value !== undefined && value === bestForModel;
                      return (
                        <td
                          className={`px-4 py-2 text-right font-mono ${
                            isBest ? "font-bold text-emerald-700" : ""
                          }`}
                          key={key}
                        >
                          {answerMetricCell(key, value)}
                        </td>
                      );
                    })}
                    <td className="px-4 py-2 text-right font-mono">
                      {row.goldRecall.toFixed(0)}%
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-slate-500">
                      {row.latency.toFixed(1)}s
                    </td>
                    <td className="px-4 py-2 text-right">
                      {row.result.system === "dag_agent" && dagWins && (
                        <span className="inline-block rounded bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-800">
                          DAG wins
                        </span>
                      )}
                    </td>
                  </tr>
                );
              }),
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function BenchmarkComparisonView({
  first,
  second,
  onSelectRecord,
  markedKeys,
  onToggleMarkedRow,
}: {
  first: HotpotBenchmarkResult;
  second: HotpotBenchmarkResult;
  onSelectRecord?: (run: HotpotBenchmarkResult, record: HotpotBenchmarkRecord) => void;
  markedKeys?: Set<string>;
  onToggleMarkedRow?: (
    focus: HotpotBenchmarkResult,
    reference: HotpotBenchmarkResult,
    focusRecord: HotpotBenchmarkRecord,
    referenceRecord: HotpotBenchmarkRecord,
  ) => void;
}) {
  const mixedSystems = first.system !== second.system;
  const focus = mixedSystems && first.system === "direct_llm" ? second : first;
  const reference = focus === first ? second : first;
  const focusLabel = mixedSystems ? systemLabel(focus.system) : "Run A";
  const referenceLabel = mixedSystems ? systemLabel(reference.system) : "Run B";
  const deltaLabel = mixedSystems ? `${systemLabel(focus.system)} impact` : "Run A impact";
  const sameSeed = focus.seed === reference.seed;
  const referenceById = new Map(reference.records.map((record) => [record.id, record]));
  const [hideEmptyResponses, setHideEmptyResponses] = useState(false);
  const aligned = sameSeed
    ? focus.records
        .map((record) => [record, referenceById.get(record.id)] as const)
        .filter((pair): pair is readonly [HotpotBenchmarkRecord, HotpotBenchmarkRecord] => Boolean(pair[1]))
    : [];
  const emptyStats = aligned.reduce(
    (stats, [focusRecord, referenceRecord]) => {
      const focusEmpty = hasEmptyBenchmarkAnswer(focusRecord);
      const referenceEmpty = hasEmptyBenchmarkAnswer(referenceRecord);
      return {
        focusEmpty: stats.focusEmpty + (focusEmpty ? 1 : 0),
        referenceEmpty: stats.referenceEmpty + (referenceEmpty ? 1 : 0),
        eitherEmpty: stats.eitherEmpty + (focusEmpty || referenceEmpty ? 1 : 0),
      };
    },
    { focusEmpty: 0, referenceEmpty: 0, eitherEmpty: 0 },
  );
  const filteredAligned = hideEmptyResponses
    ? aligned.filter(
        ([focusRecord, referenceRecord]) =>
          !hasEmptyBenchmarkAnswer(focusRecord) && !hasEmptyBenchmarkAnswer(referenceRecord),
      )
    : aligned;
  const focusMetrics = hideEmptyResponses
    ? aggregateBenchmarkRecords(filteredAligned.map(([record]) => record))
    : focus.metrics;
  const referenceMetrics = hideEmptyResponses
    ? aggregateBenchmarkRecords(filteredAligned.map(([, record]) => record))
    : reference.metrics;
  const comparisonTableKey = `${focus.run_id}:${reference.run_id}:${filteredAligned.length}:${hideEmptyResponses}`;
  const { visibleRows, sentinelRef } = useIncrementalRows(filteredAligned.length, comparisonTableKey);
  const metricDeltas = COMPARISON_METRICS.map(([key, label, format, preference]) => {
    const focusValue = focusMetrics[key] ?? 0;
    const referenceValue = referenceMetrics[key] ?? 0;
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
          <strong>{mixedSystems ? `${systemLabel(focus.system)} vs ${systemLabel(reference.system)}` : `${systemLabel(first.system)} comparison`}</strong>
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
            {sameSeed
              ? hideEmptyResponses
                ? `${filteredAligned.length} shown / ${aligned.length} aligned`
                : `${aligned.length} aligned`
              : "Different seeds"}
          </span>
        </div>
      </div>
      <div className="comparison-legend">
        <span><i className="focus-dot" />{focusLabel}</span>
        <span><i className="reference-dot" />{referenceLabel}</span>
        <strong>{deltaLabel}</strong>
      </div>
      {sameSeed && (
        <div className="comparison-filter-bar">
          <label>
            <Checkbox
              id="hide-empty-benchmark-answers"
              checked={hideEmptyResponses}
              onCheckedChange={(checked) => setHideEmptyResponses(checked === true)}
            />
            <span>Hide rows with an empty answer</span>
          </label>
          <div>
            <strong>
              {formatNumber(hideEmptyResponses ? emptyStats.eitherEmpty : 0)} of{" "}
              {formatNumber(aligned.length)} filtered
            </strong>
            <span>
              {focusLabel}: {formatNumber(emptyStats.focusEmpty)} empty · {referenceLabel}:{" "}
              {formatNumber(emptyStats.referenceEmpty)} empty
            </span>
          </div>
        </div>
      )}
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
                {onToggleMarkedRow && <th>Save</th>}
                <th>{focusLabel}</th>
                <th>{referenceLabel}</th>
                <th>{deltaLabel}</th>
                <th>Gold recall impact</th>
              </tr>
            </thead>
            <tbody>
              {filteredAligned.slice(0, visibleRows).map(([focusRecord, referenceRecord]) => {
                const cosineDelta = focusRecord.cosine_sim - referenceRecord.cosine_sim;
                const focusRecall = focusRecord.gold_supporting_fact_recall ?? 0;
                const referenceRecall = referenceRecord.gold_supporting_fact_recall ?? 0;
                const recallDelta = focusRecall - referenceRecall;
                const rowMarked = Boolean(
                  markedKeys?.has(markedComparisonKey(focus.run_id, reference.run_id, focusRecord.id)),
                );
                return (
                <tr key={focusRecord.id}>
                  <td>{focusRecord.question}</td>
                  <td>{focusRecord.gold_answer}</td>
                  {onToggleMarkedRow && (
                    <td>
                      <MarkButton
                        marked={rowMarked}
                        onToggle={(event) => {
                          event.stopPropagation();
                          onToggleMarkedRow(focus, reference, focusRecord, referenceRecord);
                        }}
                      />
                    </td>
                  )}
                  <td>
                    <button
                      className={`comparison-answer ${focusRecord.error ? "failed" : ""}`}
                      onClick={() => onSelectRecord?.(focus, focusRecord)}
                      title={focusRecord.error || "Open benchmark detail"}
                    >
                      <span>{comparisonAnswerText(focusRecord)}</span>
                      <small>Gold recall {formatPercent(focusRecall)}</small>
                    </button>
                  </td>
                  <td>
                    <button
                      className={`comparison-answer ${referenceRecord.error ? "failed" : ""}`}
                      onClick={() => onSelectRecord?.(reference, referenceRecord)}
                      title={referenceRecord.error || "Open benchmark detail"}
                    >
                      <span>{comparisonAnswerText(referenceRecord)}</span>
                      <small>Gold recall {formatPercent(referenceRecall)}</small>
                    </button>
                  </td>
                  <td>
                    <span className={`row-delta ${deltaClass(cosineDelta)}`}>
                      {cosineDelta > 0 ? "+" : ""}
                      {formatNumber(cosineDelta, 3)}
                    </span>
                  </td>
                  <td>
                    <span className={`row-delta ${deltaClass(recallDelta)}`}>
                      {recallDelta > 0 ? "+" : ""}
                      {formatPercent(recallDelta)}
                    </span>
                  </td>
                </tr>
              )})}
            </tbody>
          </table>
          {filteredAligned.length === 0 && (
            <div className="empty compact-empty">No comparable rows after filtering.</div>
          )}
          {visibleRows < filteredAligned.length && (
            <div className="table-load-sentinel" ref={sentinelRef}>
              Showing {formatNumber(visibleRows)} of {formatNumber(filteredAligned.length)} aligned rows
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DatasetView({
  recordId,
  onSelectRecord,
  llm,
  modelCatalog,
  markedKeys,
  onToggleMarkedRow,
}: {
  recordId?: string;
  onSelectRecord: (recordId?: string) => void;
  llm: LLMSelection;
  modelCatalog?: LLMModelCatalog;
  markedKeys: Set<string>;
  onToggleMarkedRow: (
    focus: HotpotBenchmarkResult,
    reference: HotpotBenchmarkResult,
    focusRecord: HotpotBenchmarkRecord,
    referenceRecord: HotpotBenchmarkRecord,
  ) => void;
}) {
  const [limit, setLimit] = useState(5);
  const [benchmarkName, setBenchmarkName] = useState("");
  const [benchmarkDataset, setBenchmarkDataset] = useState("hotpotqa");
  const [benchmarkSubset, setBenchmarkSubset] = useState<BenchmarkSubset>("validation");
  const [systems, setSystems] = useState<string[]>(["direct_llm", "dag_least_to_most"]);
  const [seedInput, setSeedInput] = useState("");
  const clusterModels = useMemo(
    () => (modelCatalog?.models ?? []).filter((m) => m.provider === "cluster"),
    [modelCatalog],
  );
  const [selectedModels, setSelectedModels] = useState<LLMSelection[]>([]);
  const [meta, setMeta] = useState<BenchmarkMeta>();
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<HotpotBenchmarkResult>();
  const [comparisonResults, setComparisonResults] = useState<HotpotBenchmarkResult[]>([]);
  const [savedBenchmarkSummaries, setSavedBenchmarkSummaries] = useState<SavedBenchmarkSummary[]>([]);
  const [unfinishedLiveRuns, setUnfinishedLiveRuns] = useState<LiveBenchmarkSummary[]>([]);
  const [selectedLiveRunId, setSelectedLiveRunId] = useState("");
  const [liveRun, setLiveRun] = useState<LiveBenchmark>();
  const [error, setError] = useState("");
  const { maxNodes, setMaxNodes, maxDepth, setMaxDepth, override: plannerOverride } =
    usePlannerSettings();
  const multiModel = selectedModels.length > 1;

  useEffect(() => {
    if (clusterModels.length > 0 && selectedModels.length === 0) {
      setSelectedModels(clusterModels.map((m) => ({ provider: m.provider, model: m.model })));
    }
  }, [clusterModels, selectedModels.length]);

  const maxExamples = meta?.total_examples && meta.total_examples > 1 ? meta.total_examples : 7405;
  const resolvedLimit = Math.min(limit, maxExamples);
  const progressPercent = liveRun?.total ? (liveRun.completed / liveRun.total) * 100 : 0;
  const benchmarkRunning = liveRun?.phase === "running" || liveRun?.phase === "stopping";
  const canStop = liveRun?.phase === "running" || liveRun?.phase === "stopping";
  const canResume = liveRun?.phase === "stopped" || liveRun?.phase === "error";
  const selectedLiveRun = unfinishedLiveRuns.find((run) => run.run_id === selectedLiveRunId);
  const selectedLiveRunProgress = selectedLiveRun?.total
    ? (selectedLiveRun.completed / selectedLiveRun.total) * 100
    : 0;

  const historicalMsBySystem = useMemo(() => {
    const estimates = new Map<string, number>();
    for (const system of ["dag_agent", "direct_llm"]) {
      const samples = savedBenchmarkSummaries
        .filter(
          (item) =>
            item.system === system &&
            (!item.model || item.model === llm.model) &&
            (item.metrics.total_runtime_ms || item.metrics.avg_latency_ms),
        )
        .slice(0, 8)
        .map((item) => {
          const examples =
            item.metrics.example_count || item.metrics.completed || item.limit || resolvedLimit;
          if (item.metrics.total_runtime_ms && examples) {
            return item.metrics.total_runtime_ms / examples;
          }
          return item.metrics.avg_latency_ms;
        })
        .filter((value): value is number => value !== undefined && Number.isFinite(value));
      estimates.set(system, median(samples) ?? (system === "dag_agent" ? 45_000 : 5_000));
    }
    return estimates;
  }, [llm.model, resolvedLimit, savedBenchmarkSummaries]);

  const plannedTotalMs = useMemo(
    () =>
      systems.reduce(
        (total, system) => total + resolvedLimit * (historicalMsBySystem.get(system) ?? 0),
        0,
      ),
    [historicalMsBySystem, resolvedLimit, systems],
  );

  const benchmarkEstimate = useMemo(() => {
    const total = liveRun?.total ?? resolvedLimit * systems.length;
    const completed = liveRun?.completed ?? 0;
    const elapsedMs = liveRun?.total_runtime_ms ?? 0;
    const serverEstimate = liveRun?.estimate;
    if (serverEstimate) {
      const remainingMs = Math.max(serverEstimate.remaining_ms ?? 0, 0);
      const totalMs = Math.max(serverEstimate.total_ms ?? elapsedMs + remainingMs, elapsedMs);
      const finishAt =
        remainingMs > 0 ? new Date(Date.now() + remainingMs).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }) : undefined;
      return {
        total,
        completed,
        remaining: Math.max(total - completed, 0),
        elapsedMs,
        remainingMs,
        totalMs,
        finishAt,
        avgLlmCallMs: serverEstimate.avg_llm_call_ms ?? undefined,
        avgDagNodeCount: serverEstimate.avg_dag_node_count ?? undefined,
        lastRecordAgeMs: liveRun?.last_record_completed_at
          ? Math.max(Date.now() - new Date(liveRun.last_record_completed_at).getTime(), 0)
          : undefined,
        confidence:
          (serverEstimate.observed_llm_call_count ?? 0) >= 8
            ? "High"
            : completed > 0
              ? "Calibrating"
              : "Fallback",
      };
    }
    const historicalMsPerExample = total > 0 ? plannedTotalMs / total : 0;
    const observedMsPerExample = completed > 0 ? elapsedMs / completed : undefined;
    const observedWeight = observedMsPerExample ? Math.min(0.9, completed / 8) : 0;
    const blendedMsPerExample =
      observedMsPerExample === undefined
        ? historicalMsPerExample
        : observedMsPerExample * observedWeight + historicalMsPerExample * (1 - observedWeight);
    const remaining = Math.max(total - completed, 0);
    const remainingMs = remaining * blendedMsPerExample;
    const totalMs = elapsedMs + remainingMs;
    const finishAt =
      remainingMs > 0 ? new Date(Date.now() + remainingMs).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }) : undefined;

    return {
      total,
      completed,
      remaining,
      elapsedMs,
      remainingMs,
      totalMs,
      finishAt,
      avgLlmCallMs: undefined,
      avgDagNodeCount: undefined,
      lastRecordAgeMs: liveRun?.last_record_completed_at
        ? Math.max(Date.now() - new Date(liveRun.last_record_completed_at).getTime(), 0)
        : undefined,
      confidence:
        completed >= 8 ? "High" : completed > 0 ? "Calibrating" : savedBenchmarkSummaries.length ? "Historical" : "Fallback",
    };
  }, [liveRun, plannedTotalMs, resolvedLimit, savedBenchmarkSummaries.length, systems.length]);

  function updateLimit(value: number) {
    if (!Number.isFinite(value)) return;
    setLimit(Math.min(Math.max(Math.trunc(value), 1), maxExamples));
  }

  useEffect(() => {
    getBenchmarkMeta(benchmarkDataset, benchmarkSubset)
      .then((nextMeta) => {
        setMeta(nextMeta);
        if (nextMeta.subset !== benchmarkSubset) {
          setBenchmarkSubset(nextMeta.subset);
        }
        setLimit(Math.min(nextMeta.default_limit, nextMeta.total_examples));
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load benchmark metadata"));
  }, [benchmarkDataset, benchmarkSubset]);

  function changeBenchmarkDataset(dataset: string) {
    setBenchmarkDataset(dataset);
    const defaultSubset =
      meta?.datasets.find((option) => option.id === dataset)?.default_subset ?? "validation";
    setBenchmarkSubset(defaultSubset);
  }

  useEffect(() => {
    listBenchmarkResults()
      .then((response) => setSavedBenchmarkSummaries(response.results))
      .catch(() => setSavedBenchmarkSummaries([]));
  }, []);

  async function refreshUnfinishedLiveRuns() {
    try {
      const response = await listLiveBenchmarks();
      setUnfinishedLiveRuns(response.runs);
      setSelectedLiveRunId((current) =>
        current && response.runs.some((run) => run.run_id === current)
          ? current
          : (response.runs[0]?.run_id ?? ""),
      );
    } catch {
      setUnfinishedLiveRuns([]);
      setSelectedLiveRunId("");
    }
  }

  useEffect(() => {
    refreshUnfinishedLiveRuns();
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
        setSelectedLiveRunId(restored.run_id);
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
    setSeedInput(formatSeedInput(started.seed));
    setBenchmarkName(started.name ?? "");
    setBenchmarkDataset(started.dataset ?? "hotpotqa");
    setBenchmarkSubset((started.subset as BenchmarkSubset | undefined) ?? "validation");
    window.localStorage.setItem(ACTIVE_BENCHMARK_STORAGE_KEY, started.run_id);
    setSelectedLiveRunId(started.run_id);

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
      const parsedSeed = parseOptionalInteger(seedInput);
      if (!systems.length) {
        throw new Error("Select at least one system.");
      }
      const name = benchmarkName.trim() || undefined;
      const modelsToRun = multiModel ? selectedModels : undefined;
      const preflight = await preflightBenchmark(
        resolvedLimit,
        systems,
        llm,
        parsedSeed,
        name,
        plannerOverride,
        benchmarkSubset,
        benchmarkDataset,
        modelsToRun,
      );
      if (!preflight.ok) {
        const failed = preflight.checks.filter((check) => !check.ok);
        throw new Error(failed.map((check) => check.detail).join(" "));
      }
      const started = await startLiveBenchmark(
        resolvedLimit,
        systems,
        llm,
        parsedSeed,
        name,
        plannerOverride,
        benchmarkSubset,
        benchmarkDataset,
        modelsToRun,
      );
      await refreshUnfinishedLiveRuns();
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
      await refreshUnfinishedLiveRuns();
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
      await refreshUnfinishedLiveRuns();
      await watchBenchmark(resumed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not resume benchmark");
    } finally {
      setBusy(false);
    }
  }

  async function loadSelectedLiveRun() {
    if (!selectedLiveRunId) return;
    setBusy(true);
    setError("");
    try {
      const selected = await getLiveBenchmark(selectedLiveRunId);
      setLiveRun(selected);
      setResult(selected.comparison_results?.[0] ?? selected);
      setComparisonResults(selected.comparison_results ?? []);
      setSeedInput(formatSeedInput(selected.seed));
      setBenchmarkName(selected.name ?? "");
      setBenchmarkDataset(selected.dataset ?? "hotpotqa");
      setBenchmarkSubset((selected.subset as BenchmarkSubset | undefined) ?? "validation");
      setSystems(
        selected.systems?.filter((system) => system in SYSTEM_LABELS) ??
          systems,
      );
      window.localStorage.setItem(ACTIVE_BENCHMARK_STORAGE_KEY, selected.run_id);
      if (selected.phase === "running" || selected.phase === "stopping") {
        await watchBenchmark(selected);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load benchmark");
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
            <p>Run a seeded benchmark sample against one or both systems.</p>
          </div>
          <Database size={22} />
        </div>

        <Card className="my-4 overflow-hidden">
          <CardContent className="grid gap-4 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 pb-3">
              <div>
                <CardTitle>Benchmark Setup</CardTitle>
                <div className="mt-1 text-xs font-medium text-slate-500">
                  {formatNumber(resolvedLimit * systems.length)} total examples · {llm.model}
                </div>
              </div>
              <Badge className="bg-emerald-50 text-emerald-800">{benchmarkEstimate.confidence} ETA</Badge>
            </div>

            <div className="grid gap-4 xl:grid-cols-[minmax(260px,0.9fr)_minmax(320px,1fr)_minmax(300px,0.9fr)]">
              <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50/70 p-3">
                <div className="grid gap-2">
                  <Label htmlFor="benchmark-dataset">Dataset</Label>
                  <select
                    className="h-9 rounded-md border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-900 shadow-sm"
                    id="benchmark-dataset"
                    value={benchmarkDataset}
                    onChange={(event) => changeBenchmarkDataset(event.target.value)}
                  >
                    {(meta?.datasets ?? [
                      { id: "hotpotqa", label: "HotpotQA", default_subset: "validation" },
                      { id: "musique", label: "MuSiQue", default_subset: "validation_3hop_plus" },
                    ]).map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="benchmark-subset">Subset</Label>
                  <select
                    className="h-9 rounded-md border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-900 shadow-sm"
                    id="benchmark-subset"
                    value={benchmarkSubset}
                    onChange={(event) => setBenchmarkSubset(event.target.value)}
                  >
                    {(meta?.subsets ?? [{ id: "validation", label: "Full HotpotQA validation" }]).map(
                      (option) => (
                        <option key={option.id} value={option.id}>
                          {option.label}
                        </option>
                      ),
                    )}
                  </select>
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="benchmark-name">Benchmark name</Label>
                  <Input
                    className="h-9 bg-white"
                    id="benchmark-name"
                    placeholder="Optional"
                    value={benchmarkName}
                    onChange={(event) => setBenchmarkName(event.target.value)}
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="benchmark-seed">Seed</Label>
                  <Input
                    autoComplete="off"
                    className="h-9 bg-white"
                    id="benchmark-seed"
                    inputMode="numeric"
                    pattern="-?[0-9]*"
                    placeholder="Random"
                    type="text"
                    value={seedInput}
                    onChange={(event) => setSeedInput(event.target.value)}
                  />
                </div>
              </div>

              <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50/70 p-3">
                <div className="flex items-center justify-between gap-3">
                  <Label>Systems</Label>
                  <span className="text-xs font-semibold text-slate-500">{systems.length} selected</span>
                </div>
                <div className="grid gap-2 sm:grid-cols-2">
                  {[
                    ["direct_llm", "Single prompt", "One LLM call with all evidence"],
                    ["dag_multi_hop", "Multi-hop DAG", "Plan then execute nodes independently"],
                    ["dag_least_to_most", "Least-to-Most", "Plan then answer all steps in one call"],
                    ["dag_ltm_conversation", "LtM Conversation", "Plan then answer each step in a chat turn"],
                  ].map(([value, label, description]) => (
                    <label
                      className="flex min-h-16 items-start gap-3 rounded-md border border-slate-200 bg-white p-3 shadow-sm transition-colors hover:border-emerald-200 hover:bg-emerald-50/40"
                      key={value}
                    >
                      <Checkbox
                        checked={systems.includes(value)}
                        onCheckedChange={(checked) =>
                          setSystems((current) =>
                            checked
                              ? [...current, value]
                              : current.filter((system) => system !== value),
                          )
                        }
                      />
                      <span>
                        <span className="block text-sm font-semibold text-slate-900">{label}</span>
                        <span className="mt-1 block text-xs font-medium text-slate-500">
                          {description}
                        </span>
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {clusterModels.length > 1 && (
              <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50/70 p-3">
                <div className="flex items-center justify-between gap-3">
                  <Label>Models</Label>
                  <span className="text-xs font-semibold text-slate-500">
                    {selectedModels.length} selected
                  </span>
                </div>
                <div className="grid gap-2">
                  {clusterModels.map((option) => (
                    <label
                      className="flex items-center gap-3 rounded-md border border-slate-200 bg-white p-3 shadow-sm transition-colors hover:border-emerald-200 hover:bg-emerald-50/40"
                      key={`${option.provider}:${option.model}`}
                    >
                      <Checkbox
                        checked={selectedModels.some(
                          (sel) => sel.provider === option.provider && sel.model === option.model,
                        )}
                        onCheckedChange={(checked) =>
                          setSelectedModels((current) =>
                            checked
                              ? [...current, { provider: option.provider, model: option.model }]
                              : current.filter(
                                  (sel) =>
                                    !(sel.provider === option.provider && sel.model === option.model),
                                ),
                          )
                        }
                      />
                      <span className="text-sm font-semibold text-slate-900">
                        {option.label}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
              )}

              <div className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50/70 p-3">
                <div className="flex items-center justify-between gap-3">
                  <Label htmlFor="benchmark-limit">Examples</Label>
                  <div className="flex items-center gap-2 text-xs font-semibold text-slate-500">
                    <Input
                      aria-label="Benchmark example count"
                      className="h-9 w-24 bg-white text-right"
                      min={1}
                      max={maxExamples}
                      type="number"
                      value={resolvedLimit}
                      onChange={(event) => updateLimit(Number(event.target.value))}
                    />
                    / {formatNumber(maxExamples)}
                  </div>
                </div>
                <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 text-xs font-medium text-slate-500">
                  <span>1</span>
                  <input
                    className="h-7 w-full accent-emerald-800"
                    id="benchmark-limit"
                    min={1}
                    max={maxExamples}
                    type="range"
                    value={resolvedLimit}
                    onChange={(event) => updateLimit(Number(event.target.value))}
                  />
                  <span>{formatNumber(maxExamples)}</span>
                </div>
                <div className="grid grid-cols-3 gap-2 rounded-md bg-white p-2">
                  <div>
                    <div className="text-[11px] font-semibold text-slate-500">Total</div>
                    <div className="text-sm font-bold text-slate-950">
                      {formatDuration(benchmarkEstimate.totalMs)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] font-semibold text-slate-500">Remaining</div>
                    <div className="text-sm font-bold text-slate-950">
                      {formatDuration(benchmarkEstimate.remainingMs)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] font-semibold text-slate-500">Finish</div>
                    <div className="text-sm font-bold text-slate-950">
                      {benchmarkEstimate.finishAt ?? "-"}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <TreeParametersControl
              maxDepth={maxDepth}
              maxNodes={maxNodes}
              onMaxDepthChange={setMaxDepth}
              onMaxNodesChange={setMaxNodes}
            />

            <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-emerald-100 bg-emerald-50/70 p-3">
              <div className="text-sm font-semibold text-emerald-950">
                Estimated full run: {formatDuration(benchmarkEstimate.totalMs)}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button disabled={busy || benchmarkRunning} onClick={runDatasetBenchmark} size="lg">
                  {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                  {busy ? "Running" : "Start benchmark"}
                </Button>
                {liveRun && (
                  <>
                    <Button disabled={!canStop} onClick={stopBenchmark} variant="outline">
                      <Square size={16} />
                      Stop
                    </Button>
                    <Button disabled={!canResume || busy} onClick={resumeBenchmark} variant="outline">
                      <RotateCcw size={16} />
                      Resume
                    </Button>
                  </>
                )}
              </div>
            </div>

            <div className="grid gap-3 rounded-lg border border-slate-200 bg-white p-3 shadow-sm lg:grid-cols-[minmax(260px,1fr)_minmax(280px,1.1fr)_auto] lg:items-center">
              <div>
                <div className="flex items-center gap-2">
                  <History size={16} className="text-slate-500" />
                  <Label htmlFor="unfinished-live-run">Unfinished runs</Label>
                  <Badge className="bg-slate-100 text-slate-700">{unfinishedLiveRuns.length}</Badge>
                </div>
                <p className="mt-1 text-xs font-medium text-slate-500">
                  Load a checkpoint, then use Resume to continue it.
                </p>
              </div>

              <div className="grid gap-2">
                <select
                  className="h-10 w-full rounded-md border border-slate-200 bg-slate-50 px-3 text-sm font-semibold text-slate-900 shadow-sm outline-none transition-colors focus:border-emerald-400 focus:bg-white"
                  disabled={!unfinishedLiveRuns.length || busy}
                  id="unfinished-live-run"
                  value={selectedLiveRunId}
                  onChange={(event) => setSelectedLiveRunId(event.target.value)}
                >
                  {!unfinishedLiveRuns.length && <option value="">No unfinished runs</option>}
                  {unfinishedLiveRuns.map((run) => (
                    <option key={run.run_id} value={run.run_id}>
                      {(run.name || "Unnamed run")} · {formatSavedDateTime(run.created_at)} ·{" "}
                      {run.model || "model"} · {formatNumber(run.completed)} /{" "}
                      {formatNumber(run.total)}
                    </option>
                  ))}
                </select>
                {selectedLiveRun && (
                  <div className="grid gap-2 rounded-md bg-slate-50 p-2">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-semibold text-slate-600">
                      <span>
                        {selectedLiveRun.current_system
                          ? `${systemLabel(selectedLiveRun.current_system)} · `
                          : ""}
                        {selectedLiveRun.phase}
                      </span>
                      <span>
                        {formatNumber(selectedLiveRun.completed)} /{" "}
                        {formatNumber(selectedLiveRun.total)} examples
                      </span>
                    </div>
                    <Progress value={selectedLiveRunProgress} />
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs font-medium text-slate-500">
                      <span>{selectedLiveRun.provider ?? "provider"}</span>
                      <span>{selectedLiveRun.model ?? "model"}</span>
                      <span>Elapsed {formatDuration(selectedLiveRun.total_runtime_ms)}</span>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-2 lg:justify-end">
                <Button
                  disabled={!selectedLiveRunId || busy}
                  onClick={loadSelectedLiveRun}
                  variant="outline"
                >
                  <RotateCcw size={16} />
                  Load
                </Button>
                <Button disabled={busy} onClick={refreshUnfinishedLiveRuns} variant="outline">
                  <History size={16} />
                  Refresh
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {error && <div className="error-box">{error}</div>}

        {liveRun && (benchmarkRunning || liveRun.phase === "stopped" || liveRun.phase === "error") && (
          <Card className="mb-4 border-emerald-200 bg-emerald-50/60">
            <CardContent className="grid gap-3 pt-4">
              <div className="flex items-start justify-between gap-4">
              <div>
                <strong className="block text-sm text-emerald-950">
                  {formatNumber(liveRun.completed)} / {formatNumber(liveRun.total)} examples
                </strong>
                <span className="mt-1 block text-sm text-emerald-800">
                  {liveRun.current_system ? `${systemLabel(liveRun.current_system)}: ` : ""}
                  {liveRun.current_question ?? "Finalizing benchmark run."}
                </span>
              </div>
                <Badge>{progressPercent.toFixed(0)}%</Badge>
            </div>
              <Progress value={progressPercent} />
              <div className="grid gap-2 text-xs font-semibold text-slate-600 sm:grid-cols-4">
                <span>Elapsed {formatDuration(benchmarkEstimate.elapsedMs)}</span>
                <span>Remaining {formatDuration(benchmarkEstimate.remainingMs)}</span>
                <span>Estimated total {formatDuration(benchmarkEstimate.totalMs)}</span>
                <span>
                  Last item{" "}
                  {benchmarkEstimate.lastRecordAgeMs === undefined
                    ? "not yet"
                    : `${formatDuration(benchmarkEstimate.lastRecordAgeMs)} ago`}
                </span>
            </div>
              {(benchmarkEstimate.avgLlmCallMs || benchmarkEstimate.avgDagNodeCount) && (
                <div className="flex flex-wrap gap-3 text-[11px] font-semibold text-emerald-900">
                  {benchmarkEstimate.avgLlmCallMs && (
                    <span>LLM call avg {formatDuration(benchmarkEstimate.avgLlmCallMs)}</span>
                  )}
                  {benchmarkEstimate.avgDagNodeCount && (
                    <span>DAG nodes avg {formatNumber(benchmarkEstimate.avgDagNodeCount, 1)}</span>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        <section className="answer-panel">
          <div className="section-header">
            <div>
              <h2>Benchmark Result</h2>
              <span>
                {result
                  ? `${result.subset_label ?? result.subset ?? result.split}, seed ${result.seed}`
                  : meta
                    ? `${formatNumber(meta.total_examples)} ${meta.subset_label} examples available`
                    : "Loading metadata"}
              </span>
            </div>
            <BarChart3 size={18} />
          </div>
          <div className="answer-body benchmark-output">
            {comparisonResults.length > 2 ? (
              <MultiModelComparisonView
                results={comparisonResults}
                onSelectResult={(run) => {
                  setResult(run);
                  setComparisonResults([]);
                }}
              />
            ) : comparisonResults.length === 2 ? (
              <BenchmarkComparisonView
                first={comparisonResults[0]}
                second={comparisonResults[1]}
                onSelectRecord={(run, record) => {
                  setResult(run);
                  onSelectRecord(record.id);
                }}
                markedKeys={markedKeys}
                onToggleMarkedRow={onToggleMarkedRow}
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
          {record.error && (
            <section className="answer-panel">
              <div className="section-header">
                <div>
                  <h2>Recorded Error</h2>
                  <span>No prompt trace was saved for this row</span>
                </div>
                <GitBranch size={18} />
              </div>
              <div className="answer-body">
                <pre>{record.error}</pre>
              </div>
            </section>
          )}
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
          {record.raw_prediction && record.raw_prediction !== record.prediction && (
            <div className="metric">
              <label>Raw Prediction</label>
              <strong>{record.raw_prediction}</strong>
            </div>
          )}
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

        <section className="answer-panel">
          <div className="section-header">
            <div>
              <h2>Gold Answer</h2>
              <span>Expected benchmark answer</span>
            </div>
            <CheckCircle2 size={18} />
          </div>
          <div className="answer-body">
            <pre>{record.gold_answer || "-"}</pre>
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
          goldAnswer={record.gold_answer}
          structuralIssues={record.structural_issues ?? []}
          goldSupportingFacts={record.gold_supporting_facts ?? []}
        />
      </aside>
    </main>
  );
}

function ResultsView({
  runId,
  recordId,
  source,
  onNavigate,
  markedKeys,
  onToggleMarkedRow,
}: {
  runId?: string;
  recordId?: string;
  source?: "marked";
  onNavigate: (runId?: string, recordId?: string) => void;
  markedKeys: Set<string>;
  onToggleMarkedRow: (
    focus: HotpotBenchmarkResult,
    reference: HotpotBenchmarkResult,
    focusRecord: HotpotBenchmarkRecord,
    referenceRecord: HotpotBenchmarkRecord,
  ) => void;
}) {
  const [items, setItems] = useState<SavedBenchmarkSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState(runId ?? "");
  const [comparisonRunId, setComparisonRunId] = useState("");
  const [selectedResult, setSelectedResult] = useState<HotpotBenchmarkResult>();
  const [comparisonResult, setComparisonResult] = useState<HotpotBenchmarkResult>();
  const [multiModelResults, setMultiModelResults] = useState<HotpotBenchmarkResult[]>([]);
  const [showMultiModel, setShowMultiModel] = useState(true);
  const [datasetFilter, setDatasetFilter] = useState("all");
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
        const groupMembers = response.results.filter(
          (item) =>
            !item.partial &&
            item.run_id !== targetRunId &&
            item.comparison_group_id &&
            item.comparison_group_id === loaded.comparison_group_id,
        );
        if (groupMembers.length > 0) {
          const allGroupResults = await Promise.all(
            groupMembers.map((item) => getBenchmarkResult(item.run_id)),
          );
          setComparisonRunId(groupMembers[0].run_id);
          setComparisonResult(allGroupResults[0]);
          setMultiModelResults([loaded, ...allGroupResults]);
        } else {
          setComparisonRunId("");
          setComparisonResult(undefined);
          setMultiModelResults([]);
          setShowMultiModel(false);
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

  const datasetOptions = useMemo(() => {
    const options = new Map<string, string>();
    for (const item of items) {
      if (!item.dataset) continue;
      options.set(item.dataset, item.dataset_label ?? item.dataset);
    }
    return [...options.entries()].sort((a, b) => a[1].localeCompare(b[1]));
  }, [items]);
  const filteredItems = useMemo(
    () =>
      datasetFilter === "all"
        ? items
        : items.filter((item) => (item.dataset ?? "hotpotqa") === datasetFilter),
    [datasetFilter, items],
  );

  useEffect(() => {
    if (datasetFilter === "all") return;
    if (selectedRunId && !filteredItems.some((item) => item.run_id === selectedRunId)) {
      setSelectedRunId("");
      setSelectedResult(undefined);
      setComparisonRunId("");
      setComparisonResult(undefined);
    }
  }, [datasetFilter, filteredItems, selectedRunId]);

  const detailRecord = selectedResult?.records.find((record) => record.id === recordId);
  if (detailRecord) {
    return (
      <BenchmarkRecordDetail
        record={detailRecord}
        onBack={() =>
          source === "marked" ? navigate({ tab: "marked" }) : onNavigate(selectedRunId)
        }
      />
    );
  }

  return (
    <main className="dataset-workspace">
      <section className="dataset-panel">
        <div className="chat-header">
          <div>
            <h1>Results</h1>
            <p>Open previously saved benchmark runs.</p>
          </div>
          <History size={22} />
        </div>

        <div className="results-browser">
          <label>
            Dataset
            <select value={datasetFilter} onChange={(event) => setDatasetFilter(event.target.value)}>
              <option value="all">All datasets</option>
              {datasetOptions.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Run A
            <select
              value={selectedRunId}
              onChange={(event) => void selectResult(event.target.value)}
            >
              <option value="" disabled>
                Select a run
              </option>
              {filteredItems.map((item) => (
                <option key={item.run_id} value={item.run_id}>
                  {truncateMiddle(benchmarkOptionLabel(item))}
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
              {filteredItems.filter((item) => item.run_id !== selectedRunId).map((item) => (
                <option key={item.run_id} value={item.run_id}>
                  {truncateMiddle(benchmarkOptionLabel(item))}
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
                : `${formatNumber(filteredItems.length)} saved runs`}
              </span>
            </div>
            <BarChart3 size={18} />
          </div>
          <div className="answer-body benchmark-output">
            {multiModelResults.length > 2 && !showMultiModel && (
              <button
                className="mb-3 inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50"
                onClick={() => setShowMultiModel(true)}
              >
                ← Back to cross-model comparison
              </button>
            )}
            {multiModelResults.length > 2 && showMultiModel ? (
              <MultiModelComparisonView
                results={multiModelResults}
                onSelectResult={(run) => {
                  setSelectedResult(run);
                  setSelectedRunId(run.run_id);
                  setShowMultiModel(false);
                }}
              />
            ) : selectedResult && comparisonResult ? (
              <BenchmarkComparisonView
                first={selectedResult}
                second={comparisonResult}
                onSelectRecord={(run, record) => onNavigate(run.run_id, record.id)}
                markedKeys={markedKeys}
                onToggleMarkedRow={onToggleMarkedRow}
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

function MarkedRowsView({
  rows,
  error,
  onOpenRecord,
  onRemoveRow,
}: {
  rows: MarkedComparisonRow[];
  error: string;
  onOpenRecord: (runId: string, recordId: string) => void;
  onRemoveRow: (key: string) => void;
}) {
  return (
    <main className="dataset-workspace">
      <section className="dataset-panel">
        <div className="chat-header">
          <div>
            <h1>Marked</h1>
            <p>Saved benchmark comparison rows for later review.</p>
          </div>
          <BookmarkCheck size={22} />
        </div>

        {error && <div className="error-box">{error}</div>}

        <section className="answer-panel">
          <div className="section-header">
            <div>
              <h2>Saved Rows</h2>
              <span>{formatNumber(rows.length)} comparison rows</span>
            </div>
            <BarChart3 size={18} />
          </div>
          <div className="answer-body benchmark-output">
            {rows.length === 0 ? (
              <div className="empty">
                Mark rows from a Results comparison table to keep them here.
              </div>
            ) : (
              <div className="record-table-wrap">
                <table className="record-table comparison-table">
                  <thead>
                    <tr>
                      <th>Question</th>
                      <th>Gold</th>
                      <th>Saved</th>
                      <th>Multi-node DAG</th>
                      <th>Single-node prompt</th>
                      <th>Multi-node impact</th>
                      <th>Gold recall impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => {
                      const focusLabel = runTypeLabel(row.focus_system);
                      const referenceLabel = runTypeLabel(row.reference_system);
                      const cosineDelta = row.focus_cosine_sim - row.reference_cosine_sim;
                      const focusRecall = row.focus_gold_recall ?? 0;
                      const referenceRecall = row.reference_gold_recall ?? 0;
                      const recallDelta = focusRecall - referenceRecall;
                      return (
                        <tr key={row.key}>
                          <td>
                            <div className="marked-question">
                              <strong>{row.question}</strong>
                              <small>
                                {row.focus_run_name || row.reference_run_name || "Saved run"} ·{" "}
                                {row.model} · seed {row.seed}
                              </small>
                            </div>
                          </td>
                          <td>{row.gold_answer}</td>
                          <td>
                            <MarkButton
                              marked
                              onToggle={(event) => {
                                event.stopPropagation();
                                onRemoveRow(row.key);
                              }}
                            />
                          </td>
                          <td>
                            <button
                              className="comparison-answer"
                              onClick={() => onOpenRecord(row.focus_run_id, row.record_id)}
                              title={`Open ${focusLabel} detail`}
                              type="button"
                            >
                              <span>{row.focus_prediction}</span>
                              <small>Gold recall {formatPercent(focusRecall)}</small>
                            </button>
                          </td>
                          <td>
                            <button
                              className="comparison-answer"
                              onClick={() => onOpenRecord(row.reference_run_id, row.record_id)}
                              title={`Open ${referenceLabel} detail`}
                              type="button"
                            >
                              <span>{row.reference_prediction}</span>
                              <small>Gold recall {formatPercent(referenceRecall)}</small>
                            </button>
                          </td>
                          <td>
                            <span className={`row-delta ${deltaClass(cosineDelta)}`}>
                              {cosineDelta > 0 ? "+" : ""}
                              {formatNumber(cosineDelta, 3)}
                            </span>
                          </td>
                          <td>
                            <span className={`row-delta ${deltaClass(recallDelta)}`}>
                              {recallDelta > 0 ? "+" : ""}
                              {formatPercent(recallDelta)}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
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

function ModelRequiredView({ action }: { action: string }) {
  return (
    <main className="dataset-workspace">
      <div className="empty">
        Loading available LLM models. {action} will be available once an execution model is loaded.
        Previous benchmark results are still available in Results.
      </div>
    </main>
  );
}

// Parse a percent input ("80") to a 0-1 fraction, or null if blank/invalid.
function bandFraction(text: string): number | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value / 100 : null;
}

// Out-of-band distance (0-1 space): 0 inside the band, else distance to the nearest edge.
function outOfBandDistance(
  score: number,
  low: number | null,
  high: number | null,
): number | null {
  if (low === null || high === null) return null;
  const lo = Math.min(low, high);
  const hi = Math.max(low, high);
  if (score < lo) return lo - score;
  if (score > hi) return score - hi;
  return 0;
}

function MetricEvaluationView() {
  const [table, setTable] = useState<AnnotationTable>();
  const [bands, setBands] = useState<Record<string, { low: string; high: string }>>({});
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const refresh = () => {
    setLoading(true);
    getAnnotationTable()
      .then((data) => {
        setTable(data);
        const seeded: Record<string, { low: string; high: string }> = {};
        for (const record of data.records) {
          seeded[record.key] = {
            low: record.human_low === null ? "" : String(Math.round(record.human_low * 1000) / 10),
            high:
              record.human_high === null ? "" : String(Math.round(record.human_high * 1000) / 10),
          };
        }
        setBands(seeded);
        setError("");
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    refresh();
  }, []);

  const metrics = table?.metrics ?? [];
  const records = table?.records ?? [];

  // Live sum of out-of-band error per metric, recomputed from the current band inputs.
  const errorSums: Record<string, number> = {};
  for (const metric of metrics) {
    let sum = 0;
    for (const record of records) {
      const band = bands[record.key];
      const distance = outOfBandDistance(
        record.scores[metric] ?? 0,
        band ? bandFraction(band.low) : null,
        band ? bandFraction(band.high) : null,
      );
      if (distance !== null) sum += distance;
    }
    errorSums[metric] = sum;
  }
  const bestMetric =
    metrics.length > 0
      ? metrics.reduce((best, metric) => (errorSums[metric] < errorSums[best] ? metric : best))
      : undefined;

  const updateBand = (key: string, edge: "low" | "high", value: string) => {
    setBands((previous) => {
      const current = previous[key] ?? { low: "", high: "" };
      return { ...previous, [key]: { ...current, [edge]: value } };
    });
  };

  const persistBand = (key: string) => {
    const band = bands[key];
    saveAnnotationBand(key, bandFraction(band?.low ?? ""), bandFraction(band?.high ?? "")).catch(
      () => {},
    );
  };

  return (
    <main className="dataset-workspace">
      <div className="section-header">
        <div>
          <h2>Metric evaluation</h2>
          <span>
            {table
              ? `${records.length} annotation records — type each acceptable score band (%); the top row sums every metric's out-of-band error (lower is better).`
              : "Loading…"}
          </span>
        </div>
        <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
          Refresh
        </Button>
      </div>
      {error && <div className="empty">{error}</div>}
      {!loading && !error && records.length === 0 && (
        <div className="empty">
          No annotation records yet. Open a benchmark result and click the clipboard on the rows you
          want to annotate, then return here to set their bands.
        </div>
      )}
      {records.length > 0 && (
        <div className="record-table-wrap">
          <table className="record-table">
            <thead>
              <tr>
                <th>Question</th>
                <th>Gold</th>
                <th>Prediction</th>
                <th>Low %</th>
                <th>High %</th>
                {metrics.map((metric) => (
                  <th key={metric}>{humanizeMetric(metric)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={5} style={{ fontWeight: 600 }}>
                  Sum of out-of-band error
                </td>
                {metrics.map((metric) => (
                  <td
                    key={metric}
                    style={{
                      fontWeight: metric === bestMetric ? 700 : 500,
                      color: metric === bestMetric ? "#065f46" : undefined,
                    }}
                  >
                    {formatNumber(errorSums[metric] * 100, 1)}
                  </td>
                ))}
              </tr>
              {records.map((record) => {
                const band = bands[record.key] ?? { low: "", high: "" };
                const low = bandFraction(band.low);
                const high = bandFraction(band.high);
                return (
                  <tr key={record.key}>
                    <td>{record.question}</td>
                    <td>{record.gold_answer}</td>
                    <td>{record.prediction}</td>
                    <td>
                      <input
                        type="number"
                        min={0}
                        max={100}
                        step={1}
                        value={band.low}
                        onChange={(event) => updateBand(record.key, "low", event.target.value)}
                        onBlur={() => persistBand(record.key)}
                        style={{ width: 64 }}
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        min={0}
                        max={100}
                        step={1}
                        value={band.high}
                        onChange={(event) => updateBand(record.key, "high", event.target.value)}
                        onBlur={() => persistBand(record.key)}
                        style={{ width: 64 }}
                      />
                    </td>
                    {metrics.map((metric) => {
                      const distance = outOfBandDistance(record.scores[metric] ?? 0, low, high);
                      const background =
                        distance === null ? undefined : distance === 0 ? "#dcfce7" : "#fee2e2";
                      return (
                        <td key={metric} style={{ background }}>
                          {formatPercent(record.scores[metric] ?? 0)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}

function App() {
  const [route, setRoute] = useState<AppRoute>(() => parseRoute());
  const [modelCatalog, setModelCatalog] = useState<LLMModelCatalog>();
  const [selectedLLM, setSelectedLLM] = useState<LLMSelection>();
  const [modelError, setModelError] = useState("");
  const [markedRows, setMarkedRows] = useState<MarkedComparisonRow[]>([]);
  const [markedRowsError, setMarkedRowsError] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => window.localStorage.getItem("dagqa-sidebar-collapsed") === "true",
  );
  const markedKeys = useMemo(() => new Set(markedRows.map((row) => row.key)), [markedRows]);

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

  useEffect(() => {
    listMarkedBenchmarkRows()
      .then(async (response) => {
        let rows = response.rows;
        const legacyRows = loadLegacyMarkedRows();
        const persistedKeys = new Set(rows.map((row) => row.key));
        const rowsToMigrate = legacyRows.filter((row) => !persistedKeys.has(row.key));
        for (const row of rowsToMigrate) {
          const saved = await saveMarkedBenchmarkRow(row);
          rows = saved.rows;
        }
        if (rowsToMigrate.length > 0) {
          window.localStorage.removeItem(LEGACY_MARKED_RECORDS_STORAGE_KEY);
        }
        setMarkedRows(rows);
        setMarkedRowsError("");
      })
      .catch((err) =>
        setMarkedRowsError(err instanceof Error ? err.message : "Could not load marked rows"),
      );
  }, []);

  function toggleSidebar() {
    setSidebarCollapsed((current) => {
      const next = !current;
      window.localStorage.setItem("dagqa-sidebar-collapsed", String(next));
      return next;
    });
  }

  async function toggleMarkedRow(
    focus: HotpotBenchmarkResult,
    reference: HotpotBenchmarkResult,
    focusRecord: HotpotBenchmarkRecord,
    referenceRecord: HotpotBenchmarkRecord,
  ) {
    const row = markedComparisonFromRow(focus, reference, focusRecord, referenceRecord);
    const previous = markedRows;
    const isMarked = previous.some((item) => item.key === row.key);
    const optimisticRows = isMarked
      ? previous.filter((item) => item.key !== row.key)
      : [row, ...previous];
    setMarkedRows(optimisticRows);
    setMarkedRowsError("");
    try {
      const response = isMarked
        ? await deleteMarkedBenchmarkRow(row.key)
        : await saveMarkedBenchmarkRow(row);
      setMarkedRows(response.rows);
    } catch (err) {
      setMarkedRows(previous);
      setMarkedRowsError(err instanceof Error ? err.message : "Could not update marked rows");
    }
  }

  async function removeMarkedRow(key: string) {
    const previous = markedRows;
    setMarkedRows((current) => current.filter((row) => row.key !== key));
    setMarkedRowsError("");
    try {
      const response = await deleteMarkedBenchmarkRow(key);
      setMarkedRows(response.rows);
    } catch (err) {
      setMarkedRows(previous);
      setMarkedRowsError(err instanceof Error ? err.message : "Could not update marked rows");
    }
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
          <button
            className={route.tab === "marked" ? "active" : ""}
            onClick={() => navigate({ tab: "marked" })}
            title="Marked"
          >
            <BookmarkCheck size={18} />
            <span>Marked</span>
          </button>
          <button
            className={route.tab === "metrics" ? "active" : ""}
            onClick={() => navigate({ tab: "metrics" })}
            title="Metric evaluation"
          >
            <BarChart3 size={18} />
            <span>Metrics</span>
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
      {route.tab === "results" ? (
        <ResultsView
          runId={route.runId}
          recordId={route.recordId}
          source={route.source}
          onNavigate={(runId, recordId) => navigate({ tab: "results", runId, recordId })}
          markedKeys={markedKeys}
          onToggleMarkedRow={toggleMarkedRow}
        />
      ) : route.tab === "marked" ? (
        <MarkedRowsView
          rows={markedRows}
          error={markedRowsError}
          onOpenRecord={(runId, recordId) =>
            navigate({ tab: "results", runId, recordId, source: "marked" })
          }
          onRemoveRow={removeMarkedRow}
        />
      ) : route.tab === "metrics" ? (
        <MetricEvaluationView />
      ) : !selectedLLM ? (
        <ModelRequiredView action={route.tab === "chat" ? "Chat" : "Benchmark actions"} />
      ) : route.tab === "chat" ? (
        <ChatView llm={selectedLLM} />
      ) : route.tab === "dataset" ? (
        <DatasetView
          recordId={route.recordId}
          onSelectRecord={(recordId) => navigate({ tab: "dataset", recordId })}
          llm={selectedLLM}
          modelCatalog={modelCatalog}
          markedKeys={markedKeys}
          onToggleMarkedRow={toggleMarkedRow}
        />
      ) : (
        <ModelRequiredView action="This action" />
      )}
    </div>
  );
}

export default App;

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
