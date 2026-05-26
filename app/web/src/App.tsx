import {
  Activity,
  BarChart3,
  CheckCircle2,
  Database,
  GitBranch,
  Loader2,
  MessageSquare,
  Play,
  Send,
  Timer,
} from "lucide-react";
import mermaid from "mermaid";
import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  benchmark,
  getLiveAsk,
  startLiveAsk,
  type LiveRun,
  type NodeTrace,
  type RunTrace,
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

type Tab = "chat" | "dataset";
type RunPhase = "idle" | "planning" | "executing" | "complete" | "error";

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

function GraphView({ mermaidText, phase }: { mermaidText?: string; phase: RunPhase }) {
  const [svg, setSvg] = useState("");

  useEffect(() => {
    if (!mermaidText) {
      setSvg("");
      return;
    }
    mermaid.render("dag-graph", mermaidText).then((result) => setSvg(result.svg));
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
            <span>Azure is creating a structured DAG plan. The graph will appear as soon as planning finishes.</span>
            <div className="progress-rail">
              <span />
            </div>
          </div>
        ) : svg ? (
          <div dangerouslySetInnerHTML={{ __html: svg }} />
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

function NodeInspector({ node }: { node?: NodeTrace }) {
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
          <div>
            <label>Question</label>
            <pre>{node.resolved_question ?? ""}</pre>
          </div>
          <div>
            <label>Returned Value</label>
            <pre>{JSON.stringify(node.returned_value, null, 2)}</pre>
          </div>
          <div>
            <label>Raw Response</label>
            <pre>{formatRawResponse(node.raw_response)}</pre>
          </div>
        </div>
      )}
    </section>
  );
}

function ChatView() {
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [run, setRun] = useState<LiveRun | null>(null);
  const [selectedNode, setSelectedNode] = useState<NodeTrace | undefined>();
  const [phase, setPhase] = useState<RunPhase>("idle");
  const [error, setError] = useState("");
  const busy = phase === "planning" || phase === "executing";

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
    setSelectedNode(undefined);
    try {
      const started = await startLiveAsk(question.trim());
      setRun(started);
      setPhase(started.phase);

      let current = started;
      while (current.phase === "planning" || current.phase === "executing") {
        await new Promise((resolve) => window.setTimeout(resolve, 700));
        current = await getLiveAsk(started.run_id);
        setRun(current);
        setPhase(current.phase);
        if (!selectedNode && current.nodes.length > 0) {
          setSelectedNode(current.nodes[0]);
        }
      }

      if (current.phase === "error") {
        setError(current.error ?? "Request failed");
      } else {
        setSelectedNode(current.nodes[0]);
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
          <span className="model-pill">Azure GPT-4o</span>
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
        <GraphView mermaidText={run?.mermaid ?? undefined} phase={phase} />
        <TracePanel nodes={run?.nodes ?? []} selected={selectedNode?.node_id} onSelect={setSelectedNode} />
        <NodeInspector node={selectedNode} />
      </aside>
    </main>
  );
}

function DatasetView() {
  const [limit, setLimit] = useState(5);
  const [system, setSystem] = useState("dag_agent");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<unknown>();
  const [error, setError] = useState("");

  async function runDatasetBenchmark() {
    setBusy(true);
    setError("");
    try {
      setResult(await benchmark(limit, system));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="dataset-workspace">
      <section className="dataset-panel">
        <div className="chat-header">
          <div>
            <h1>Dataset</h1>
            <p>Run a benchmark sample against the configured HotpotQA dataset pipeline.</p>
          </div>
          <Database size={22} />
        </div>

        <div className="benchmark-form">
          <label>
            System
            <select value={system} onChange={(event) => setSystem(event.target.value)}>
              <option value="dag_agent">DAG agent</option>
              <option value="single_shot">Single shot</option>
            </select>
          </label>
          <label>
            Limit
            <input
              min={1}
              max={50}
              type="number"
              value={limit}
              onChange={(event) => setLimit(Number(event.target.value))}
            />
          </label>
          <button className="primary-button" disabled={busy} onClick={runDatasetBenchmark}>
            {busy ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            {busy ? "Running" : "Start benchmark"}
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}

        <section className="answer-panel">
          <div className="section-header">
            <div>
              <h2>Benchmark Result</h2>
              <span>{result ? "Completed run" : "No benchmark run yet"}</span>
            </div>
            <BarChart3 size={18} />
          </div>
          <div className="answer-body benchmark-output">
            {result ? <pre>{JSON.stringify(result, null, 2)}</pre> : <div className="empty">Metrics will appear here.</div>}
          </div>
        </section>
      </section>
    </main>
  );
}

function App() {
  const [tab, setTab] = useState<Tab>("chat");

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">DQ</div>
          <div>
            <strong>DAG QA</strong>
            <span>Multi-hop evaluation</span>
          </div>
        </div>
        <nav className="tab-list">
          <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>
            <MessageSquare size={18} />
            Chat
          </button>
          <button className={tab === "dataset" ? "active" : ""} onClick={() => setTab("dataset")}>
            <Database size={18} />
            Dataset
          </button>
        </nav>
        <div className="sidebar-footer">
          <Timer size={16} />
          <span>Runs use the active backend config.</span>
        </div>
      </aside>
      {tab === "chat" ? <ChatView /> : <DatasetView />}
    </div>
  );
}

export default App;

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
