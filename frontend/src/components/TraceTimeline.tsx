import type { TraceStep } from "../api/chat";
import "./TraceTimeline.css";

type TimelineStatus = "idle" | "streaming" | "complete" | "error";

type TraceTimelineProps = {
  trace: TraceStep[];
  status?: TimelineStatus;
};

type SummaryRow = {
  label: string;
  value: string;
  emphasis?: boolean;
};

const STEP_TITLES: Record<string, string> = {
  load_context: "\u4e0a\u4e0b\u6587\u8bfb\u53d6",
  follow_up_rewrite: "\u8ffd\u95ee\u6539\u5199",
  query_understanding: "\u610f\u56fe\u8bc6\u522b",
  route_by_intent: "\u610f\u56fe\u8def\u7531",
  product_retrieval: "\u5546\u54c1\u53ec\u56de",
  knowledge_retrieval: "\u77e5\u8bc6\u68c0\u7d22",
  product_comparison: "\u5019\u9009\u5546\u54c1\u6bd4\u8f83",
  response_compose: "\u56de\u7b54\u751f\u6210",
  response_composer: "\u56de\u7b54\u7ec4\u88c5",
  llm_answer: "LLM \u56de\u7b54",
  answer_draft_delta: "LLM \u8349\u7a3f",
  answer_grounding_guard: "\u56de\u7b54\u53ef\u4fe1\u6821\u9a8c",
  conversation_memory: "\u4f1a\u8bdd\u4fdd\u5b58",
  agent_workflow: "Agent \u5de5\u4f5c\u6d41",
  node_start: "节点开始",
  node_progress: "节点进度",
  node_end: "节点完成",
  retrieval: "检索过程",
  stream_guard: "流式安全守卫",
  error: "错误事件",
  save_trace: "Trace \u8bb0\u5f55",
  clarification: "\u6f84\u6e05\u56de\u7b54",
  chitchat: "\u95f2\u804a\u56de\u7b54",
};

const SUCCESS_STATUSES = new Set([
  "success",
  "ok",
  "completed",
  "saved",
  "loaded",
  "compared",
  "composed",
  "recorded",
  "routed",
]);

export function TraceTimeline({
  trace,
  status = "idle",
}: TraceTimelineProps) {
  return (
    <section className="panel trace-timeline-panel">
      <div className="timeline-heading">
        <div>
          <h2>Agent Timeline</h2>
          <p className="timeline-subtitle">
            Visual workflow steps from AgentWorkflow trace events.
          </p>
        </div>
        <span className={`timeline-run-status timeline-run-${status}`}>
          {formatRunStatus(status)}
        </span>
      </div>

      {trace.length > 0 ? (
        <ol className="timeline-list">
          {trace.map((step, index) => {
            const normalizedStep = normalizeStepName(step);
            const statusText = getStatusText(step);
            const rows = buildSummaryRows(normalizedStep, step);

            return (
              <li className="timeline-item" key={`${normalizedStep}-${index}`}>
                <div className="timeline-marker">{index + 1}</div>
                <article
                  className={`timeline-card timeline-card-${statusKind(statusText)}`}
                >
                  <div className="timeline-card-header">
                    <div>
                      <h3>
                        {STEP_TITLES[normalizedStep] ?? "\u672a\u77e5\u8282\u70b9"}
                        <span>{normalizedStep}</span>
                      </h3>
                    </div>
                    <span className={`status-pill status-${statusKind(statusText)}`}>
                      {formatStepStatus(statusText)}
                    </span>
                  </div>

                  {rows.length > 0 ? (
                    <dl className="timeline-summary">
                      {rows.map((row) => (
                        <div
                          className={row.emphasis ? "summary-row-emphasis" : ""}
                          key={`${row.label}-${row.value}`}
                        >
                          <dt>{row.label}</dt>
                          <dd>{row.value}</dd>
                        </div>
                      ))}
                    </dl>
                  ) : (
                    <p className="timeline-muted">No compact summary available.</p>
                  )}

                  <details className="timeline-json">
                    <summary>View raw JSON</summary>
                    <pre>{JSON.stringify(step, null, 2)}</pre>
                  </details>
                </article>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="muted">No timeline events yet.</p>
      )}
    </section>
  );
}

function normalizeStepName(step: TraceStep): string {
  const stepName = stringValue(step.step, "unknown_step");
  if (stepName === "agent_node") {
    return stringValue(step.node, "agent_node");
  }
  return stepName;
}

function getStatusText(step: TraceStep): string {
  return stringValue(step.status, "success");
}

function buildSummaryRows(stepName: string, step: TraceStep): SummaryRow[] {
  if (stepName === "follow_up_rewrite") {
    return compactRows([
      row("Original query", step.original_query),
      row("Actual retrieval query", step.rewritten_query, true),
      row("Reason", step.reason),
      row("Source turn", step.source_turn_index),
      row("Referenced products", step.referenced_product_ids),
      row("Resolved products", step.resolved_product_ids),
    ]);
  }

  if (stepName === "node_start") {
    return compactRows([
      row("Node", step.node, true),
      row("Label", step.label),
      row("Started at", step.started_at),
      row("Status", step.status),
    ]);
  }

  if (stepName === "node_end") {
    return compactRows([
      row("Node", step.node, true),
      row("Label", step.label),
      row("Status", step.status),
      row("Duration", formatDuration(step.duration_ms), true),
      row("Summary", step.summary),
    ]);
  }

  if (stepName === "node_progress") {
    return compactRows([
      row("Node", step.node, true),
      row("Status", step.status),
      row("Message", step.message),
    ]);
  }

  if (stepName === "retrieval") {
    return compactRows([
      row("Type", step.type, true),
      row("Status", step.status),
      row("Cache", step.cache_status),
      row("Query", step.query),
      row("Returned products", step.returned_products),
      row("Product IDs", step.candidate_product_ids ?? step.returned_product_ids),
      row("Returned chunks", step.returned_chunks),
      row("Chunk IDs", step.chunk_ids),
      row("Missing products", step.missing_product_ids),
    ]);
  }

  if (stepName === "stream_guard") {
    return compactRows([
      row("Node", step.node, true),
      row("Status", step.status),
      row("Reason", step.reason, true),
      row("Matched phrase", step.matched_phrase),
      row("Severity", step.severity),
    ]);
  }

  if (stepName === "error") {
    return compactRows([
      row("Failed node", step.failed_node, true),
      row("Error type", step.error_type),
      row("Message", step.message, true),
      row("Duration", formatDuration(step.duration_ms)),
    ]);
  }

  if (stepName === "query_understanding") {
    return compactRows([
      row("Intent", step.intent),
      row("Category", joinParts([step.category_id, step.category_path])),
      row("Budget min", step.budget_min),
      row("Budget max", step.budget_max),
      row("Preferences", step.preferences),
      row("Need clarification", step.need_clarification),
    ]);
  }

  if (stepName === "product_retrieval") {
    return compactRows([
      row("Category", step.category_id),
      row("Budget min", step.budget_min),
      row("Budget max", step.budget_max),
      row("Candidates", step.candidate_count),
      row("Product IDs", step.product_ids),
    ]);
  }

  if (stepName === "knowledge_retrieval") {
    return compactRows([
      row("Category", step.category_id),
      row("Citations", step.citation_count),
    ]);
  }

  if (stepName === "product_comparison") {
    return compactRows([
      row("Source", step.source),
      row("Requested products", step.requested_product_ids, true),
      row("Returned products", step.returned_product_ids, true),
      row("Missing products", step.missing_product_ids),
      row("Focus", step.focus_preferences),
    ]);
  }

  if (stepName === "response_compose" || stepName === "response_composer") {
    return compactRows([
      row("Status", step.status),
      row("Answer source", step.answer_source),
      row("LLM used", step.llm_used),
      row("Product cards", step.product_count ?? step.product_card_count),
      row("Citations", step.citation_count),
    ]);
  }

  if (stepName === "llm_answer") {
    return compactRows([
      row("Enabled", step.enabled),
      row("Provider", step.provider),
      row("Status", step.status),
    ]);
  }

  if (stepName === "conversation_memory") {
    return compactRows([
      row("Status", step.status),
      row("Session", step.session_id),
      row("Turn index", step.turn_index),
      row("Note", isFailure(step.status) ? "Memory save failed; response still returned." : null),
    ]);
  }

  if (stepName === "load_context") {
    return compactRows([
      row("Status", step.status),
      row("Recent turns", step.turn_count),
    ]);
  }

  if (stepName === "route_by_intent") {
    return compactRows([row("Route", step.route), row("Status", step.status)]);
  }

  if (stepName === "agent_workflow") {
    return compactRows([
      row("Status", step.status),
      row("Error", step.error, true),
    ]);
  }

  if (stepName === "save_trace") {
    return compactRows([
      row("Status", step.status),
      row("Trace count", step.trace_count),
    ]);
  }

  return compactRows([row("Status", step.status), row("Node", step.node)]);
}

function row(
  label: string,
  value: unknown,
  emphasis = false,
): SummaryRow | null {
  const formatted = formatValue(value);
  if (!formatted) {
    return null;
  }
  return { label, value: formatted, emphasis };
}

function compactRows(rows: Array<SummaryRow | null>): SummaryRow[] {
  return rows.filter((item): item is SummaryRow => item !== null);
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  if (Array.isArray(value)) {
    return value.length > 0 ? value.map(String).join(", ") : "";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function joinParts(values: unknown[]): string {
  return values.map(formatValue).filter(Boolean).join(" / ");
}

function formatDuration(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  return `${String(value)} ms`;
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value ? value : fallback;
}

function isFailure(value: unknown): boolean {
  return value === "failed" || value === "error";
}

function statusKind(status: string): string {
  if (isFailure(status)) {
    return "failed";
  }
  if (status === "skipped" || status === "not_follow_up" || status === "disabled") {
    return "skipped";
  }
  if (status === "rewritten") {
    return "rewritten";
  }
  if (
    status === "fallback" ||
    status === "insufficient_products" ||
    status === "blocked" ||
    status === "guarded"
  ) {
    return "warning";
  }
  if (status === "running") {
    return "rewritten";
  }
  if (SUCCESS_STATUSES.has(status)) {
    return "success";
  }
  return "neutral";
}

function formatStepStatus(status: string): string {
  if (SUCCESS_STATUSES.has(status)) {
    return "\u6210\u529f";
  }
  if (status === "skipped") {
    return "\u8df3\u8fc7";
  }
  if (status === "not_follow_up") {
    return "\u975e\u8ffd\u95ee";
  }
  if (status === "rewritten") {
    return "\u5df2\u6539\u5199";
  }
  if (status === "running") {
    return "运行中";
  }
  if (status === "failed" || status === "error") {
    return "\u5931\u8d25";
  }
  if (status === "fallback") {
    return "\u964d\u7ea7";
  }
  if (status === "blocked") {
    return "已拦截";
  }
  if (status === "guarded") {
    return "安全拦截";
  }
  if (status === "disabled") {
    return "\u5173\u95ed";
  }
  if (status === "insufficient_products") {
    return "\u5019\u9009\u4e0d\u8db3";
  }
  return status;
}

function formatRunStatus(status: TimelineStatus): string {
  if (status === "streaming") {
    return "Agent running...";
  }
  if (status === "complete") {
    return "Execution complete";
  }
  if (status === "error") {
    return "Stream failed";
  }
  return "Idle";
}
