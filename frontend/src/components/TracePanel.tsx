import type { TraceStep } from "../api/chat";

type TracePanelProps = {
  trace: TraceStep[];
};

export function TracePanel({ trace }: TracePanelProps) {
  return (
    <section className="panel">
      <h2>Raw Trace JSON</h2>
      {trace.length > 0 ? (
        <div className="trace-list">
          {trace.map((step, index) => (
            <article className="trace-item" key={`${String(step.step ?? "step")}-${index}`}>
              <p className="trace-summary">{formatTraceSummary(step)}</p>
              <pre>{JSON.stringify(step, null, 2)}</pre>
            </article>
          ))}
        </div>
      ) : (
        <p className="muted">暂无 trace</p>
      )}
    </section>
  );
}

function formatTraceSummary(step: TraceStep): string {
  const stepName = getString(step.step, "unknown_step");

  if (stepName === "query_understanding") {
    const intent = getString(step.intent, "-");
    const category = getString(step.category_id, "-");
    const budgetMax = getNumber(step.budget_max);
    const preferences = Array.isArray(step.preferences)
      ? step.preferences.join(", ")
      : "-";
    return `query_understanding：intent=${intent}，category=${category}，budget_max=${budgetMax ?? "-"}，preferences=${preferences || "-"}`;
  }

  if (stepName === "product_retrieval") {
    return `product_retrieval：召回 ${getNumber(step.candidate_count) ?? 0} 个商品`;
  }

  if (stepName === "knowledge_retrieval") {
    return `knowledge_retrieval：召回 ${getNumber(step.citation_count) ?? 0} 条 citation`;
  }

  if (stepName === "response_composer") {
    return `response_composer：生成 ${getNumber(step.product_count) ?? 0} 个商品卡片，使用 ${getNumber(step.citation_count) ?? 0} 条 citation`;
  }

  return `${stepName}：查看原始 JSON`;
}

function getString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function getNumber(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}
