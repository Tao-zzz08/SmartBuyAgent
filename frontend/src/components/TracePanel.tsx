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
            <article
              className="trace-item"
              key={`${String(step.step ?? "step")}-${index}`}
            >
              <p className="trace-summary">{formatTraceSummary(step)}</p>
              <pre>{JSON.stringify(step, null, 2)}</pre>
            </article>
          ))}
        </div>
      ) : (
        <p className="muted">No trace yet</p>
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
    return `query_understanding: intent=${intent}, category=${category}, budget_max=${budgetMax ?? "-"}, preferences=${preferences || "-"}`;
  }

  if (stepName === "follow_up_rewrite") {
    return `follow_up_rewrite: ${getString(step.status, "-")} -> ${getString(step.rewritten_query, "no rewrite")}`;
  }

  if (stepName === "product_retrieval") {
    return `product_retrieval: ${getNumber(step.candidate_count) ?? 0} product candidate(s)`;
  }

  if (stepName === "knowledge_retrieval") {
    return `knowledge_retrieval: ${getNumber(step.citation_count) ?? 0} citation(s)`;
  }

  if (stepName === "product_comparison") {
    return `product_comparison: ${getString(step.status, "-")}`;
  }

  if (stepName === "response_composer" || stepName === "response_compose") {
    return `response_compose: ${getNumber(step.product_count ?? step.product_card_count) ?? 0} product card(s), ${getNumber(step.citation_count) ?? 0} citation(s)`;
  }

  return `${stepName}: view raw JSON`;
}

function getString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function getNumber(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}
