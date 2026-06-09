import type { TraceStep } from "../api/chat";

type TracePanelProps = {
  trace: TraceStep[];
};

export function TracePanel({ trace }: TracePanelProps) {
  return (
    <section className="panel">
      <h2>Trace</h2>
      {trace.length > 0 ? (
        <div className="trace-list">
          {trace.map((step, index) => (
            <pre key={`${String(step.step ?? "step")}-${index}`}>
              {JSON.stringify(step, null, 2)}
            </pre>
          ))}
        </div>
      ) : (
        <p className="muted">暂无 trace</p>
      )}
    </section>
  );
}
