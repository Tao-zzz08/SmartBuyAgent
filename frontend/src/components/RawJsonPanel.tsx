type RawJsonPanelProps = {
  data: unknown;
  title?: string;
};

export function RawJsonPanel({ data, title = "Raw JSON" }: RawJsonPanelProps) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      <pre className="raw-json">
        {data ? JSON.stringify(data, null, 2) : "No response yet"}
      </pre>
    </section>
  );
}
