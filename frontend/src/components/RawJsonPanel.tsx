type RawJsonPanelProps = {
  data: unknown;
};

export function RawJsonPanel({ data }: RawJsonPanelProps) {
  return (
    <section className="panel">
      <h2>Raw JSON</h2>
      <pre className="raw-json">
        {data ? JSON.stringify(data, null, 2) : "暂无响应"}
      </pre>
    </section>
  );
}
