import { useState } from "react";

import { checkHealth, type HealthResponse } from "./api/health";

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleHealthCheck = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await checkHealth();
      setHealth(result);
    } catch (err) {
      setHealth(null);
      setError(err instanceof Error ? err.message : "Health check failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#f5f7fb",
        color: "#172033",
        fontFamily:
          "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        padding: "32px",
      }}
    >
      <section
        style={{
          width: "100%",
          maxWidth: "720px",
          background: "#ffffff",
          border: "1px solid #dfe5ef",
          borderRadius: "8px",
          padding: "32px",
          boxShadow: "0 16px 40px rgba(23, 32, 51, 0.08)",
        }}
      >
        <h1 style={{ margin: 0, fontSize: "36px", lineHeight: 1.1 }}>SmartBuyAgent</h1>
        <p style={{ margin: "12px 0 24px", color: "#526078", fontSize: "18px" }}>
          多品类电商智能导购 RAG Agent
        </p>

        <button
          type="button"
          onClick={handleHealthCheck}
          disabled={loading}
          style={{
            border: 0,
            borderRadius: "6px",
            background: loading ? "#8da2c0" : "#2457d6",
            color: "#ffffff",
            cursor: loading ? "default" : "pointer",
            fontSize: "16px",
            fontWeight: 600,
            padding: "12px 18px",
          }}
        >
          {loading ? "Checking..." : "Health Check"}
        </button>

        <pre
          style={{
            minHeight: "120px",
            margin: "24px 0 0",
            padding: "16px",
            overflowX: "auto",
            background: "#101828",
            borderRadius: "8px",
            color: error ? "#ffb4a8" : "#d8f3dc",
            fontSize: "14px",
            lineHeight: 1.6,
          }}
        >
          {error ?? (health ? JSON.stringify(health, null, 2) : "Click Health Check to query the backend.")}
        </pre>
      </section>
    </main>
  );
}

export default App;
