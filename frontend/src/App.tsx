import { useState, type FormEvent } from "react";

import { sendChatMessage, type ChatResponse } from "./api/chat";
import { AnswerPanel } from "./components/AnswerPanel";
import { CitationList } from "./components/CitationList";
import { ProductCardList } from "./components/ProductCardList";
import { RawJsonPanel } from "./components/RawJsonPanel";
import { TracePanel } from "./components/TracePanel";
import "./App.css";

const DEFAULT_QUERY = "预算3000，推荐一款拍照好的手机";

function App() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [debug, setDebug] = useState(true);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const result = await sendChatMessage({ query, debug });
      setResponse(result);
    } catch (err) {
      setResponse(null);
      setError(err instanceof Error ? `请求失败：${err.message}` : "请求失败：未知错误");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page-shell">
      <div className="debug-layout">
        <header className="page-header">
          <h1>SmartBuyAgent Web Debug</h1>
        </header>

        <section className="panel">
          <form className="query-form" onSubmit={handleSubmit}>
            <label className="field-label" htmlFor="query">
              Query
            </label>
            <textarea
              id="query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              rows={5}
            />

            <div className="form-actions">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={debug}
                  onChange={(event) => setDebug(event.target.checked)}
                />
                Debug
              </label>
              <button type="submit" disabled={loading}>
                {loading ? "发送中..." : "发送请求"}
              </button>
            </div>
          </form>

          {error ? <p className="error-message">{error}</p> : null}
        </section>

        <AnswerPanel answer={response?.answer} />
        <ProductCardList productCards={response?.product_cards ?? []} />
        <CitationList citations={response?.citations ?? []} />
        <TracePanel trace={response?.trace ?? []} />
        <RawJsonPanel data={response} />
      </div>
    </main>
  );
}

export default App;
