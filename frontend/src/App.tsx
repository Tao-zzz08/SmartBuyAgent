import { useState, type FormEvent } from "react";

import { sendChatMessage, type ChatResponse } from "./api/chat";
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

        <section className="panel">
          <h2>Answer</h2>
          <p className="answer-text">{response?.answer ?? "暂无回答"}</p>
        </section>

        <section className="panel">
          <h2>Product Cards</h2>
          {response && response.product_cards.length > 0 ? (
            <div className="product-list">
              {response.product_cards.map((card) => (
                <article className="product-card" key={card.product_id}>
                  <div className="product-header">
                    <h3>{card.title}</h3>
                    <span className="price">¥{card.price}</span>
                  </div>
                  <p className="muted">{card.brand ?? "未知品牌"}</p>
                  <p>{card.recommend_reason}</p>

                  <div className="tag-list">
                    {card.tags.map((tag) => (
                      <span className="tag" key={tag}>
                        {tag}
                      </span>
                    ))}
                  </div>

                  <dl className="attribute-list">
                    {Object.entries(card.attributes).map(([key, value]) => (
                      <div key={key}>
                        <dt>{key}</dt>
                        <dd>{value}</dd>
                      </div>
                    ))}
                  </dl>

                  <div className="link-row">
                    {card.source_url ? (
                      <a href={card.source_url} target="_blank" rel="noreferrer">
                        source_url
                      </a>
                    ) : null}
                    {card.compare_url ? (
                      <a href={card.compare_url} target="_blank" rel="noreferrer">
                        compare_url
                      </a>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <p className="muted">暂无商品卡片</p>
          )}
        </section>

        <section className="panel">
          <h2>Citations</h2>
          {response && response.citations.length > 0 ? (
            <div className="citation-list">
              {response.citations.map((citation) => (
                <article className="citation-item" key={citation.chunk_id}>
                  <h3>{citation.title ?? citation.chunk_id}</h3>
                  <p className="muted">{citation.section ?? "未命名章节"}</p>
                  <p className="muted">{citation.source_file ?? "未知来源"}</p>
                  <p>{citation.content_preview}</p>
                  <p className="score">score: {citation.score.toFixed(4)}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="muted">暂无 citation</p>
          )}
        </section>

        <section className="panel">
          <h2>Trace</h2>
          {response && response.trace.length > 0 ? (
            <div className="trace-list">
              {response.trace.map((step, index) => (
                <pre key={`${String(step.step ?? "step")}-${index}`}>
                  {JSON.stringify(step, null, 2)}
                </pre>
              ))}
            </div>
          ) : (
            <p className="muted">暂无 trace</p>
          )}
        </section>

        <section className="panel">
          <h2>Raw JSON</h2>
          <pre className="raw-json">
            {response ? JSON.stringify(response, null, 2) : "暂无响应"}
          </pre>
        </section>
      </div>
    </main>
  );
}

export default App;
