import { useState, type FormEvent } from "react";

import { sendChatMessage, type ChatResponse } from "./api/chat";
import { AnswerPanel } from "./components/AnswerPanel";
import { CitationList } from "./components/CitationList";
import { ExampleQueries, type ExampleQuery } from "./components/ExampleQueries";
import { ProductCardList } from "./components/ProductCardList";
import { RawJsonPanel } from "./components/RawJsonPanel";
import { TracePanel } from "./components/TracePanel";
import "./App.css";

const DEFAULT_QUERY = "预算3000，推荐一款拍照好的手机";
const EXAMPLE_QUERIES: ExampleQuery[] = [
  {
    label: "手机导购",
    query: "预算3000，推荐一款拍照好的手机",
    description: "触发商品召回和知识 citation",
  },
  {
    label: "鞋靴导购",
    query: "500以内，想买一双通勤防滑的鞋",
    description: "测试鞋靴品类和预算过滤",
  },
  {
    label: "护肤导购",
    query: "敏感肌用什么保湿修护面霜，预算300以内",
    description: "测试护肤品类和偏好解析",
  },
  {
    label: "知识问答",
    query: "为什么手机拍照不能只看像素",
    description: "只触发知识文档 citation 召回",
  },
  {
    label: "澄清追问",
    query: "推荐一下",
    description: "测试缺少品类时的澄清链路",
  },
];

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
};

function App() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [debug, setDebug] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleExampleSelect = (exampleQuery: string) => {
    setQuery(exampleQuery);
    setError(null);
  };

  const handleNewSession = () => {
    setSessionId(null);
    setMessages([]);
    setResponse(null);
    setError(null);
    setLoading(false);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const request = sessionId
        ? { query, debug, session_id: sessionId }
        : { query, debug };
      const result = await sendChatMessage(request);
      const nextSessionId = result.session_id ?? sessionId;

      setSessionId(nextSessionId);
      setResponse(result);
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: currentMessages.length + 1,
          role: "user",
          content: query.trim(),
        },
        {
          id: currentMessages.length + 2,
          role: "assistant",
          content: result.answer,
        },
      ]);
    } catch (err) {
      setResponse(null);
      setError(
        err instanceof Error
          ? `请求失败：${err.message}`
          : "请求失败：未知错误",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page-shell">
      <div className="debug-layout">
        <header className="page-header">
          <div>
            <h1>SmartBuyAgent Web Debug</h1>
            <p className="session-line">
              当前 session_id：
              <span className={sessionId ? "session-id" : "session-empty"}>
                {sessionId ?? "未创建"}
              </span>
            </p>
          </div>
          <button
            className="secondary-button"
            type="button"
            onClick={handleNewSession}
            disabled={loading}
          >
            新会话
          </button>
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
            <ExampleQueries
              examples={EXAMPLE_QUERIES}
              onSelect={handleExampleSelect}
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

          {loading ? (
            <p className="request-status">正在请求后端 /api/chat...</p>
          ) : null}
          {!loading && response ? (
            <p className="request-status">
              请求完成：返回 {response.product_cards.length} 个商品卡片，
              {response.citations.length} 条 citation，{response.trace.length} 个 trace step
            </p>
          ) : null}
          {error ? <p className="error-message">{error}</p> : null}
        </section>

        <section className="panel">
          <h2>Conversation</h2>
          {messages.length > 0 ? (
            <div className="message-list">
              {messages.map((message) => (
                <article
                  className={`message-item message-${message.role}`}
                  key={message.id}
                >
                  <p className="message-role">
                    {message.role === "user" ? "User" : "Assistant"}
                  </p>
                  <p className="message-content">{message.content}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="muted">暂无对话消息</p>
          )}
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
