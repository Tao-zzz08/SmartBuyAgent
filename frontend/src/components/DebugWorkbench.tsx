import { useState, type FormEvent } from "react";

import {
  sendChatMessage,
  sendChatMessageStream,
  type ChatResponse,
  type TraceStep,
} from "../api/chat";
import { AnswerPanel } from "./AnswerPanel";
import { CitationList } from "./CitationList";
import { ExampleQueries, type ExampleQuery } from "./ExampleQueries";
import { FeedbackPanel } from "./FeedbackPanel";
import { ProductCardList } from "./ProductCardList";
import { RawJsonPanel } from "./RawJsonPanel";
import { TracePanel } from "./TracePanel";
import { TraceTimeline } from "./TraceTimeline";

const EXAMPLE_QUERIES: ExampleQuery[] = [
  {
    label: "\u624b\u673a\u5bfc\u8d2d",
    query:
      "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a",
    description: "Product retrieval + knowledge citation",
  },
  {
    label: "\u978b\u9774\u5bfc\u8d2d",
    query:
      "500\u4ee5\u5185\uff0c\u60f3\u4e70\u4e00\u53cc\u901a\u52e4\u9632\u6ed1\u7684\u978b",
    description: "Category and budget filter",
  },
  {
    label: "\u62a4\u80a4\u5bfc\u8d2d",
    query:
      "\u654f\u611f\u808c\u7528\u4ec0\u4e48\u4fdd\u6e7f\u4fee\u62a4\u9762\u971c\uff0c\u9884\u7b97300\u4ee5\u5185",
    description: "Skincare preferences",
  },
  {
    label: "\u77e5\u8bc6\u95ee\u7b54",
    query:
      "\u4e3a\u4ec0\u4e48\u624b\u673a\u62cd\u7167\u4e0d\u80fd\u53ea\u770b\u50cf\u7d20",
    description: "Knowledge-only path",
  },
  {
    label: "\u6f84\u6e05\u8ffd\u95ee",
    query: "\u63a8\u8350\u4e00\u4e0b",
    description: "Clarification path",
  },
  {
    label: "\u9884\u7b97\u8ffd\u95ee",
    query: "\u9884\u7b97\u63d0\u9ad8\u52304000\u5462",
    description: "Follow-up rewrite",
  },
  {
    label: "\u5019\u9009\u6bd4\u8f83",
    query:
      "\u7b2c\u4e00\u4e2a\u548c\u7b2c\u4e8c\u4e2a\u6709\u4ec0\u4e48\u533a\u522b",
    description: "In-session comparison",
  },
];

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
};

type RequestMode = "normal" | "stream" | null;

type DebugWorkbenchProps = {
  query: string;
  onQueryChange: (query: string) => void;
};

export function DebugWorkbench({ query, onQueryChange }: DebugWorkbenchProps) {
  const [debug, setDebug] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [latestResponseQuery, setLatestResponseQuery] = useState<string | null>(null);
  const [streamTrace, setStreamTrace] = useState<TraceStep[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [requestMode, setRequestMode] = useState<RequestMode>(null);

  const traceForDisplay = response?.trace ?? streamTrace;
  const timelineStatus =
    loading && requestMode === "stream"
      ? "streaming"
      : error && requestMode === "stream"
        ? "error"
        : response
          ? "complete"
          : "idle";
  const rawJsonData =
    response ??
    (requestMode === "stream" || streamTrace.length > 0
      ? {
          session_id: sessionId,
          trace: streamTrace,
          streaming: loading && requestMode === "stream",
        }
      : null);

  const handleExampleSelect = (exampleQuery: string) => {
    onQueryChange(exampleQuery);
    setError(null);
  };

  const handleNewSession = () => {
    setSessionId(null);
    setMessages([]);
    setResponse(null);
    setLatestResponseQuery(null);
    setStreamTrace([]);
    setError(null);
    setLoading(false);
    setRequestMode(null);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setRequestMode("normal");
    setError(null);
    setResponse(null);
    setLatestResponseQuery(null);
    setStreamTrace([]);

    try {
      const trimmedQuery = query.trim();
      const request = sessionId
        ? { query: trimmedQuery, debug, session_id: sessionId }
        : { query: trimmedQuery, debug };
      const result = await sendChatMessage(request);
      const nextSessionId = result.session_id ?? sessionId;

      setSessionId(nextSessionId);
      setResponse(result);
      setLatestResponseQuery(trimmedQuery);
      appendConversationTurn(trimmedQuery, result.answer);
    } catch (err) {
      setResponse(null);
      setError(formatRequestError(err, "Request failed"));
    } finally {
      setLoading(false);
    }
  };

  const handleStreamSubmit = async () => {
    const trimmedQuery = query.trim();

    if (!trimmedQuery) {
      setError("\u8bf7\u8f93\u5165\u95ee\u9898");
      return;
    }

    setLoading(true);
    setRequestMode("stream");
    setError(null);
    setResponse(null);
    setLatestResponseQuery(null);
    setStreamTrace([]);
    appendMessage("user", trimmedQuery);

    try {
      const request = sessionId
        ? { query: trimmedQuery, debug, session_id: sessionId }
        : { query: trimmedQuery, debug };

      await sendChatMessageStream(request, {
        onSession: (payload) => {
          setSessionId(payload.session_id);
        },
        onTrace: (payload) => {
          setStreamTrace((currentTrace) => [...currentTrace, payload]);
        },
        onResult: (payload) => {
          setSessionId(payload.session_id ?? sessionId);
          setResponse(payload);
          setLatestResponseQuery(trimmedQuery);
          setStreamTrace(payload.trace ?? []);
          if (payload.answer.trim()) {
            appendMessage("assistant", payload.answer);
          }
        },
        onDone: (payload) => {
          if (payload.status === "error") {
            setError((currentError) => currentError ?? "Stream ended with error");
          }
          setLoading(false);
        },
        onError: (payload) => {
          setError(`Stream request failed: ${payload.message}`);
          setLoading(false);
        },
      });
    } catch (err) {
      setError(formatRequestError(err, "Stream request failed"));
    } finally {
      setLoading(false);
    }
  };

  const appendConversationTurn = (userQuery: string, assistantAnswer: string) => {
    setMessages((currentMessages) => [
      ...currentMessages,
      createMessage("user", userQuery, currentMessages.length),
      createMessage("assistant", assistantAnswer, currentMessages.length + 1),
    ]);
  };

  const appendMessage = (role: ChatMessage["role"], content: string) => {
    setMessages((currentMessages) => [
      ...currentMessages,
      createMessage(role, content, currentMessages.length),
    ]);
  };

  return (
    <div className="debug-layout">
      <header className="page-header">
        <div>
          <h1>SmartBuyAgent Web Debug</h1>
          <p className="session-line">
            Current session_id:
            <span className={sessionId ? "session-id" : "session-empty"}>
              {sessionId ?? "not created"}
            </span>
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          onClick={handleNewSession}
          disabled={loading}
        >
          New Session
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
            onChange={(event) => onQueryChange(event.target.value)}
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
            <div className="button-row">
              <button type="submit" disabled={loading}>
                {loading && requestMode === "normal"
                  ? "Sending..."
                  : "Send request"}
              </button>
              <button
                className="stream-button"
                type="button"
                onClick={handleStreamSubmit}
                disabled={loading}
              >
                {loading && requestMode === "stream"
                  ? "Streaming..."
                  : "Stream send"}
              </button>
            </div>
          </div>
        </form>

        {loading ? (
          <p className="request-status">
            {requestMode === "stream"
              ? `Streaming from /api/chat/stream... received ${streamTrace.length} trace step(s).`
              : "Requesting backend /api/chat..."}
          </p>
        ) : null}
        {!loading && response ? (
          <p className="request-status">
            {requestMode === "stream" ? "Stream complete" : "Request complete"}:
            returned {response.product_cards.length} product card(s),{" "}
            {response.citations.length} citation(s), {response.trace.length} trace
            step(s).
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
          <p className="muted">No conversation messages yet</p>
        )}
      </section>

      <AnswerPanel answer={response?.answer} />
      <FeedbackPanel
        sessionId={sessionId}
        query={latestResponseQuery ?? ""}
        answer={response?.answer}
      />
      <ProductCardList productCards={response?.product_cards ?? []} />
      <CitationList citations={response?.citations ?? []} />
      <TraceTimeline trace={traceForDisplay} status={timelineStatus} />
      <TracePanel trace={traceForDisplay} />
      <RawJsonPanel data={rawJsonData} />
    </div>
  );
}

function createMessage(
  role: ChatMessage["role"],
  content: string,
  offset: number,
): ChatMessage {
  return {
    id: Date.now() + offset,
    role,
    content,
  };
}

function formatRequestError(err: unknown, prefix: string): string {
  return err instanceof Error
    ? `${prefix}: ${err.message}`
    : `${prefix}: unknown error`;
}
