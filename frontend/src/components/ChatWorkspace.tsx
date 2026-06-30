import { useMemo, useState } from "react";

import {
  sendChatMessage,
  sendChatMessageStream,
  type ChatResponse,
  type ChatStreamErrorPayload,
  type ChatStreamGroundingGuardPayload,
  type ChatStreamGuardPayload,
  type ChatStreamNodePayload,
  type ChatStreamRetrievalPayload,
  type ChatStreamTokenPayload,
  type TraceStep,
} from "../api/chat";
import { ChatInputBar } from "./ChatInputBar";
import { ChatMessageList } from "./ChatMessageList";
import { ChatSidebar } from "./ChatSidebar";
import { WelcomePanel } from "./WelcomePanel";
import type { ChatMessage, ChatSession, ExamplePrompt } from "./chatTypes";
import "./ChatWorkspace.css";

const EXAMPLE_PROMPTS: ExamplePrompt[] = [
  {
    label: "手机导购",
    query: "预算3000，推荐一款拍照好的手机",
    description: "商品召回 + 知识引用",
  },
  {
    label: "预算追问",
    query: "预算提高到4000呢",
    description: "结合上一轮会话改写 query",
  },
  {
    label: "候选比较",
    query: "第一个和第二个有什么区别",
    description: "只比较上一轮候选商品",
  },
  {
    label: "鞋靴导购",
    query: "推荐一双适合通勤的男鞋",
    description: "通勤、舒适、防滑偏好",
  },
  {
    label: "护肤知识",
    query: "敏感肌应该注意哪些成分",
    description: "日常护理和成分注意事项",
  },
];

const INITIAL_SESSION = createSession();

export function ChatWorkspace() {
  const [sessions, setSessions] = useState<ChatSession[]>([INITIAL_SESSION]);
  const [activeSessionId, setActiveSessionId] = useState(INITIAL_SESSION.localId);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);

  const activeSession = useMemo(
    () =>
      sessions.find((session) => session.localId === activeSessionId) ??
      sessions[0],
    [activeSessionId, sessions],
  );

  const handleNewSession = () => {
    const session = createSession();
    setSessions((current) => [session, ...current]);
    setActiveSessionId(session.localId);
    setInputValue("");
    setWorkspaceError(null);
  };

  const handleSelectExample = (query: string) => {
    setInputValue(query);
    setWorkspaceError(null);
  };

  const handleSend = async () => {
    const query = inputValue.trim();
    if (!query || loading || !activeSession) {
      return;
    }

    const targetSessionId = activeSession.localId;
    const backendSessionId = activeSession.backendSessionId;
    setLoading(true);
    setWorkspaceError(null);
    setInputValue("");
    appendMessage(targetSessionId, createUserMessage(query), query);

    try {
      const response = await sendChatMessage({
        query,
        debug: true,
        session_id: backendSessionId,
      });
      updateSessionBackendId(targetSessionId, response.session_id ?? null);
      appendMessage(
        targetSessionId,
        createAssistantMessage(response.answer, query, {
          response,
          status: "done",
        }),
      );
    } catch (error) {
      const message = formatError(error, "Request failed");
      appendMessage(
        targetSessionId,
        createAssistantMessage(message, query, {
          status: "error",
          error: message,
        }),
      );
      setWorkspaceError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleStreamSend = async () => {
    const query = inputValue.trim();
    if (!query || loading || !activeSession) {
      return;
    }

    const targetSessionId = activeSession.localId;
    const backendSessionId = activeSession.backendSessionId;
    const assistantMessageId = createId("assistant");
    setLoading(true);
    setWorkspaceError(null);
    setInputValue("");
    appendMessage(targetSessionId, createUserMessage(query), query);
    appendMessage(
      targetSessionId,
      createAssistantMessage("", query, {
        id: assistantMessageId,
        status: "streaming",
        streamTrace: [],
      }),
    );

    try {
      await sendChatMessageStream(
        {
          query,
          debug: true,
          session_id: backendSessionId,
        },
        {
          onSession: (payload) => {
            updateSessionBackendId(targetSessionId, payload.session_id);
          },
          onTrace: (payload) => {
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              streamTrace: appendStreamTrace(message.streamTrace, payload),
            }));
          },
          onNodeStart: (payload) => {
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              streamTrace: appendStreamTrace(
                message.streamTrace,
                streamEventTrace("node_start", payload),
              ),
            }));
          },
          onNodeProgress: (payload) => {
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              streamTrace: appendStreamTrace(
                message.streamTrace,
                streamEventTrace("node_progress", payload),
              ),
            }));
          },
          onRetrieval: (payload) => {
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              streamTrace: appendStreamTrace(
                message.streamTrace,
                retrievalTrace(payload),
              ),
            }));
          },
          onToken: (payload) => {
            // Legacy token events are also treated as debug-only in stream mode.
            // The official assistant answer comes from final_answer/result.answer.
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              streamTrace: appendStreamTrace(
                message.streamTrace,
                legacyTokenTrace(payload),
              ),
            }));
          },
          onAnswerDraftDelta: (payload) => {
            // Draft deltas are debug-only. The official assistant answer is
            // updated only by final_answer and reconciled by result.answer.
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              streamTrace: appendStreamTrace(
                message.streamTrace,
                answerDraftTrace(payload),
              ),
            }));
          },
          onStreamGuard: (payload) => {
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              streamTrace: appendStreamTrace(
                message.streamTrace,
                streamGuardTrace(payload),
              ),
            }));
          },
          onGroundingGuardResult: (payload) => {
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              streamTrace: appendStreamTrace(
                message.streamTrace,
                groundingGuardTrace(payload),
              ),
            }));
          },
          onFinalAnswer: (payload) => {
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              content: payload.answer,
            }));
          },
          onNodeEnd: (payload) => {
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              streamTrace: appendStreamTrace(
                message.streamTrace,
                streamEventTrace("node_end", payload),
              ),
            }));
          },
          onResult: (payload) => {
            updateSessionBackendId(targetSessionId, payload.session_id ?? null);
            updateMessage(targetSessionId, assistantMessageId, (message) => ({
              ...message,
              content: payload.answer,
              status: "done",
              response: payload,
              streamTrace:
                message.streamTrace && message.streamTrace.length > 0
                  ? message.streamTrace
                  : payload.trace,
            }));
          },
          onDone: (payload) => {
            if (payload.status === "error") {
              const message = "Stream ended with error";
              updateMessage(targetSessionId, assistantMessageId, (current) => ({
                ...current,
                status: "error",
                error: current.error ?? message,
              }));
              setWorkspaceError(message);
            }
            if (payload.status === "guarded") {
              updateMessage(targetSessionId, assistantMessageId, (current) => ({
                ...current,
                status: current.response ? "done" : current.status,
              }));
            }
          },
          onError: (payload) => {
            if (payload.error_type === "StreamSafetyViolation") {
              updateMessage(targetSessionId, assistantMessageId, (current) => ({
                ...current,
                streamTrace: appendStreamTrace(
                  current.streamTrace,
                  streamErrorTrace(payload),
                ),
              }));
              return;
            }

            const message = `Stream request failed: ${payload.message}`;
            updateMessage(targetSessionId, assistantMessageId, (current) => ({
              ...current,
              status: "error",
              error: message,
              content: message,
            }));
            setWorkspaceError(message);
          },
        },
      );
    } catch (error) {
      const message = formatError(error, "Stream request failed");
      updateMessage(targetSessionId, assistantMessageId, (current) => ({
        ...current,
        status: "error",
        error: message,
        content: message,
      }));
      setWorkspaceError(message);
    } finally {
      setLoading(false);
    }
  };

  const appendMessage = (
    sessionLocalId: string,
    message: ChatMessage,
    firstQueryForTitle?: string,
  ) => {
    setSessions((current) =>
      current.map((session) => {
        if (session.localId !== sessionLocalId) {
          return session;
        }

        const shouldSetTitle =
          session.messages.length === 0 &&
          message.role === "user" &&
          firstQueryForTitle;

        return {
          ...session,
          title: shouldSetTitle
            ? titleFromQuery(firstQueryForTitle)
            : session.title,
          messages: [...session.messages, message],
          updatedAt: new Date().toISOString(),
        };
      }),
    );
  };

  const updateMessage = (
    sessionLocalId: string,
    messageId: string,
    updater: (message: ChatMessage) => ChatMessage,
  ) => {
    setSessions((current) =>
      current.map((session) =>
        session.localId === sessionLocalId
          ? {
              ...session,
              messages: session.messages.map((message) =>
                message.id === messageId ? updater(message) : message,
              ),
              updatedAt: new Date().toISOString(),
            }
          : session,
      ),
    );
  };

  const updateSessionBackendId = (
    sessionLocalId: string,
    backendSessionId: string | null,
  ) => {
    if (!backendSessionId) {
      return;
    }

    setSessions((current) =>
      current.map((session) =>
        session.localId === sessionLocalId
          ? {
              ...session,
              backendSessionId,
              updatedAt: new Date().toISOString(),
            }
          : session,
      ),
    );
  };

  return (
    <div className="chat-workspace">
      <ChatSidebar
        sessions={sessions}
        activeSessionId={activeSession.localId}
        onNewSession={handleNewSession}
        onSelectSession={setActiveSessionId}
      />

      <main className="chat-main">
        <header className="chat-main-header">
          <div>
            <p className="workspace-kicker">Chat Workspace</p>
            <h1>{activeSession.title}</h1>
          </div>
          <p className="backend-session">
            backend session:
            <span>{activeSession.backendSessionId ?? "not created"}</span>
          </p>
        </header>

        <section className="chat-scroll-region">
          {activeSession.messages.length === 0 ? (
            <WelcomePanel
              examples={EXAMPLE_PROMPTS}
              onSelectExample={handleSelectExample}
            />
          ) : (
            <ChatMessageList
              messages={activeSession.messages}
              sessionId={activeSession.backendSessionId}
            />
          )}
        </section>

        {workspaceError ? (
          <p className="workspace-error">{workspaceError}</p>
        ) : null}

        <ChatInputBar
          value={inputValue}
          loading={loading}
          onChange={setInputValue}
          onSend={handleSend}
          onStreamSend={handleStreamSend}
        />
      </main>
    </div>
  );
}

function createSession(): ChatSession {
  const now = new Date().toISOString();
  return {
    localId: createId("session"),
    backendSessionId: null,
    title: "新会话",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

function createUserMessage(content: string): ChatMessage {
  return {
    id: createId("user"),
    role: "user",
    content,
    createdAt: new Date().toISOString(),
  };
}

function createAssistantMessage(
  content: string,
  relatedUserQuery: string,
  options: Partial<ChatMessage> = {},
): ChatMessage {
  return {
    id: options.id ?? createId("assistant"),
    role: "assistant",
    content,
    createdAt: new Date().toISOString(),
    status: options.status ?? "done",
    response: options.response,
    streamTrace: options.streamTrace,
    error: options.error,
    relatedUserQuery,
  };
}

function createId(prefix: string): string {
  const randomId = globalThis.crypto?.randomUUID?.();
  return randomId
    ? `${prefix}_${randomId}`
    : `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function titleFromQuery(query: string): string {
  const normalized = query.replace(/\s+/g, " ").trim();
  return normalized.length > 20 ? `${normalized.slice(0, 20)}...` : normalized;
}

function formatError(error: unknown, prefix: string): string {
  return error instanceof Error
    ? `${prefix}: ${error.message}`
    : `${prefix}: unknown error`;
}

function appendStreamTrace(
  current: TraceStep[] | undefined,
  next: TraceStep,
): TraceStep[] {
  return [...(current ?? []), next];
}

function streamEventTrace(
  eventName: string,
  payload: ChatStreamNodePayload,
): TraceStep {
  return {
    step: eventName,
    ...payload,
  };
}

function retrievalTrace(payload: ChatStreamRetrievalPayload): TraceStep {
  return {
    step: "retrieval",
    ...payload,
  };
}

function answerDraftTrace(payload: ChatStreamTokenPayload): TraceStep {
  return {
    step: "answer_draft_delta",
    status: "debug",
    node: payload.node,
    delta: payload.delta,
  };
}

function legacyTokenTrace(payload: ChatStreamTokenPayload): TraceStep {
  return {
    step: "token",
    status: "debug",
    node: payload.node,
    delta: payload.delta,
  };
}

function streamGuardTrace(payload: ChatStreamGuardPayload): TraceStep {
  return {
    step: "stream_guard",
    ...payload,
  };
}

function groundingGuardTrace(payload: ChatStreamGroundingGuardPayload): TraceStep {
  return {
    step: "answer_grounding_guard",
    status: payload.status ?? payload.action ?? "checked",
    action: payload.action,
    passed: payload.passed,
    violations: payload.violations,
  };
}

function streamErrorTrace(payload: ChatStreamErrorPayload): TraceStep {
  return {
    step: "error",
    status: "failed",
    ...payload,
  };
}
