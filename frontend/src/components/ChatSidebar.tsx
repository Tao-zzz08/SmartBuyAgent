import type { ChatSession } from "./chatTypes";
import "./ChatSidebar.css";

type ChatSidebarProps = {
  sessions: ChatSession[];
  activeSessionId: string;
  onNewSession: () => void;
  onSelectSession: (localId: string) => void;
};

export function ChatSidebar({
  sessions,
  activeSessionId,
  onNewSession,
  onSelectSession,
}: ChatSidebarProps) {
  return (
    <aside className="chat-sidebar">
      <div className="sidebar-brand">
        <span className="brand-mark">S</span>
        <div>
          <h1>SmartBuyAgent</h1>
          <p>RAG Agent Guide</p>
        </div>
      </div>

      <button className="new-session-button" type="button" onClick={onNewSession}>
        + 新会话
      </button>

      <nav className="session-list" aria-label="Chat sessions">
        {sessions.map((session) => (
          <button
            className={
              session.localId === activeSessionId
                ? "session-item active"
                : "session-item"
            }
            key={session.localId}
            type="button"
            onClick={() => onSelectSession(session.localId)}
          >
            <span className="session-title">{session.title}</span>
            <span className="session-meta">
              {session.messages.length} messages
              {session.backendSessionId
                ? ` · ${shortSessionId(session.backendSessionId)}`
                : " · local"}
            </span>
          </button>
        ))}
      </nav>
    </aside>
  );
}

function shortSessionId(sessionId: string): string {
  return sessionId.length > 8 ? sessionId.slice(0, 8) : sessionId;
}
