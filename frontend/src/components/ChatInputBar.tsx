import type { KeyboardEvent } from "react";
import "./ChatInputBar.css";

type ChatInputBarProps = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSend: () => void;
  onStreamSend: () => void;
};

export function ChatInputBar({
  value,
  loading,
  onChange,
  onSend,
  onStreamSend,
}: ChatInputBarProps) {
  const isBlank = !value.trim();

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!loading && !isBlank) {
        onSend();
      }
    }
  };

  return (
    <footer className="chat-input-shell">
      <div className="chat-input-card">
        <textarea
          aria-label="Chat query"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入你的导购问题，Enter 发送，Shift + Enter 换行"
          rows={3}
          disabled={loading}
        />
        <div className="chat-input-actions">
          <span>当前请求会自动携带本会话的 session_id</span>
          <div>
            <button type="button" onClick={onSend} disabled={loading || isBlank}>
              {loading ? "Sending..." : "Send"}
            </button>
            <button
              className="stream-button"
              type="button"
              onClick={onStreamSend}
              disabled={loading || isBlank}
            >
              {loading ? "Streaming..." : "Stream"}
            </button>
          </div>
        </div>
      </div>
    </footer>
  );
}
