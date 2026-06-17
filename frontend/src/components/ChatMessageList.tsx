import { CitationList } from "./CitationList";
import { FeedbackPanel } from "./FeedbackPanel";
import { MessageDebugPanel } from "./MessageDebugPanel";
import { ProductCardList } from "./ProductCardList";
import type { ChatMessage } from "./chatTypes";
import "./ChatMessageList.css";

type ChatMessageListProps = {
  messages: ChatMessage[];
  sessionId: string | null;
};

export function ChatMessageList({ messages, sessionId }: ChatMessageListProps) {
  return (
    <div className="chat-message-list">
      {messages.map((message) =>
        message.role === "user" ? (
          <article className="chat-message user-message" key={message.id}>
            <div className="message-bubble user-bubble">
              <p>{message.content}</p>
            </div>
          </article>
        ) : (
          <article className="chat-message assistant-message" key={message.id}>
            <div className="assistant-avatar">S</div>
            <div className="assistant-body">
              {message.status === "streaming" ? (
                <p className="message-status">Agent 正在执行...</p>
              ) : null}
              {message.status === "error" ? (
                <p className="message-error">
                  {message.error ?? "Request failed"}
                </p>
              ) : (
                <p className="assistant-answer">{message.content}</p>
              )}

              {message.response ? (
                <div className="assistant-result">
                  {message.response.product_cards.length > 0 ? (
                    <ProductCardList
                      productCards={message.response.product_cards}
                    />
                  ) : null}
                  {message.response.citations.length > 0 ? (
                    <CitationList citations={message.response.citations} />
                  ) : null}
                  <FeedbackPanel
                    sessionId={sessionId}
                    query={message.relatedUserQuery ?? ""}
                    answer={message.response.answer}
                  />
                </div>
              ) : null}

              <MessageDebugPanel
                response={message.response}
                streamTrace={message.streamTrace}
                status={message.status}
                error={message.error}
              />
            </div>
          </article>
        ),
      )}
    </div>
  );
}
