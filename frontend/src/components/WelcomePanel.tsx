import type { ExamplePrompt } from "./chatTypes";
import "./WelcomePanel.css";

type WelcomePanelProps = {
  examples: ExamplePrompt[];
  onSelectExample: (query: string) => void;
};

export function WelcomePanel({ examples, onSelectExample }: WelcomePanelProps) {
  return (
    <section className="welcome-panel">
      <div className="welcome-copy">
        <p className="welcome-kicker">RAG + LangGraph Agent</p>
        <h2>SmartBuyAgent</h2>
        <p>
          你的新零售智能导购助手。可以问商品推荐、预算追问、候选比较，
          也可以查看每条回答背后的 Agent Timeline。
        </p>
      </div>

      <div className="welcome-examples">
        {examples.map((example) => (
          <button
            className="welcome-example"
            key={example.label}
            type="button"
            onClick={() => onSelectExample(example.query)}
          >
            <span>{example.label}</span>
            <strong>{example.query}</strong>
            {example.description ? <small>{example.description}</small> : null}
          </button>
        ))}
      </div>
    </section>
  );
}
