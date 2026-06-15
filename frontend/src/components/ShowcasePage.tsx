import "./ShowcasePage.css";

type ShowcasePageProps = {
  onEnterDebug: () => void;
  onSelectExample: (query: string) => void;
};

type Scenario = {
  title: string;
  subtitle: string;
  description: string;
  prompts: string[];
};

const SCENARIOS: Scenario[] = [
  {
    title: "Phone",
    subtitle: "\u624b\u673a\u5bfc\u8d2d",
    description:
      "Budget-aware product recommendation for camera, performance, battery, and system preference scenarios.",
    prompts: [
      "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a",
      "4000\u4ee5\u5185\u9002\u5408\u6e38\u620f\u7684\u624b\u673a",
      "\u7b2c\u4e00\u4e2a\u548c\u7b2c\u4e8c\u4e2a\u6709\u4ec0\u4e48\u533a\u522b",
    ],
  },
  {
    title: "Shoes",
    subtitle: "\u978b\u9774\u5bfc\u8d2d",
    description:
      "Commute, comfort, anti-slip, material, and durability guidance for new shoes and boots.",
    prompts: [
      "\u63a8\u8350\u4e00\u53cc\u9002\u5408\u901a\u52e4\u7684\u7537\u978b",
      "\u9884\u7b97800\uff0c\u60f3\u8981\u9632\u6ed1\u8010\u7a7f\u7684\u9774\u5b50",
      "\u54ea\u53cc\u66f4\u9002\u5408\u51ac\u5929\u7a7f",
    ],
  },
  {
    title: "Skincare",
    subtitle: "\u62a4\u80a4\u5bfc\u8d2d",
    description:
      "Daily skincare guidance for skin type, texture preference, ingredients, and routine explanations.",
    prompts: [
      "\u63a8\u8350\u9002\u5408\u6cb9\u76ae\u7684\u57fa\u7840\u62a4\u80a4\u54c1",
      "\u654f\u611f\u808c\u5e94\u8be5\u6ce8\u610f\u54ea\u4e9b\u6210\u5206",
      "\u8fd9\u4e24\u4e2a\u4ea7\u54c1\u6709\u4ec0\u4e48\u533a\u522b",
    ],
  },
];

const CAPABILITIES = [
  {
    title: "\u591a\u8f6e\u8ffd\u95ee\u7406\u89e3",
    body: "Keeps session context for budget updates and follow-up comparison questions.",
  },
  {
    title: "\u9884\u7b97\u4e0e\u504f\u597d\u6539\u5199",
    body: "Turns short follow-ups into explicit retrieval queries while preserving the original user message.",
  },
  {
    title: "RAG \u77e5\u8bc6\u89e3\u91ca",
    body: "Combines product data with Markdown knowledge documents and citation traces.",
  },
  {
    title: "\u5019\u9009\u5546\u54c1\u5185\u6bd4\u8f83",
    body: "Limits comparison to products from the previous recommendation turn.",
  },
  {
    title: "LangGraph Workflow",
    body: "Routes query understanding, retrieval, comparison, answer composition, and memory steps.",
  },
  {
    title: "SSE Debug",
    body: "Streams session, trace, result, done, and error events into a visual Agent Timeline.",
  },
];

export function ShowcasePage({
  onEnterDebug,
  onSelectExample,
}: ShowcasePageProps) {
  const firstPrompt = SCENARIOS[0].prompts[0];

  return (
    <div className="showcase-page">
      <section className="showcase-hero">
        <div className="workflow-scene" aria-hidden="true">
          <span className="scene-node scene-node-query">Query</span>
          <span className="scene-node scene-node-rag">RAG</span>
          <span className="scene-node scene-node-agent">AgentWorkflow</span>
          <span className="scene-node scene-node-sse">SSE Trace</span>
          <span className="scene-line scene-line-one" />
          <span className="scene-line scene-line-two" />
          <span className="scene-line scene-line-three" />
        </div>
        <div className="hero-content">
          <p className="hero-kicker">SmartBuyAgent Showcase</p>
          <h1>AI shopping guide for structured product data and RAG knowledge.</h1>
          <p>
            SmartBuyAgent is an intelligent shopping-guide Agent for new retail
            products. It combines product tables, a knowledge base, LangGraph
            orchestration, multi-turn follow-ups, candidate comparison, and a
            visual debugging workflow.
          </p>
          <div className="hero-actions">
            <button type="button" onClick={onEnterDebug}>
              Open Web Debug
            </button>
            <button
              className="secondary-button hero-secondary"
              type="button"
              onClick={() => onSelectExample(firstPrompt)}
            >
              Start with an example
            </button>
          </div>
        </div>
      </section>

      <section className="showcase-section">
        <div className="section-heading">
          <p className="section-kicker">MVP Scenarios</p>
          <h2>Three shopping domains ready for demo</h2>
        </div>
        <div className="scenario-grid">
          {SCENARIOS.map((scenario) => (
            <article className="scenario-card" key={scenario.title}>
              <p className="scenario-subtitle">{scenario.subtitle}</p>
              <h3>{scenario.title}</h3>
              <p>{scenario.description}</p>
              <div className="prompt-list">
                {scenario.prompts.map((prompt) => (
                  <button
                    className="prompt-button"
                    type="button"
                    onClick={() => onSelectExample(prompt)}
                    key={prompt}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="showcase-section">
        <div className="section-heading">
          <p className="section-kicker">Core Capabilities</p>
          <h2>From user query to explainable answer</h2>
        </div>
        <div className="capability-grid">
          {CAPABILITIES.map((capability) => (
            <article className="capability-card" key={capability.title}>
              <h3>{capability.title}</h3>
              <p>{capability.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="debug-entry">
        <div>
          <p className="section-kicker">Debug Workspace</p>
          <h2>Inspect every Agent step before building the showcase flow.</h2>
          <p>
            Use Web Debug to compare normal HTTP responses with SSE trace
            streaming, inspect product cards and citations, and verify the Agent
            Timeline for multi-turn questions.
          </p>
        </div>
        <button type="button" onClick={onEnterDebug}>
          Enter Web Debug
        </button>
      </section>
    </div>
  );
}
