type AnswerPanelProps = {
  answer?: string;
};

export function AnswerPanel({ answer }: AnswerPanelProps) {
  return (
    <section className="panel">
      <h2>Answer</h2>
      <p className="answer-text">{answer || "暂无回答"}</p>
    </section>
  );
}
