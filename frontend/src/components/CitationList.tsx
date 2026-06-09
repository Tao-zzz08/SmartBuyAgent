import type { Citation } from "../api/chat";

type CitationListProps = {
  citations: Citation[];
};

export function CitationList({ citations }: CitationListProps) {
  return (
    <section className="panel">
      <h2>Citations</h2>
      {citations.length > 0 ? (
        <div className="citation-list">
          {citations.map((citation) => (
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
  );
}
