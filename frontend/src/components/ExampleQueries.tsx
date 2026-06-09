export type ExampleQuery = {
  label: string;
  query: string;
  description?: string;
};

type ExampleQueriesProps = {
  examples: ExampleQuery[];
  onSelect: (query: string) => void;
};

export function ExampleQueries({ examples, onSelect }: ExampleQueriesProps) {
  return (
    <div className="example-query-group">
      <p className="example-title">示例问题</p>
      <div className="example-list">
        {examples.map((example) => (
          <button
            className="example-button"
            key={example.label}
            onClick={() => onSelect(example.query)}
            title={example.description}
            type="button"
          >
            {example.label}
          </button>
        ))}
      </div>
    </div>
  );
}
