import { useState } from "react";

import { DebugWorkbench } from "./components/DebugWorkbench";
import { ShowcasePage } from "./components/ShowcasePage";
import "./App.css";

const DEFAULT_QUERY =
  "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a";

type PageMode = "showcase" | "debug";

function App() {
  const [pageMode, setPageMode] = useState<PageMode>("showcase");
  const [query, setQuery] = useState(DEFAULT_QUERY);

  const handleSelectExample = (exampleQuery: string) => {
    setQuery(exampleQuery);
    setPageMode("debug");
  };

  return (
    <main className="page-shell">
      <nav className="app-nav" aria-label="Primary navigation">
        <button
          className="brand-button"
          type="button"
          onClick={() => setPageMode("showcase")}
        >
          SmartBuyAgent
        </button>
        <div className="nav-actions">
          <button
            className={pageMode === "showcase" ? "nav-link active" : "nav-link"}
            type="button"
            onClick={() => setPageMode("showcase")}
          >
            Showcase
          </button>
          <button
            className={pageMode === "debug" ? "nav-link active" : "nav-link"}
            type="button"
            onClick={() => setPageMode("debug")}
          >
            Web Debug
          </button>
        </div>
      </nav>

      {pageMode === "showcase" ? (
        <ShowcasePage
          onEnterDebug={() => setPageMode("debug")}
          onSelectExample={handleSelectExample}
        />
      ) : null}

      <div hidden={pageMode !== "debug"}>
        <DebugWorkbench query={query} onQueryChange={setQuery} />
      </div>
    </main>
  );
}

export default App;
