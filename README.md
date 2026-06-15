# SmartBuyAgent

SmartBuyAgent 是一个面向多品类全新商品的电商智能导购 RAG Agent 系统。MVP 后续会支持商品导入、商品知识库 RAG、LangGraph Agent 编排、SSE 流式输出、商品卡片、Web Debug 和 Web Showcase。

当前已完成阶段 1 的数据库模型、基础分类导入和 mini 商品导入。阶段 2 已新增 17 份 Markdown 知识文档，支持将 Markdown 知识文档导入 `documents` / `document_chunks` 表，并支持使用 mock `EmbeddingService` 将商品文本和文档 chunk 写入 Chroma 的 `product_text` / `knowledge_docs` collection。已新增基础 `RetrievalService`，支持商品候选召回和知识文档 citation 召回。阶段 3 已新增规则版 `QueryUnderstandingService`、模板版 `ResponseComposer`、最小 `ChatService` 和基础 `/api/chat` 接口，可以通过 HTTP 调用 ChatService 完成一次最小导购回答闭环。阶段 4 已新增前端 Chat API 客户端和 Web Debug 单页布局，并已拆分为 ProductCardList、CitationList、TracePanel、RawJsonPanel 等展示组件，方便后续 SSE、LangGraph trace 和 Showcase 页面复用。当前还未接入真实 bge-m3 embedding。

## Start Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run Backend Tests

```bash
cd backend
pytest
```

## Initialize Database

```bash
cd backend
python ../scripts/init_db.py
```

## Import Category Seed Data

```bash
cd backend
python ../scripts/init_db.py
python ../scripts/import_categories.py
```

## Import Mini Product Seed Data

```bash
cd backend
python ../scripts/init_db.py
python ../scripts/import_categories.py
python ../scripts/import_products.py --dataset mini
```

## Import Knowledge Documents

```bash
cd backend
python ../scripts/import_docs.py
```

## Rebuild Chroma Index

```bash
cd backend
python ../scripts/init_db.py
python ../scripts/import_categories.py
python ../scripts/import_products.py --dataset mini
python ../scripts/import_docs.py
python ../scripts/rebuild_index.py
```

### Embedding Provider

The default index rebuild uses mock embedding:

```bash
cd backend
python ../scripts/rebuild_index.py
```

To use a real OpenAI-compatible embedding service, configure `.env`:

```env
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_DIM=your-model-dimension
EMBEDDING_API_BASE=https://your-embedding-api.example.com/v1
EMBEDDING_API_KEY=your-api-key
EMBEDDING_MODEL=your-embedding-model
```

After switching embedding provider or model, rebuild Chroma indexes. The indexing
stage and query stage must use the same embedding provider and dimension. Never
commit real API keys. The `/api/chat` query path also uses `get_embedding_service()`,
so it reads the same embedding provider configuration.

## LLM Provider

The default LLM provider is `mock`, which does not call external APIs. To use an
OpenAI-compatible Chat Completions service later, configure:

```env
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://your-llm-api.example.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-chat-model
```

Task 6.1 adds the provider abstraction and tests. Task 6.2 adds
`LLMAnswerComposer`, a standalone component that can generate a controlled answer
from retrieved product candidates and citations. Task 6.3 connects
`LLMAnswerComposer` to `ChatService` for `shopping_guide` and
`product_knowledge`; the LLM only generates the `answer` text, while product
cards and citations still come from retrieval services.
Task 6.4 adds lightweight LLM answer validation. If the LLM output contains
purchase actions, unsupported discounts, medical claims, JSON/table output,
unknown product IDs, or unknown URLs, the composer returns a safe fallback so
`ChatService` can fall back to the template answer.

## Conversation Memory

Task 7.1 adds session and turn persistence. If `session_id` is missing,
`/api/chat` generates one. Each turn stores the user query, assistant answer,
intent/category/budget/preferences, returned product IDs, and citation chunk IDs.
This stage only stores memory; it does not use memory for query rewriting,
retrieval, or recommendation decisions yet.
Task 7.2 adds a minimal frontend multi-turn experience. The Web Debug page stores
the returned `session_id` in page state, sends it with later `/api/chat` requests,
and provides a "新会话" action that clears the page messages, latest debug
response, and current `session_id`.
Task 7.3 adds rule-based follow-up detection and query rewriting. It only runs
when a request includes `session_id` and recent turns are available. Budget
changes, vague references such as "这几款", and ordinal references such as
"第一个/第二个" are rewritten into clearer queries and exposed in debug trace.
This stage does not use LLM rewrite and does not constrain retrieval to the
previous product IDs yet; that is reserved for Task 7.4.
Task 7.4 adds in-session product comparison. Follow-up queries like "这几款" and
"第一个/第二个" are resolved to the previous turn's product IDs, and the comparison
branch only loads those products from SQLite instead of running broad product
retrieval. The comparison logic is still lightweight and rule-based; Stage 8 will
move toward LangGraph orchestration.

## Agent Skeleton

Stage 8.1 adds `AgentState` and stub agent nodes as preparation for LangGraph.
The current `/api/chat` path still uses `ChatService` directly. No runtime
behavior is changed yet, and the skeleton does not call retrieval, LLM, or
database services.
Stage 8.2 implements executable Agent nodes by reusing the existing
`QueryUnderstandingService`, retrieval services, `ProductComparisonService`,
`ResponseComposer`, and optional `LLMAnswerComposer`. The runtime `/api/chat`
path still uses `ChatService` directly; LangGraph is not connected yet.
Stage 8.3 adds a LangGraph-based `AgentWorkflow` that orchestrates the executable
agent nodes. The workflow is independently testable, but the runtime `/api/chat`
path still uses `ChatService` directly. Switching `/api/chat` to AgentWorkflow is
reserved for Stage 8.4.
Stage 8.4 routes `ChatService` through `AgentWorkflow`. The runtime `/api/chat`
path now uses the LangGraph-based agent orchestration while preserving the
existing response schema and conversation memory behavior.
Stage 9.1 adds a backend SSE endpoint `/api/chat/stream`. It streams `session`,
`trace`, `result`, `done`, and `error` events using `text/event-stream`. The
current implementation runs AgentWorkflow first and then emits trace events in
order; frontend SSE integration is reserved for Stage 9.2.
Stage 9.2 adds frontend SSE consumption for `/api/chat/stream`. The Web Debug
page can now send streaming debug requests, append trace events as they arrive,
and render the final result event. The original non-streaming `/api/chat`
request path remains available.
Stage 9.3 enhances the Web Debug experience with an Agent Timeline view. Both
normal `/api/chat` responses and SSE `/api/chat/stream` events can now be
visualized as readable workflow steps, while the raw JSON debug view remains
available.

## Retrieval Evaluation

Eval cases are stored in `data/eval/retrieval_eval_cases.json`. They cover product
recall and knowledge citation recall for phone, shoes, and skincare scenarios.

Prepare data and indexes first:

```bash
cd backend
python ../scripts/import_categories.py
python ../scripts/import_products.py --dataset mini
python ../scripts/import_docs.py
python ../scripts/rebuild_index.py
```

Run retrieval eval:

```bash
cd backend
python ../scripts/eval_retrieval.py
```

Run multiturn eval:

```bash
cd backend
python ../scripts/eval_multiturn.py
```

The current metrics are lightweight rule checks, not final business metrics. Mock
embedding can verify the eval workflow, but it does not represent real retrieval
quality. After switching to real embedding, rebuild Chroma indexes before running
eval again. Product recall now applies lightweight preference reranking after
category, budget, and stock filters. Knowledge citation recall applies keyword
reranking after category/doc_type filtering. Eval reports include failure reasons
and failure reason counts to make misses easier to inspect.

## Chat API

`POST /api/chat`

Request:

```json
{
  "query": "预算3000，推荐一款拍照好的手机",
  "debug": true
}
```

Response contains:

- `answer`
- `product_cards`
- `citations`
- `trace`
- `session_id`

`POST /api/chat/stream`

Response: `text/event-stream`

Events:

- `session`: returns the generated or reused `session_id`
- `trace`: emits each trace step when `debug=true`
- `result`: returns the final `answer`, `product_cards`, `citations`, `trace`, and `session_id`
- `done`: marks stream completion
- `error`: reports stream-level failures

The frontend supports both normal POST `/api/chat` and SSE debug POST
`/api/chat/stream`.

## Knowledge Documents

Markdown seed documents are stored under `data/knowledge_docs/`:

- `phone/`: phone buying, camera, battery, performance, and FAQ guides.
- `shoes/`: commute, size, material, care, and FAQ guides.
- `skincare/`: sensitive skin, moisturizing, ingredients, usage, and FAQ guides.
- `common/`: after-sales policy and shopping guide tone documents.

## Start Frontend

```bash
cd frontend
npm install
npm run dev
```

## Not Implemented Yet

- 300 条 demo 商品数据
- 真实 embedding
- 生产级 LLM 回答质量优化
- LangGraph Agent
- Token-level LLM streaming
- 完整 Web Debug 高级功能
- Web Showcase

## Next Stage

Next Stage: Stage 10 Web Showcase + Feedback Loop.
