# Evaluation

## 1. Backend Tests

Run:

```bash
cd backend
pytest
```

Current backend tests cover:

- Health API
- Chat API
- Chat SSE API
- AgentState and agent nodes
- LangGraph AgentWorkflow
- Query understanding
- Follow-up query rewrite
- Product retrieval
- Knowledge retrieval
- Chroma indexer
- Product comparison
- Response composer
- LLM provider and LLM answer guardrails
- Conversation memory
- Feedback API
- Import scripts for categories, products, and documents
- Real product normalizer, dataset validator, and JSONL import upsert behavior
- Redis-compatible cache service using in-memory fakes
- Session recent-turn cache behavior
- Product and knowledge retrieval cache hit/miss behavior
- Per-session rate limit behavior
- Feedback short-term counter aggregation
- Streaming SSE event formatting
- Realtime node_start/node_end events with duration_ms
- Granular product_retrieval and knowledge_retrieval stream nodes
- Retrieval stream events for product and knowledge recall
- Provider-native LLM token streaming with chunked fallback behavior
- Error-node exposure in `/api/chat/stream`

## 2. Frontend Build

Run:

```bash
cd frontend
npm run build
```

This validates:

- TypeScript types
- React component imports
- Vite production build
- Chat API client
- SSE client
- Feedback API client
- Chat Workspace composition

## 3. Retrieval and Multiturn Evaluation

Retrieval cases live in:

```text
data/eval/retrieval_eval_cases.json
```

Multiturn cases live in:

```text
data/eval/multiturn_eval_cases.json
```

Run:

```bash
cd backend
python ../scripts/eval_retrieval.py
python ../scripts/eval_multiturn.py
```

These checks are lightweight workflow and rule assertions. They are not final business metrics.

## 4. Data Import Evaluation

Stage Data-1 adds tests for:

- phone CSV normalization, brand inference, tag derivation, and data-quality warnings
- JSONL normalization for schema-like input
- processed JSONL validation and min-count failures
- skincare medical-claim banned term detection
- real-product import dry-run behavior
- real-product upsert behavior without duplicate products

The tests use tiny local fixtures and SQLite. They do not require large external datasets, MySQL, Redis, crawlers, or API access.

## 5. Manual Demo Cases

Recommended manual checks:

1. Single-turn phone guide: `预算3000，推荐一款拍照好的手机`
2. Budget follow-up: `预算提高到4000呢`
3. Ordinal comparison: `第一个和第二个有什么区别`
4. Product knowledge: `为什么手机拍照不能只看像素`
5. Skincare boundary: ask for sensitive-skin guidance and confirm there are no medical cure claims.
6. Feedback submission: submit helpful or not helpful feedback from the Web Debug page.

## 6. Current Limitations

- Seed data is intentionally small.
- The default local setup uses mock embedding and mock LLM providers.
- Backend tests use SQLite and in-memory/fake cache services; they do not require real MySQL or Redis.
- MySQL 5.7 compatibility is maintained through SQLAlchemy model choices and documented configuration rather than live MySQL integration tests.
- The project is not connected to live ecommerce inventory.
- SSE streams realtime workflow node events, separate retrieval nodes, retrieval summaries, provider-native token chunks, trace, final result, and done/error events.
- Feedback is stored but does not yet train ranking or retrieval.
- The system does not perform checkout, payment, orders, or fulfillment.
