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
- QueryUnderstanding 2.0 structured result, shopping-memory merge, and validated LLM JSON-slot fallback
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
- StreamSafetyGuard phrase matching with rolling-buffer cross-token detection
- Guarded `/api/chat/stream` behavior: `stream_guard`, response node failure, safe fallback result, and no unsafe token exposure
- Query-understanding stream trace consistency, including `llm_fallback_attempted`, `llm_fallback_status`, `source`, `confidence`, and `reason`

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

## 3. Evaluation Suites

SmartBuyAgent now keeps four eval files with separate responsibilities:

```text
data/eval/query_understanding_regression_cases.json
data/eval/retrieval_eval_cases.json
data/eval/rag_eval_cases.json
data/eval/multiturn_eval_cases.json
```

### QueryUnderstanding Eval

`query_understanding_regression_cases.json` validates structured query
understanding and routing:

- intent, route, category, budget, preferences, and negative preferences
- structured memory merge for budget updates and short numeric follow-ups
- category switching and incompatible preference filtering
- validated LLM slot fallback for ambiguous follow-ups
- product comparison references resolved from previous real product cards
- the regression where budget follow-ups must not route to product comparison
- no previous products means ordinal comparison must clarify instead of
  fabricating product references

### Retrieval Eval

`retrieval_eval_cases.json` is retrieval-focused, not final answer-focused. It
uses structured filters and soft assertions rather than depending primarily on
fixed product IDs:

- product category compliance
- budget compliance
- positive preference keyword matching
- negative preference / forbidden term violations
- minimum product result count
- knowledge chunk keyword hit rate
- forbidden term violations in retrieved chunks

### RAG Eval

`rag_eval_cases.json` validates grounded final answers:

- minimum citation count
- citation fields such as `chunk_id`, source, and text/content preview
- citation keywords that support the answer
- answer terms that should be present
- forbidden purchase, payment, and shopping-cart language
- skincare medical-claim safety boundaries
- unknown knowledge questions should clarify or answer safely instead of
  fabricating clinical conclusions

### Multiturn Eval

`multiturn_eval_cases.json` validates complete multi-turn shopping flows:

- budget ladders across several turns
- preference refinement and negative preferences
- category switch chains across phone, shoes, and skincare
- recommendation followed by comparison
- comparison followed by a return to shopping-guide retrieval
- stream and non-stream consistency for structured query understanding

Run:

```bash
cd backend
python ../scripts/run_query_understanding_eval.py --suite query_understanding
python ../scripts/run_query_understanding_eval.py --suite multiturn
python ../scripts/run_query_understanding_eval.py --suite rag
python ../scripts/run_query_understanding_eval.py --suite retrieval
python ../scripts/run_query_understanding_eval.py --suite all
python ../scripts/eval_retrieval.py
```

Pytest coverage includes schema validation and fake-client runner checks:

```bash
cd backend
pytest tests/test_eval_cases_schema.py
pytest tests/test_query_understanding_regression_eval.py
pytest tests/test_retrieval_eval_cases.py
pytest tests/test_rag_eval_cases.py
pytest tests/test_multiturn_eval_cases.py
```

These checks are deterministic regression and boundary assertions. They do not
use external LLM APIs, external network calls, or production databases.

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
- Streaming token output is protected by a lightweight safety guard; guarded streams return a safe fallback final result.
- Feedback is stored but does not yet train ranking or retrieval.
- The system does not perform checkout, payment, orders, or fulfillment.
