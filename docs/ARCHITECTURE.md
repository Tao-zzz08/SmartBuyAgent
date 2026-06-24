# Architecture

## 1. System Overview

SmartBuyAgent is a RAG + Agent shopping-guide prototype for new retail product scenarios. It supports phones, shoes, and skincare as MVP categories.

The system has five main layers:

- Data layer: MySQL-compatible SQLAlchemy tables, SQLite fallback for lightweight local/test usage, Markdown knowledge documents, and Chroma indexes.
- Cache/state layer: Redis for short-lived session cache, retrieval cache, SSE trace state, per-session rate limiting, and feedback counters.
- Service layer: query understanding, retrieval, response composition, LLM answer composition, memory, and feedback.
- Agent layer: LangGraph AgentWorkflow and executable agent nodes.
- Frontend layer: Chat Workspace, in-memory session sidebar, per-answer Agent Timeline, SSE streaming, and feedback UI.

## 2. Backend Layers

FastAPI exposes:

- `GET /health`
- `POST /api/chat`
- `POST /api/chat/stream`
- `POST /api/feedback`

Core backend modules:

- `app/core`: configuration, database, logging.
- `app/models`: SQLAlchemy tables for products, documents, sessions, turns, and feedback.
- `app/services`: embedding and LLM provider abstractions.
- `app/retrieval`: Chroma indexing and product/knowledge retrieval.
- `app/chat`: query understanding, response composition, memory, follow-up rewrite, product comparison, and ChatService facade.
- `app/agent`: AgentState, runtime context, nodes, and LangGraph workflow.

`ChatService` is the stable API-facing facade. Runtime chat execution is routed through `AgentWorkflow`.

## 2.1 QueryUnderstanding 2.0

Query understanding is rule-first and returns a unified `QueryUnderstandingResult` stored on `AgentState`.

- Step 1 adds structured shopping memory, memory merge, abbreviated budget follow-up handling, category switching, preference updates, and stable `effective_query` generation.
- Step 2 makes the structured result shared by chat, streaming, retrieval, comparison, and answer composition.
- Step 3 adds an optional LLM fallback only for low-confidence or ambiguous follow-up queries. The LLM is instructed to output JSON slots only; it cannot answer the user, create product cards, create citations, create purchase links, or decide recommendation candidates.

LLM fallback output is parsed as JSON, validated with Pydantic, sanitized through category/intent/budget/product-id whitelists, stripped of purchase-boundary terms, and filtered for skincare medical claims before it is merged back into the rule result and session shopping memory. If parsing, validation, or the LLM call fails, the system falls back to the rule result or clarification instead of returning HTTP 500.

## 2.2 Database and Cache Infrastructure

MySQL is the primary relational database target for persistent data in deployed or full local environments. The project reads `DATABASE_URL`, for example:

```env
DATABASE_URL=mysql+pymysql://smartbuy:smartbuy@127.0.0.1:3306/smartbuy_agent?charset=utf8mb4
```

SQLite remains the default fallback when `DATABASE_URL` is not configured:

```env
DATABASE_URL=sqlite:///./data/smartbuy.db
```

The SQLAlchemy models use ordinary `VARCHAR`, `TEXT`, `INTEGER`, `DATETIME`, and boolean-like fields and store JSON payloads as text where needed. The code does not depend on MySQL 8-only features such as window functions, CTEs, or CHECK constraint enforcement, so a local Windows MySQL Server 5.7 installation is supported.

Redis is an optional cache and short-term state layer. Redis responsibilities:

- recent session turns cache
- last candidate product IDs
- product retrieval result cache
- knowledge retrieval result cache
- SSE trace/debug state
- per-session rate limiting
- short-term feedback counters

Redis is not the source of truth. If Redis is unavailable or a cache miss happens, the system falls back to MySQL/SQLite, Chroma, and the existing service logic. The final LLM answer is not cached.

## 3. Frontend Layers

The frontend is built with React, TypeScript, and Vite.

Main frontend surfaces:

- Chat Workspace: ChatGPT-style primary interface with an in-memory session sidebar, active conversation stream, and sticky input bar.
- Welcome panel: empty-session showcase prompts that fill the input without auto-sending.
- Assistant message results: answer, product cards, citations, and feedback rendered per assistant reply.
- Per-answer Debug panel: expandable Agent Timeline, Raw Trace JSON, and Raw Response JSON for the selected reply.
- SSE stream mode: appends realtime node, retrieval, token, and trace events to the active assistant placeholder before the final result arrives.

The frontend does not fabricate product cards or citations. It renders the backend response.

## 4. Data Model

Product and category data:

- `categories`
- `category_attribute_defs`
- `category_profiles`
- `products`
- `product_attributes`
- `product_tags`

Knowledge data:

- `documents`
- `document_chunks`

Conversation and feedback:

- `chat_sessions`
- `chat_turns`
- `chat_feedback`

`chat_turns` stores compact summaries: original user query, answer, intent, category, budget, preferences, product IDs, and citation chunk IDs. It does not store vectors or full product/citation payloads.

## 5. AgentWorkflow

The AgentWorkflow orchestrates:

1. Load recent conversation context.
2. Rewrite follow-up queries when a session context exists.
3. Understand intent, category, budget, and preferences.
4. Route to shopping guide, product knowledge, comparison, clarification, or chitchat.
5. Retrieve products and knowledge citations when needed.
6. Compare only the referenced in-session candidate products when comparison context exists.
7. Compose the response.

Product cards are produced by product retrieval or product comparison. Citations are produced by knowledge retrieval. The LLM only controls the `answer` wording and is guarded by output validation.

## 6. RAG Pipeline

Seed data is imported into SQLite:

- Categories and category profiles
- Mini product CSV data
- Markdown knowledge documents

Real product datasets can also be imported through the Data-1 pipeline:

```text
data/raw/products/*.csv|*.json|*.jsonl
-> normalize_real_products.py
-> data/processed/products/*.jsonl
-> validate_product_dataset.py
-> import_real_products.py
-> relational database
-> rebuild_index.py
```

The processed product schema preserves source IDs, category, title, brand, price, currency, description, image URL, source URL, tags, attributes, and data quality warnings.

Document import splits Markdown files into `document_chunks`. Index rebuild writes product text and knowledge chunks into Chroma collections:

- `product_text`
- `knowledge_docs`

During chat, product retrieval combines structured filters and vector recall, while knowledge retrieval searches Chroma and returns citation views.

## 7. Conversation Memory

The API layer owns memory persistence:

- If a request has no `session_id`, `/api/chat` and `/api/chat/stream` generate one.
- If a request provides `session_id`, it is reused.
- The original `user_query` is saved.
- The rewritten query is only exposed in trace and is not saved as the user query.

Memory currently supports follow-up rewrite and in-session comparison. It is not a long-term personalization engine.

## 8. Feedback Loop

`POST /api/feedback` stores answer feedback:

- session ID
- optional turn ID
- rating
- reason
- optional comment
- original query
- answer preview

Feedback does not affect current retrieval, ranking, or recommendation behavior. It is collected for later evaluation and quality analysis.

## 8.1 Streaming Architecture

`POST /api/chat` uses the normal ChatService facade and LangGraph AgentWorkflow path.

`POST /api/chat/stream` uses a realtime AgentStreamRunner that reuses the executable agent nodes and emits SSE events as nodes run:

- `node_start` before a node executes
- separate `product_retrieval` and `knowledge_retrieval` nodes with independent timing
- `retrieval` when product recall, knowledge retrieval, or product comparison completes
- `token` chunks from provider-native LLM streaming when available, with a chunked non-streaming fallback
- `trace` for backwards-compatible debug trace steps
- `node_end` with `duration_ms`
- `error` when a node fails
- `result` as the final source of truth for answer, product cards, citations, trace, and session ID

Product cards remain sourced from ProductRetrievalService or ProductComparisonService. Citations remain sourced from KnowledgeRetrievalService. LLM output only affects the answer text and does not decide candidates or citations.

The stream distinguishes draft answer text from the official final answer:

- `answer_draft_delta` is LLM wording for debug/timeline display only.
- `grounding_guard_result` reports the final-answer grounding check.
- `final_answer` is the official answer text after AnswerGroundingGuard validation.
- `result.answer` remains the final source of truth for persisted chat output.

## 8.2 Streaming Safety Guard

`LLMAnswerComposer.stream_compose()` uses a lightweight `StreamSafetyGuard` before token chunks are emitted to the frontend.

- A rolling buffer keeps the latest 120 characters so phrases split across token boundaries can still be detected.
- The streamer holds back the latest 80 characters and releases only older safe text, reducing the chance that the first half of a risky phrase is already visible.
- The guard blocks phrase-level purchase actions, fabricated purchase/payment links, fabricated ecommerce-source claims, and skincare medical claims.
- Normal advice such as "worth buying" or "compare before buying" is allowed because the guard does not block the standalone word "purchase/buy".
- When a violation is detected, `/api/chat/stream` emits `stream_guard`, `error`, and `node_end(status=failed)`, then returns a final `result` with a safe fallback answer and `done(status=guarded)`.

The guard is not a full factuality verifier. It is a fast stream-time boundary for the highest-risk output classes. The final non-streaming LLM answer validation still runs before an answer is accepted.

## 8.3 Answer Grounding Guard

`AnswerGroundingGuard` runs after response composition for both `/api/chat` and `/api/chat/stream`.

It validates the final answer against the retrieved product cards, knowledge citations, comparison result, route, and structured query understanding. The first implementation is rule-based and does not use LLM-as-judge.

The guard checks:

- purchase-boundary language such as checkout, payment, cart, purchase links, coupons, or limited-time promotions
- explicit price claims against product card prices, query budget, or comparison evidence
- unsupported inventory, sales, discount, ranking, and authenticity claims
- brand or product mentions that are not grounded in current product cards or conflict with negative preferences
- citation support for product-knowledge answers
- skincare medical-claim terms such as treatment, cure, drug effect, prescription, medical repair, or disease repair

If the answer passes, the LLM wording is returned. If it fails, the backend returns a deterministic fallback answer built from the available product cards, citations, route, and preferences. Product cards and citations are not changed by the guard and are never generated by the LLM.

## 9. Safety and Boundaries

SmartBuyAgent does not provide login, shopping cart, orders, payment, fulfillment, or after-sales tickets.

LLM output is constrained by guardrails. Unsafe purchase actions, unsupported discount claims, skincare medical claims, JSON/table output, unknown product IDs, and unknown URLs are rejected and fall back to template answers.

Skincare content is limited to daily care and ingredient guidance. It does not provide diagnosis, treatment, cure, or drug-effect claims.
