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
- AnswerGroundingGuard checks for unsupported prices, purchase-boundary language, unsupported stock/discount/ranking claims, unsupported brand mentions, missing citation support, and skincare medical claims

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

SmartBuyAgent now keeps five eval files with separate responsibilities:

```text
data/eval/query_understanding_regression_cases.json
data/eval/retrieval_eval_cases.json
data/eval/rag_eval_cases.json
data/eval/multiturn_eval_cases.json
data/eval/grounding_guard_eval_cases.json
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

### QueryUnderstanding Intent Eval

`eval_query_understanding_intent.py` provides field-level metrics for the
QueryUnderstanding layer.

It evaluates:

- intent accuracy
- category accuracy
- budget min/max extraction accuracy
- preference precision / recall / F1
- negative preference precision / recall / F1
- follow-up accuracy
- clarification accuracy
- compare reference resolution accuracy
- intent-in boundary checks
- forbidden preference violation rate

The suite also reports diagnostic-only metrics for LLM fallback trigger
observation and multi-intent capability gaps. Diagnostic checks do not affect
case pass/fail.

This script is independent from `run_eval_all.py` and is intended for
QueryUnderstanding baseline analysis before changing fallback logic, dialog
state, or multi-intent support.

P0-2 adds a fallback trigger decision layer. The eval runner still disables
real LLM calls by default, but it records a theoretical
`llm_fallback_should_call` diagnostic based on the current fallback decision
policy. This allows long-tail first-turn queries and multi-intent queries to be
measured without calling external LLM services.

The fallback trigger policy covers:

- ambiguous follow-ups
- product references
- first-turn long-tail shopping queries
- multi-intent queries
- unknown-category purchase needs
- low-confidence product-help queries

Safety-boundary queries such as purchase/payment requests or skincare medical
claims are excluded from LLM fallback trigger expansion.

P0-3 adds lightweight dialog-state awareness. QueryUnderstanding now records
the state used to interpret the current turn and the inferred next state, such
as `awaiting_budget`, `awaiting_category`, `showing_products`,
`comparing_products`, and `answering_knowledge`.

The intent eval suite reports dialog-state metrics:

- dialog_state_accuracy
- next_dialog_state_accuracy
- dialog_state_in_accuracy

These metrics help verify that short follow-ups such as "3000以内", "手机",
"第一个和第二个哪个好", and "就第一个吧" are interpreted differently depending
on the previous conversation stage.

Dialog state is not a purchase workflow. The project still does not support
cart, order, payment, or checkout actions.

Run:

```bash
python scripts/eval_query_understanding_intent.py \
  --cases data/eval/query_understanding_intent_eval_cases.json \
  --output results/query-understanding-intent-report.md \
  --details results/query-understanding-intent-details.json
```

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

Retrieval Eval also reports ranking and compliance metrics for annotated
product cases:

- Recall@K
- nDCG@K
- MRR@K
- Filter Compliance Rate
- Negative Preference Violation Rate
- Empty Rate
- Latency P50/P95

These metrics are computed from manually annotated `gold_relevance` and
`hard_filters` in selected retrieval eval cases. Cases without gold relevance
still run the rule-based assertions, but they are skipped for ranking averages.

### Retrieval A/B Strategy Benchmark

`retrieval_ab_benchmark.py` compares deterministic retrieval strategies offline:

- `structured_filter_only`
- `lexical_keyword`
- `hybrid_filter_keyword`
- `hybrid_plus_rerank`

The benchmark reports Recall@K, nDCG@K, MRR@K, filter compliance, empty rate,
latency, best strategy by metric, deltas versus baseline, strategy win counts,
and category-level breakdown.

This benchmark is intended for offline analysis and is not part of the default
regression suite unless explicitly enabled. It does not call external LLM or
embedding services.

Run:

```bash
python scripts/retrieval_ab_benchmark.py \
  --cases data/eval/retrieval_eval_cases.json \
  --products data/processed/products/all_products_900.jsonl \
  --strategies structured_filter_only,lexical_keyword,hybrid_filter_keyword,hybrid_plus_rerank \
  --top-k 5
```

### Feedback-to-Eval Pipeline

`feedback_to_eval.py` converts structured user feedback and trace snapshots into
reviewable eval candidates.

It supports deterministic classification for:

- negative preference violations
- budget violations
- category mismatches
- missing or unsupported citations
- purchase boundary violations
- skincare medical claims
- prompt injection failures
- compare resolution failures
- missing clarification

The pipeline does not automatically modify formal eval files. It writes
reviewable candidates with `needs_review: true` and `review_status: pending`,
so humans can decide which cases should be promoted into `data/eval/*.json`.

Run:

```bash
python scripts/feedback_to_eval.py \
  --input data/feedback/sample_feedback.jsonl \
  --output results/eval-candidates.json \
  --markdown results/eval-candidates.md
```

`results/eval-candidates.json` and `results/eval-candidates.md` are generated
review artifacts and should not be committed.

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

RAG Eval also includes deterministic claim support checks:

- Claim Support Rate
- Citation Coverage Rate
- Unsupported Claim Rate
- Grounded Answer Rate
- Missing Required Claims
- Hallucination Violation Count

Each claim is declared in the eval case with answer trigger terms and citation
support terms. The eval does not use LLM-as-judge; it deterministically checks
whether answer claims are supported by retrieved citations.

### Multiturn Eval

`multiturn_eval_cases.json` validates complete multi-turn shopping flows:

- budget ladders across several turns
- preference refinement and negative preferences
- category switch chains across phone, shoes, and skincare
- recommendation followed by comparison
- comparison followed by a return to shopping-guide retrieval
- stream and non-stream consistency for structured query understanding

Multiturn Eval also reports session-level task success metrics in addition to
turn-level pass/fail:

- Session Success Rate
- Context Carryover Accuracy
- Category Switch Accuracy
- Compare Resolution Accuracy
- Clarification Accuracy
- Route Stability Rate

These metrics evaluate whether a complete shopping conversation remains stable
across budget follow-ups, category switches, negative preference updates, and
compare follow-ups.

### GroundingGuard Eval

`grounding_guard_eval_cases.json` validates the final-answer guard layer:

- purchase-boundary and discount language is blocked before final output
- explicit price claims must match product cards, query budget, or comparison evidence
- unsupported inventory, sales, coupon, and ranking claims fall back to deterministic answers
- brand mentions must be grounded in current product cards and negative preferences
- product-knowledge answers require citation support
- skincare answers cannot make treatment, cure, drug-effect, prescription, or disease-repair claims

### Red Team Safety Eval

`red_team_eval_cases.json` validates safety boundary behavior under
adversarial or policy-violating user requests.

It covers:

- purchase / checkout / payment boundary
- fabricated inventory and shipping guarantees
- fabricated discounts, coupons, and lowest-price claims
- skincare medical claims
- prompt injection and system-prompt leakage
- citation and product-card fabrication

The suite reports:

- Red Team Pass Rate
- Safe Response Rate
- Violation Rate
- Total Violations
- Risk-type pass rates

Run:

```bash
cd backend
python ../scripts/run_query_understanding_eval.py --suite query_understanding
python ../scripts/run_query_understanding_eval.py --suite multiturn
python ../scripts/run_query_understanding_eval.py --suite rag
python ../scripts/run_query_understanding_eval.py --suite retrieval
python ../scripts/run_query_understanding_eval.py --suite grounding_guard
python ../scripts/run_query_understanding_eval.py --suite red_team
python ../scripts/run_query_understanding_eval.py --suite all
python ../scripts/eval_retrieval.py
```

### Unified Eval Report

Run all deterministic eval suites and write a Markdown summary plus JSON details:

```bash
python scripts/run_eval_all.py \
  --output results/eval-report.md \
  --details results/eval-details.json
```

From `backend/`, use:

```bash
python ../scripts/run_eval_all.py \
  --output ../results/eval-report.md \
  --details ../results/eval-details.json
```

The unified report aggregates QueryUnderstanding, Retrieval, RAG, Multiturn,
and GroundingGuard suites. It includes suite summaries, pass rates, failure
reason counts, failed case details, suite metrics such as retrieval ranking
metrics, RAG claim support metrics, multiturn session success metrics, and
Red Team safety metrics, and stable machine-readable JSON. The
report layer is deterministic and does not require external LLM APIs or
external network access.

Pytest coverage includes schema validation and fake-client runner checks:

```bash
cd backend
pytest tests/test_eval_cases_schema.py
pytest tests/test_query_understanding_regression_eval.py
pytest tests/test_retrieval_eval_cases.py
pytest tests/test_rag_eval_cases.py
pytest tests/test_multiturn_eval_cases.py
pytest tests/test_answer_grounding_guard.py
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
