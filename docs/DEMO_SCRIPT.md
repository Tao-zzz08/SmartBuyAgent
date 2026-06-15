# Demo Script

This is a 3-5 minute demo flow for SmartBuyAgent.

## 1. Opening

Introduce SmartBuyAgent as a RAG + Agent prototype for new retail shopping guidance.

Key points:

- Structured product data
- Knowledge-document RAG
- LangGraph AgentWorkflow
- Multiturn follow-up handling
- Web Debug trace visualization
- Feedback collection

## 2. Showcase Page

Open the frontend and show the Showcase page.

Explain:

- The MVP supports phones, shoes, and skincare.
- Example prompts can be moved into Web Debug.
- The project is a guide and explanation system, not a transaction system.

## 3. Single-turn Shopping Guide

Prompt:

```text
预算3000，推荐一款拍照好的手机
```

Show:

- Answer
- Product cards
- Tags and attributes
- Citations
- Recommendation reasons

Explain that product cards come from retrieval, not from free-form LLM generation.

## 4. Budget Follow-up

Prompt:

```text
预算提高到4000呢
```

Show Agent Timeline:

- `load_context`
- `follow_up_rewrite`
- `query_understanding`
- `product_retrieval`
- `knowledge_retrieval`
- `response_compose`

Explain that `follow_up_rewrite` creates an effective query while memory still stores the original user query.

## 5. Candidate Comparison

Prompt:

```text
第一个和第二个有什么区别
```

Show:

- `resolved_product_ids`
- `product_comparison`
- Product cards limited to the referenced previous candidates

Explain that the comparison branch does not search the full product database.

## 6. Product Knowledge

Prompt:

```text
为什么手机拍照不能只看像素
```

Show:

- Knowledge citations
- Source file preview
- Agent Timeline knowledge retrieval step

Explain that citations come from imported Markdown knowledge chunks.

## 7. SSE Debug

Use `Stream send`.

Show:

- Session event
- Trace events
- Result event
- Done event

Explain that the current stream is workflow trace streaming, not token-level LLM streaming.

## 8. Feedback Loop

Submit helpful or not helpful feedback.

Explain:

- Feedback is saved through `/api/feedback`.
- It is used for future evaluation.
- It does not change current recommendations.

## 9. Closing

Summarize the value:

- RAG knowledge grounding
- Structured product retrieval
- Multiturn context handling
- Candidate-only comparison
- LangGraph workflow orchestration
- Observable debug timeline
- Feedback loop for future quality analysis
