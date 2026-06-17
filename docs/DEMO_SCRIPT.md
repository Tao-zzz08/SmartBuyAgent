# Demo Script

This is a 3-5 minute demo flow for SmartBuyAgent.

## 1. Opening

Introduce SmartBuyAgent as a RAG + Agent prototype for new retail shopping guidance.

Key points:

- Structured product data
- Knowledge-document RAG
- LangGraph AgentWorkflow
- Multiturn follow-up handling
- Chat-style workspace
- Per-answer expandable Agent Timeline
- Feedback collection

## 2. Chat Workspace

Open the frontend and show the Chat Workspace.

Explain:

- The left sidebar stores in-memory local sessions.
- The center area is the active conversation.
- The bottom input bar supports normal send and SSE stream send.
- The empty conversation welcome panel shows example prompts.
- Example prompts fill the input only; they do not auto-send.
- The project is a guide and explanation system, not a transaction system.

## 3. Single-turn Shopping Guide

Prompt:

```text
预算3000，推荐一款拍照好的手机
```

Show the assistant reply:

- Answer
- Product cards
- Tags and attributes
- Citations
- Recommendation reasons
- Feedback Panel

Explain that product cards come from retrieval, not from free-form LLM generation.

## 4. Expand Debug for One Answer

Click `查看 Debug` under the assistant reply.

Show:

- Agent Timeline
- Raw Trace JSON
- Raw Response JSON

Explain that debug data belongs to this specific assistant reply, not to a global response panel.

## 5. Budget Follow-up

Prompt:

```text
预算提高到4000呢
```

Open this answer's Debug panel and show:

- `load_context`
- `follow_up_rewrite`
- `query_understanding`
- `product_retrieval`
- `knowledge_retrieval`
- `response_compose`

Explain that `follow_up_rewrite` creates an effective query while memory still stores the original user query.

## 6. Candidate Comparison

Prompt:

```text
第一个和第二个有什么区别
```

Open this answer's Debug panel and show:

- `resolved_product_ids`
- `product_comparison`
- Product cards limited to the referenced previous candidates

Explain that the comparison branch does not search the full product database.

## 7. Product Knowledge

Prompt:

```text
为什么手机拍照不能只看像素
```

Show:

- Knowledge citations
- Source file preview
- Agent Timeline knowledge retrieval step

Explain that citations come from imported Markdown knowledge chunks.

## 8. SSE Stream Mode

Use `Stream`.

Show:

- Assistant placeholder while the workflow is running
- Trace events appended to the message's debug state
- Final result replacing the assistant placeholder

Explain that the current stream is workflow trace streaming, not token-level LLM streaming.

## 9. Session Switching

Create a second local session from the sidebar.

Show:

- A new empty welcome panel
- A separate backend `session_id` after the first request
- Returning to the previous session keeps its messages and debug panels

## 10. Feedback Loop

Submit helpful or not helpful feedback on one assistant reply.

Explain:

- Feedback is saved through `/api/feedback`.
- Feedback is tied to the current session and answer preview.
- It is used for future evaluation.
- It does not change current recommendations.

## 11. Closing

Summarize the value:

- RAG knowledge grounding
- Structured product retrieval
- Multiturn context handling
- Candidate-only comparison
- LangGraph workflow orchestration
- Per-answer observable debug timeline
- Feedback loop for future quality analysis
