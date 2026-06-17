from app.agent.context import AgentRuntimeContext
from app.agent.stream_runner import AgentStreamRunner
from app.chat.query_understanding import QueryUnderstandingResult
from app.retrieval.retrieval_service import Citation, ProductCandidate


class FakeQueryUnderstandingService:
    def understand(self, query: str) -> QueryUnderstandingResult:
        return QueryUnderstandingResult(
            raw_query=query,
            intent="shopping_guide",
            category_id="cat_phone",
            category_path="digital/phone",
            budget_min=None,
            budget_max=3000,
            preferences=["camera"],
            need_clarification=False,
            clarification_question=None,
        )


class FakeProductRetrievalService:
    last_cache_status = "miss"

    def search_products(self, query, filters=None, top_k=3):
        return [_product()]


class FakeKnowledgeRetrievalService:
    last_cache_status = "hit"

    def search_knowledge(self, query, category_id=None, top_k=3):
        return [_citation()]


class FakeStreamingLLMAnswerComposer:
    def stream_compose(
        self,
        query,
        query_result,
        product_candidates=None,
        citations=None,
        on_token=None,
    ) -> str:
        chunks = ["Native ", "streamed ", "answer for phone_001."]
        for chunk in chunks:
            if on_token is not None:
                on_token(chunk)
        return "".join(chunks)


def test_agent_stream_runner_uses_native_llm_token_chunks() -> None:
    runner = AgentStreamRunner(
        AgentRuntimeContext(
            query_understanding_service=FakeQueryUnderstandingService(),
            product_retrieval_service=FakeProductRetrievalService(),
            knowledge_retrieval_service=FakeKnowledgeRetrievalService(),
            llm_answer_composer=FakeStreamingLLMAnswerComposer(),
        )
    )

    events, state = _collect_stream(
        runner,
        query="recommend a camera phone",
        request_id="req_native_tokens",
    )
    tokens = [
        event.data["delta"]
        for event in events
        if event.event == "token"
    ]

    assert tokens == ["Native ", "streamed ", "answer for phone_001."]
    assert state.answer == "".join(tokens)
    assert state.product_cards
    assert state.citations


def _collect_stream(
    runner: AgentStreamRunner,
    *,
    query: str,
    request_id: str,
) -> tuple[list, object]:
    generator = runner.stream(query, request_id=request_id)
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            return events, stop.value


def _product() -> ProductCandidate:
    return ProductCandidate(
        product_id="phone_001",
        title="Camera Phone",
        brand="Brand",
        category_id="cat_phone",
        price=2599,
        stock=10,
        description="Camera-focused phone.",
        image_url=None,
        tags=["camera"],
        attributes={"camera": "50MP"},
        source_url=None,
        compare_url=None,
        distance=0.1,
        score=0.9,
        product_text="Camera Phone product text",
    )


def _citation() -> Citation:
    return Citation(
        chunk_id="chunk_001",
        document_id="doc_001",
        title="Phone camera guide",
        section="Camera",
        section_path="Guide/Camera",
        source_file="phone.md",
        doc_type="guide",
        category_id="cat_phone",
        category_path="digital/phone",
        content_preview="Camera specs need context.",
        distance=0.2,
        score=0.8,
    )

