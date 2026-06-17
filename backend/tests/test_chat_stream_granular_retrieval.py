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
        self.query = query
        self.filters = filters
        self.top_k = top_k
        return [_product("phone_001"), _product("phone_002")]


class FakeKnowledgeRetrievalService:
    last_cache_status = "hit"

    def search_knowledge(self, query, category_id=None, top_k=3):
        self.query = query
        self.category_id = category_id
        self.top_k = top_k
        return [_citation("chunk_001"), _citation("chunk_002")]


class FakeStreamingLLMAnswerComposer:
    def stream_compose(
        self,
        query,
        query_result,
        product_candidates=None,
        citations=None,
        on_token=None,
    ) -> str:
        answer = "Use phone_001 when camera is the priority."
        if on_token is not None:
            on_token(answer)
        return answer


def test_agent_stream_runner_emits_granular_retrieval_nodes() -> None:
    product_service = FakeProductRetrievalService()
    knowledge_service = FakeKnowledgeRetrievalService()
    runner = AgentStreamRunner(
        AgentRuntimeContext(
            query_understanding_service=FakeQueryUnderstandingService(),
            product_retrieval_service=product_service,
            knowledge_retrieval_service=knowledge_service,
            llm_answer_composer=FakeStreamingLLMAnswerComposer(),
        )
    )

    events, _state = _collect_stream(
        runner,
        query="recommend a camera phone",
        request_id="req_granular_retrieval",
    )
    node_starts = [
        event.data.get("node")
        for event in events
        if event.event == "node_start"
    ]
    node_ends = [
        event.data
        for event in events
        if event.event == "node_end"
    ]
    retrieval_events = [
        event.data
        for event in events
        if event.event == "retrieval"
    ]

    assert "product_retrieval" in node_starts
    assert "knowledge_retrieval" in node_starts
    assert any(
        event["node"] == "product_retrieval"
        and event["summary"]["returned_products"] == 2
        and event["summary"]["candidate_product_ids"] == ["phone_001", "phone_002"]
        and event["summary"]["cache_status"] == "miss"
        for event in node_ends
    )
    assert any(
        event["node"] == "knowledge_retrieval"
        and event["summary"]["returned_chunks"] == 2
        and event["summary"]["chunk_ids"] == ["chunk_001", "chunk_002"]
        and event["summary"]["cache_status"] == "hit"
        for event in node_ends
    )
    assert any(
        event["type"] == "product"
        and event["returned_products"] == 2
        and event["candidate_product_ids"] == ["phone_001", "phone_002"]
        for event in retrieval_events
    )
    assert any(
        event["type"] == "knowledge"
        and event["returned_chunks"] == 2
        and event["chunk_ids"] == ["chunk_001", "chunk_002"]
        for event in retrieval_events
    )


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


def _product(product_id: str) -> ProductCandidate:
    return ProductCandidate(
        product_id=product_id,
        title=f"Camera Phone {product_id}",
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


def _citation(chunk_id: str) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
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

