from app.chat.llm_answer_composer import LLMAnswerComposer
from app.chat.query_understanding import QueryUnderstandingResult
from app.retrieval.retrieval_service import Citation, ProductCandidate
from app.services.llm import BaseLLMService, LLMMessage, LLMResponse


class StreamingLLMService(BaseLLMService):
    def __init__(self, chunks: list[str], chat_content: str | None = None) -> None:
        self.chunks = chunks
        self.chat_content = chat_content or "".join(chunks)
        self.stream_calls = 0
        self.chat_calls = 0

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.chat_calls += 1
        return LLMResponse(
            content=self.chat_content,
            model="fake-stream-model",
            provider="fake",
        )

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        self.stream_calls += 1
        yield from self.chunks


class FailingStreamingLLMService(StreamingLLMService):
    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        self.stream_calls += 1
        raise RuntimeError("stream unavailable")
        yield ""


def _query_result() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        raw_query="recommend a camera phone",
        intent="shopping_guide",
        category_id="cat_phone",
        category_path="digital/phone",
        budget_min=None,
        budget_max=3000,
        preferences=["camera"],
        need_clarification=False,
        clarification_question=None,
    )


def _product() -> ProductCandidate:
    return ProductCandidate(
        product_id="phone_001",
        title="Camera Phone",
        brand="Brand",
        category_id="cat_phone",
        price=2599,
        stock=10,
        description="A phone with camera-focused tags.",
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
        content_preview="Camera specs should be read with sensor and stabilization.",
        distance=0.2,
        score=0.8,
    )


def test_stream_compose_uses_provider_stream_and_returns_joined_answer() -> None:
    service = StreamingLLMService(["Choose ", "phone_001", " for camera."])
    composer = LLMAnswerComposer(service)
    tokens: list[str] = []

    answer = composer.stream_compose(
        query="recommend a camera phone",
        query_result=_query_result(),
        product_candidates=[_product()],
        citations=[_citation()],
        on_token=tokens.append,
    )

    assert service.stream_calls == 1
    assert service.chat_calls == 0
    assert tokens
    assert answer == "".join(tokens)


def test_stream_compose_falls_back_to_non_streaming_chat_when_stream_fails() -> None:
    service = FailingStreamingLLMService(
        chunks=[],
        chat_content="Fallback chat answer for phone_001.",
    )
    composer = LLMAnswerComposer(service)
    tokens: list[str] = []

    answer = composer.stream_compose(
        query="recommend a camera phone",
        query_result=_query_result(),
        product_candidates=[_product()],
        citations=[_citation()],
        on_token=tokens.append,
    )

    assert service.stream_calls == 1
    assert service.chat_calls == 1
    assert answer == "Fallback chat answer for phone_001."
    assert "".join(tokens) == answer
