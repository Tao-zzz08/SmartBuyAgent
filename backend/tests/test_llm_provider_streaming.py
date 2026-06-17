from app.services.llm import (
    LLMMessage,
    MockLLMService,
    _iter_openai_compatible_deltas,
)


def test_mock_llm_service_stream_chat_yields_deterministic_chunks() -> None:
    service = MockLLMService()
    messages = [LLMMessage(role="user", content="recommend a phone")]

    chunks = list(service.stream_chat(messages))
    full_response = service.chat(messages).content

    assert len(chunks) > 1
    assert "".join(chunks) == full_response


def test_openai_compatible_delta_parser_reads_streaming_chunks() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"hello"}}]}',
        b'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    ]

    assert list(_iter_openai_compatible_deltas(lines)) == ["hello", " world"]

