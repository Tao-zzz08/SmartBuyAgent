from __future__ import annotations

from typing import Any

import pytest

from app.services import embedding as embedding_module
from app.services.embedding import (
    BaseEmbeddingService,
    MockEmbeddingService,
    OpenAICompatibleEmbeddingService,
    get_embedding_service,
)


class FakeEmbeddingResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeEmbeddingClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        endpoint: str,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> FakeEmbeddingResponse:
        self.calls.append({"endpoint": endpoint, "json": json, "headers": headers})
        return FakeEmbeddingResponse(self.payload)


def test_embed_text_returns_float_vector() -> None:
    service = MockEmbeddingService(embedding_dim=32)

    vector = service.embed_text("phone battery")

    assert isinstance(vector, list)
    assert len(vector) == 32
    assert all(isinstance(value, float) for value in vector)


def test_embed_text_is_stable_for_same_text() -> None:
    service = MockEmbeddingService(embedding_dim=32)

    first = service.embed_text("camera phone")
    second = service.embed_text("camera phone")

    assert first == second


def test_embed_text_differs_for_different_texts() -> None:
    service = MockEmbeddingService(embedding_dim=32)

    first = service.embed_text("camera phone")
    second = service.embed_text("commute shoes")

    assert first != second


def test_embed_texts_returns_batch_vectors() -> None:
    service = MockEmbeddingService(embedding_dim=16)

    vectors = service.embed_texts(["phone", "shoes", "skincare"])

    assert len(vectors) == 3
    assert all(len(vector) == 16 for vector in vectors)


def test_empty_string_returns_fixed_dimension_vector() -> None:
    service = MockEmbeddingService(embedding_dim=32)

    vector = service.embed_text("")

    assert len(vector) == 32
    assert vector == service.embed_text("")


def test_get_embedding_service_returns_mock_service() -> None:
    service = get_embedding_service("mock", embedding_dim=24)

    assert isinstance(service, BaseEmbeddingService)
    assert service.embedding_dim == 24
    assert len(service.embed_text("test")) == 24


def test_get_embedding_service_uses_settings_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_module.settings, "EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(embedding_module.settings, "EMBEDDING_DIM", 12)

    service = get_embedding_service()

    assert isinstance(service, MockEmbeddingService)
    assert service.embedding_dim == 12


def test_get_embedding_service_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embedding_service("unknown")


def test_real_provider_missing_config_raises_value_error() -> None:
    with pytest.raises(ValueError, match="EMBEDDING_API_BASE"):
        OpenAICompatibleEmbeddingService(
            api_base=None,
            api_key="test-key",
            model="test-model",
            embedding_dim=3,
        )

    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        OpenAICompatibleEmbeddingService(
            api_base="https://embedding.example.com/v1",
            api_key=None,
            model="test-model",
            embedding_dim=3,
        )

    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        OpenAICompatibleEmbeddingService(
            api_base="https://embedding.example.com/v1",
            api_key="test-key",
            model=None,
            embedding_dim=3,
        )


def test_get_embedding_service_creates_openai_compatible_provider() -> None:
    service = get_embedding_service(
        provider="openai_compatible",
        embedding_dim=3,
        api_base="https://embedding.example.com/v1",
        api_key="test-key",
        model="test-model",
    )

    assert isinstance(service, OpenAICompatibleEmbeddingService)
    assert service.embedding_dim == 3


def test_real_provider_embed_texts_uses_openai_compatible_payload() -> None:
    fake_client = FakeEmbeddingClient(
        {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }
    )
    service = OpenAICompatibleEmbeddingService(
        api_base="https://embedding.example.com/v1/",
        api_key="test-key",
        model="test-model",
        embedding_dim=3,
        http_client=fake_client,
    )

    vectors = service.embed_texts(["phone", "shoes"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["endpoint"] == "https://embedding.example.com/v1/embeddings"
    assert fake_client.calls[0]["json"] == {
        "model": "test-model",
        "input": ["phone", "shoes"],
    }
    assert fake_client.calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_real_provider_embed_text_returns_single_vector() -> None:
    fake_client = FakeEmbeddingClient({"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    service = OpenAICompatibleEmbeddingService(
        api_base="https://embedding.example.com/v1",
        api_key="test-key",
        model="test-model",
        embedding_dim=3,
        http_client=fake_client,
    )

    vector = service.embed_text("phone")

    assert vector == [0.1, 0.2, 0.3]


def test_real_provider_returns_empty_batch_without_http_call() -> None:
    fake_client = FakeEmbeddingClient({"data": []})
    service = OpenAICompatibleEmbeddingService(
        api_base="https://embedding.example.com/v1",
        api_key="test-key",
        model="test-model",
        embedding_dim=3,
        http_client=fake_client,
    )

    assert service.embed_texts([]) == []
    assert fake_client.calls == []


def test_real_provider_response_count_mismatch_raises_error() -> None:
    fake_client = FakeEmbeddingClient({"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    service = OpenAICompatibleEmbeddingService(
        api_base="https://embedding.example.com/v1",
        api_key="test-key",
        model="test-model",
        embedding_dim=3,
        http_client=fake_client,
    )

    with pytest.raises(RuntimeError, match="count mismatch"):
        service.embed_texts(["phone", "shoes"])
