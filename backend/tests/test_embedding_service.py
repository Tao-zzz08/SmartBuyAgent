import pytest

from app.services.embedding import (
    BaseEmbeddingService,
    MockEmbeddingService,
    get_embedding_service,
)


def test_embed_text_returns_float_vector() -> None:
    service = MockEmbeddingService(embedding_dim=32)

    vector = service.embed_text("手机续航")

    assert isinstance(vector, list)
    assert len(vector) == 32
    assert all(isinstance(value, float) for value in vector)


def test_embed_text_is_stable_for_same_text() -> None:
    service = MockEmbeddingService(embedding_dim=32)

    first = service.embed_text("拍照好的手机")
    second = service.embed_text("拍照好的手机")

    assert first == second


def test_embed_text_differs_for_different_texts() -> None:
    service = MockEmbeddingService(embedding_dim=32)

    first = service.embed_text("拍照好的手机")
    second = service.embed_text("通勤舒适的鞋")

    assert first != second


def test_embed_texts_returns_batch_vectors() -> None:
    service = MockEmbeddingService(embedding_dim=16)

    vectors = service.embed_texts(["手机", "鞋靴", "护肤"])

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
    assert len(service.embed_text("测试")) == 24


def test_get_embedding_service_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        get_embedding_service("unknown")
