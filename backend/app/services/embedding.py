from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math


class BaseEmbeddingService(ABC):
    embedding_dim: int

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Return one embedding vector for one text."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class MockEmbeddingService(BaseEmbeddingService):
    def __init__(self, embedding_dim: int = 32) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        self.embedding_dim = embedding_dim

    def embed_text(self, text: str) -> list[float]:
        digest = self._stable_digest(text, self.embedding_dim)
        values = [(byte / 127.5) - 1.0 for byte in digest]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    @staticmethod
    def _stable_digest(text: str, length: int) -> bytes:
        encoded = text.encode("utf-8")
        output = bytearray()
        counter = 0

        while len(output) < length:
            hasher = hashlib.sha256()
            hasher.update(counter.to_bytes(4, byteorder="big"))
            hasher.update(b":")
            hasher.update(encoded)
            output.extend(hasher.digest())
            counter += 1

        return bytes(output[:length])


def get_embedding_service(
    provider: str = "mock",
    embedding_dim: int = 32,
) -> BaseEmbeddingService:
    if provider.lower() == "mock":
        return MockEmbeddingService(embedding_dim=embedding_dim)

    raise ValueError(f"Unsupported embedding provider: {provider}")
