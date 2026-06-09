from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import hashlib
import math

import httpx

from app.core.config import settings


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


class OpenAICompatibleEmbeddingService(BaseEmbeddingService):
    def __init__(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str | None,
        embedding_dim: int,
        timeout_seconds: float = 30.0,
        http_client: Any | None = None,
    ) -> None:
        if not api_base:
            raise ValueError("EMBEDDING_API_BASE is required for openai_compatible provider")
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for openai_compatible provider")
        if not model:
            raise ValueError("EMBEDDING_MODEL is required for openai_compatible provider")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.embedding_dim = embedding_dim
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": [text if text is not None else "" for text in texts],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._post_embeddings(payload=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Embedding response is not valid JSON: {exc}") from exc

        if not isinstance(body, dict):
            raise RuntimeError("Embedding response must be a JSON object")

        data = body.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding response missing data list")
        if len(data) != len(texts):
            raise RuntimeError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(data)}"
            )

        vectors: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise RuntimeError("Embedding response item missing embedding list")
            vector = [float(value) for value in item["embedding"]]
            if len(vector) != self.embedding_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(vector)}"
                )
            vectors.append(vector)

        return vectors

    def _post_embeddings(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        endpoint = f"{self.api_base}/embeddings"
        if self._http_client is not None:
            return self._http_client.post(endpoint, json=payload, headers=headers)

        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(endpoint, json=payload, headers=headers)


def get_embedding_service(
    provider: str | None = None,
    embedding_dim: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
) -> BaseEmbeddingService:
    selected_provider = (provider or settings.EMBEDDING_PROVIDER).lower()
    selected_dim = embedding_dim if embedding_dim is not None else settings.EMBEDDING_DIM

    if selected_provider == "mock":
        return MockEmbeddingService(embedding_dim=selected_dim)

    if selected_provider in {"openai_compatible", "real"}:
        return OpenAICompatibleEmbeddingService(
            api_base=api_base or settings.EMBEDDING_API_BASE,
            api_key=api_key or settings.EMBEDDING_API_KEY,
            model=model or settings.EMBEDDING_MODEL,
            embedding_dim=selected_dim,
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else settings.EMBEDDING_TIMEOUT_SECONDS
            ),
        )

    raise ValueError(f"Unsupported embedding provider: {selected_provider}")
