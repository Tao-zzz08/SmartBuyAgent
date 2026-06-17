from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import time
from typing import Any, Protocol

from app.core.config import settings


class CacheService(Protocol):
    def get_json(self, key: str) -> dict[str, Any] | list[Any] | None:
        ...

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        ...

    def delete(self, key: str) -> None:
        ...

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        ...


class NullCacheService:
    def get_json(self, key: str) -> dict[str, Any] | list[Any] | None:
        return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        return None

    def delete(self, key: str) -> None:
        return None

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        return 1


@dataclass
class InMemoryCacheService:
    _store: dict[str, tuple[Any, float | None]] = field(default_factory=dict)

    def get_json(self, key: str) -> dict[str, Any] | list[Any] | None:
        item = self._store.get(key)
        if item is None:
            return None
        value, expires_at = item
        if expires_at is not None and expires_at <= time.time():
            self._store.pop(key, None)
            return None
        return deepcopy(value)

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else None
        self._store[key] = (deepcopy(value), expires_at)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        current = self.get_json(key)
        if isinstance(current, int):
            value = current + 1
        else:
            value = 1
        expires_at = (
            time.time() + ttl_seconds
            if ttl_seconds is not None and ttl_seconds > 0
            else None
        )
        self._store[key] = (value, expires_at)
        return value


class RedisCacheService:
    def __init__(self, client: Any) -> None:
        self.client = client

    def get_json(self, key: str) -> dict[str, Any] | list[Any] | None:
        raw_value = self.client.get(key)
        if raw_value is None:
            return None
        return json.loads(raw_value)

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        self.client.set(
            key,
            json.dumps(value, ensure_ascii=False),
            ex=ttl_seconds if ttl_seconds > 0 else None,
        )

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        value = int(self.client.incr(key))
        if ttl_seconds is not None and ttl_seconds > 0 and value == 1:
            self.client.expire(key, ttl_seconds)
        return value


_cache_service_singleton: CacheService | None = None


def get_cache_service() -> CacheService:
    global _cache_service_singleton
    if _cache_service_singleton is not None:
        return _cache_service_singleton

    if not settings.REDIS_URL:
        _cache_service_singleton = NullCacheService()
        return _cache_service_singleton

    try:
        from app.cache.redis_client import create_redis_client

        client = create_redis_client(settings.REDIS_URL)
        client.ping()
        _cache_service_singleton = RedisCacheService(client)
    except Exception:
        _cache_service_singleton = NullCacheService()
    return _cache_service_singleton
