from __future__ import annotations

from dataclasses import dataclass

from app.cache.cache_service import CacheService


class RateLimitExceeded(Exception):
    pass


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    count: int
    limit: int
    window_seconds: int
    cache_status: str


def check_rate_limit(
    cache_service: CacheService,
    key: str,
    limit: int,
    window_seconds: int,
) -> RateLimitResult:
    try:
        count = cache_service.incr(key, ttl_seconds=window_seconds)
    except Exception:
        return RateLimitResult(
            allowed=True,
            count=0,
            limit=limit,
            window_seconds=window_seconds,
            cache_status="failed",
        )

    if count > limit:
        raise RateLimitExceeded(f"rate limit exceeded for {key}")

    return RateLimitResult(
        allowed=True,
        count=count,
        limit=limit,
        window_seconds=window_seconds,
        cache_status="ok",
    )
