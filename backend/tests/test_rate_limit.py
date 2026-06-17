import pytest

from app.cache.cache_service import InMemoryCacheService
from app.cache.rate_limit import RateLimitExceeded, check_rate_limit


class FailingCache:
    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        raise RuntimeError("cache unavailable")


def test_rate_limit_blocks_after_limit() -> None:
    cache = InMemoryCacheService()

    check_rate_limit(cache, "smartbuy:rate:test", limit=2, window_seconds=10)
    check_rate_limit(cache, "smartbuy:rate:test", limit=2, window_seconds=10)

    with pytest.raises(RateLimitExceeded):
        check_rate_limit(cache, "smartbuy:rate:test", limit=2, window_seconds=10)


def test_rate_limit_allows_when_cache_fails() -> None:
    result = check_rate_limit(
        FailingCache(),
        "smartbuy:rate:test",
        limit=1,
        window_seconds=10,
    )

    assert result.allowed is True
    assert result.cache_status == "failed"
