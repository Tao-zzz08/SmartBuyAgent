from __future__ import annotations

from typing import Any

import redis


def create_redis_client(redis_url: str) -> Any:
    return redis.Redis.from_url(redis_url, decode_responses=True)
