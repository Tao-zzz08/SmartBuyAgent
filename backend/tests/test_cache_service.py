from app.cache.cache_service import InMemoryCacheService


def test_in_memory_cache_json_get_set_delete() -> None:
    cache = InMemoryCacheService()
    payload = {"items": [{"id": "phone_001"}], "count": 1}

    cache.set_json("smartbuy:test:json", payload, ttl_seconds=60)

    assert cache.get_json("smartbuy:test:json") == payload

    cache.delete("smartbuy:test:json")

    assert cache.get_json("smartbuy:test:json") is None


def test_in_memory_cache_incr() -> None:
    cache = InMemoryCacheService()

    assert cache.incr("smartbuy:test:counter", ttl_seconds=60) == 1
    assert cache.incr("smartbuy:test:counter", ttl_seconds=60) == 2
