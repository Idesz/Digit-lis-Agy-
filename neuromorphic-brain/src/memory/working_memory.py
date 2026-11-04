"""
Working Memory Module.
Provides high-speed contextual storage with TTL support.
Connects to Redis when available, or seamlessly falls back to a thread-safe in-memory cache.
"""

import time
import json
import threading
from typing import Any, Optional


class InMemoryCache:
    """Thread-safe in-memory key-value cache with TTL expiration."""

    def __init__(self):
        self._store: dict[str, tuple[str, Optional[float]]] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        expire_at = time.time() + ex if ex else None
        with self._lock:
            self._store[key] = (value, expire_at)
        return True

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._store:
                return None
            val, expire_at = self._store[key]
            if expire_at is not None and time.time() > expire_at:
                del self._store[key]
                return None
            return val

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def flushall(self) -> None:
        with self._lock:
            self._store.clear()

    def ping(self) -> bool:
        return True


class WorkingMemory:
    """
    Working Memory manager for rapid state retrieval and contextual caching.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, force_fallback: bool = False):
        self.is_redis_connected = False
        self._client = None

        if not force_fallback:
            try:
                import redis
                client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=0.5)
                client.ping()
                self._client = client
                self.is_redis_connected = True
            except Exception:
                self._client = InMemoryCache()
                self.is_redis_connected = False
        else:
            self._client = InMemoryCache()
            self.is_redis_connected = False

    def put(self, key: str, data: Any, ttl: Optional[int] = 3600) -> bool:
        """Serializes and stores item in working memory."""
        serialized = json.dumps(data)
        if self.is_redis_connected:
            return bool(self._client.set(name=key, value=serialized, ex=ttl))
        return self._client.set(key=key, value=serialized, ex=ttl)

    def get(self, key: str) -> Optional[Any]:
        """Retrieves and deserializes item from working memory."""
        raw = self._client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def delete(self, key: str) -> bool:
        if self.is_redis_connected:
            return bool(self._client.delete(key))
        return self._client.delete(key)

    def clear(self) -> None:
        self._client.flushall()
