"""Background pre-fetching of LLM narrations to reduce gameplay pauses."""

from __future__ import annotations

import asyncio
from typing import Coroutine


class PrefetchCache:
    """Cache for pre-generated LLM narrations with background task management."""

    def __init__(self):
        self._cache: dict[str, str] = {}
        self._pending: dict[str, asyncio.Task] = {}
        self._schedule_version: int = 0

    @property
    def schedule_version(self) -> int:
        return self._schedule_version

    def get(self, key: str) -> str | None:
        """Return cached result if available, else None."""
        return self._cache.get(key)

    def submit(self, key: str, coro: Coroutine) -> None:
        """Fire a background task to generate and cache a result.

        If the key is already cached or a task is already pending, the coro
        is closed without running.
        """
        if key in self._cache or key in self._pending:
            coro.close()
            return

        async def _run():
            try:
                result = await coro
                # Only store if key hasn't been invalidated while running
                if key not in self._cache:
                    self._cache[key] = result
            except Exception:
                pass  # Prefetch failures are non-critical
            finally:
                self._pending.pop(key, None)

        self._pending[key] = asyncio.create_task(_run())

    async def wait_or_generate(self, key: str, coro: Coroutine) -> str:
        """Cache hit -> return, pending -> await, miss -> run directly."""
        # Cache hit
        cached = self._cache.get(key)
        if cached is not None:
            coro.close()
            return cached

        # Pending task — await it
        task = self._pending.get(key)
        if task is not None:
            coro.close()
            await task
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # Miss — run directly
        result = await coro
        self._cache[key] = result
        return result

    def invalidate_schedule(self) -> None:
        """Bump schedule version so stale observation keys miss on lookup."""
        self._schedule_version += 1

    def clear(self) -> None:
        """Cancel all pending tasks and clear cache (for loop reset)."""
        for task in self._pending.values():
            task.cancel()
        self._pending.clear()
        self._cache.clear()
