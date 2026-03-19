"""Shared LLM query wrapper with resilient message parsing."""

from __future__ import annotations

import asyncio
import os

if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]

from claude_code_sdk import query, ClaudeCodeOptions

from .config import MODEL


async def llm_query(prompt: str, max_retries: int = 3, timeout: int = 300) -> str:
    """Query Claude with timeout protection and automatic retry.

    Handles rate limits, unknown SDK message types, and hung subprocesses
    via asyncio.wait_for timeout.
    """
    for attempt in range(max_retries + 1):
        try:
            result_text = ""

            async def _run():
                nonlocal result_text
                async for msg in query(
                    prompt=prompt,
                    options=ClaudeCodeOptions(
                        model=MODEL,
                        max_turns=1,
                    ),
                ):
                    if hasattr(msg, "content"):
                        for block in msg.content:
                            if hasattr(block, "text"):
                                result_text += block.text

            await asyncio.wait_for(_run(), timeout=timeout)
            return result_text
        except asyncio.TimeoutError:
            if attempt < max_retries:
                wait = 30 * (attempt + 1)
                print(f"  [Timeout after {timeout}s — waiting {wait}s before retry {attempt + 2}/{max_retries + 1}]")
                await asyncio.sleep(wait)
                continue
            raise TimeoutError(
                f"LLM query timed out after {max_retries + 1} attempts ({timeout}s each)"
            )
        except Exception as exc:
            is_rate_limit = (
                "rate_limit" in str(exc).lower()
                or "unknown message type" in str(exc).lower()
            )
            if is_rate_limit and attempt < max_retries:
                wait = 30 * (attempt + 1)
                print(f"  [Rate limited — waiting {wait}s before retry {attempt + 2}/{max_retries + 1}]")
                await asyncio.sleep(wait)
                continue
            raise
    return ""
