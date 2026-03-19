"""Shared LLM query wrapper with resilient message parsing."""

from __future__ import annotations

import asyncio
import os

if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]

from claude_code_sdk import query, ClaudeCodeOptions

from .config import MODEL


async def llm_query(prompt: str, max_retries: int = 2) -> str:
    """Query Claude with automatic retry on rate limits and unknown message types.

    Handles MessageParseError from unrecognized SDK message types (e.g.,
    rate_limit_event) by retrying after a brief pause.
    """
    for attempt in range(max_retries + 1):
        try:
            result_text = ""
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
            return result_text
        except Exception as exc:
            is_rate_limit = "rate_limit" in str(exc).lower() or "unknown message type" in str(exc).lower()
            if is_rate_limit and attempt < max_retries:
                wait = 10 * (attempt + 1)
                print(f"  [Rate limited — waiting {wait}s before retry {attempt + 2}/{max_retries + 1}]")
                await asyncio.sleep(wait)
                continue
            raise
    return ""
