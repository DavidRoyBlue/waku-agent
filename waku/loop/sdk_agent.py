"""The subscription loop — Waku's turn handed to the Claude Agent SDK.

agent.py is Waku's own loop: raw Messages API, needs an API key. This module
is the second loop: the Claude Agent SDK (the engine inside Claude Code) runs
the reason-act cycle itself, authenticated by your Claude subscription
(`claude login`) instead of a key. Waku's tools ride along as in-process MCP
tools, and the observer events + LoopResult contract match agent.py exactly —
tracing, the dashboard, and evals can't tell which loop answered.

Honest differences from agent.py:
  - WAKU_MAX_TOKENS is not enforced (the SDK has no per-call output cap);
    WAKU_MAX_ITERATIONS maps to the SDK's max_turns.
  - Streaming is per-message, not per-token: text arrives in turn-sized chunks.
  - The system prompt is fixed per turn (fine: Waku rebuilds it every turn and
    starts a fresh SDK session, so the retrieval gate's injections still land).

All claude_agent_sdk imports are lazy so core Waku runs without the extra:
uv pip install -e '.[claude-code]' to enable (needs Claude Code installed and
signed in — `claude login`, once per machine).
"""

from __future__ import annotations

import asyncio
from typing import Any


def build_prompt(messages: list[dict]) -> str:
    """Waku replays a sliding window of history each turn; the SDK takes one
    prompt string per session. Render the window as a labeled transcript with
    the current user message last."""
    *history, last = messages
    lines = [f'{"User" if m["role"] == "user" else "Waku"}: {m["content"]}'
             for m in history if isinstance(m.get("content"), str)]
    if not lines:
        return last["content"]
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nUser: {last['content']}"


def _tokens(usage: Any) -> tuple[int, int]:
    """SDK usage arrives as a dict or an object depending on version."""
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return int(getattr(usage, "input_tokens", 0)), int(getattr(usage, "output_tokens", 0))


def _run(coro):
    """asyncio.run, but callable from inside a running loop (telegram gateway):
    fall back to a fresh thread with its own event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
