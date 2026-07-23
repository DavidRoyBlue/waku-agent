# Claude Code Subscription Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `claude-code` provider that runs Waku on a Claude Max/Pro subscription via the Claude Agent SDK (no API key), keeping the LoopResult/observer contract so tracing, dashboard, and evals keep working.

**Architecture:** Approach B — a second, honest loop. New `waku/loop/sdk_agent.py` exposes `run_sdk_loop(...)` (same signature and `LoopResult` contract as `waku/loop/agent.py:run_loop`) which hands the whole turn to the Agent SDK: Waku's tools become in-process MCP tools, built-in Claude Code tools are disabled, and SDK stream messages are translated into the same observer events (`text`, `tool`, `llm`). `app.py` branches once on provider kind. Memory's small-model calls (retrieval gate + consolidation) go through a thin `ClaudeAgentClient` whose `messages.create` is a one-shot SDK query — the whole app runs keyless.

**Tech Stack:** Python 3.10+, `claude-agent-sdk` (optional extra `[claude-code]`), stdlib asyncio. Tests use a fake `claude_agent_sdk` module injected into `sys.modules` — the deterministic suite never needs the real SDK or a login.

## Global Constraints

- No new core dependencies: `claude-agent-sdk` lives ONLY behind the `[claude-code]` extra in `pyproject.toml`; all imports of it are lazy (inside functions), so core Waku runs without it.
- Tests live in `evals/deterministic/` (pytest, 0/1, offline) — never `tests/`.
- No emojis anywhere (dashboard, CLI, README).
- Providers framed neutrally in docs — `claude-code` is "run on your Claude subscription", no ranking language.
- Gate before push: `make gate` must pass.
- Commit each task when its tests pass. Branch: `worktree-agent-sdk-provider` (already checked out). Never push to main.
- Honest limitations, documented not hidden: `WAKU_MAX_TOKENS` is not enforceable through the SDK (no per-call output cap); streaming is per-message chunks, not per-token; `max_iterations` maps to SDK `max_turns`.
- SDK facts (verified 2026-07-23): package `claude-agent-sdk`; subscription auth is used when `ANTHROPIC_API_KEY` is unset and Claude Code is logged in (`claude login`); `ClaudeAgentOptions(tools=[])` disables all built-ins; custom tools via `@tool(name, description, schema)` + `create_sdk_mcp_server(name=, version=, tools=)`, allow-listed as `mcp__<server>__<tool>`; `permission_mode="dontAsk"` never prompts and denies unlisted tools; stream yields `AssistantMessage` (with `TextBlock`/`ToolUseBlock`) then `ResultMessage` with `.result`, `.subtype` ("success", "error_max_turns", ...), `.num_turns`, `.usage` (input_tokens/output_tokens, may be dict or object), `.session_id`; tool handlers are `async def handler(args: dict) -> {"content": [{"type": "text", "text": ...}]}`.

---

### Task 1: `sdk_agent.py` pure helpers — `build_prompt` and `_tokens`

**Files:**
- Create: `waku/loop/sdk_agent.py`
- Test: `evals/deterministic/test_sdk_agent.py`

**Interfaces:**
- Produces: `build_prompt(messages: list[dict]) -> str` — renders Waku's replayed history window + current user message into one SDK prompt string. `_tokens(usage) -> tuple[int, int]` — (input, output) tokens from a dict-or-object usage, (0, 0) on None. `_run(coro)` — asyncio.run that survives being called from inside a running event loop.

- [ ] **Step 1: Write the failing tests**

```python
# evals/deterministic/test_sdk_agent.py
"""OFFLINE checks for the subscription loop (waku/loop/sdk_agent.py).

The real claude-agent-sdk needs a Claude Code login, so these tests inject a
fake `claude_agent_sdk` module into sys.modules — the deterministic suite
stays keyless and offline, like every other provider test."""

from __future__ import annotations

import sys
import types

import pytest


def test_build_prompt_single_message_is_bare():
    from waku.loop.sdk_agent import build_prompt

    assert build_prompt([{"role": "user", "content": "hello"}]) == "hello"


def test_build_prompt_renders_history_as_transcript():
    from waku.loop.sdk_agent import build_prompt

    prompt = build_prompt([
        {"role": "user", "content": "book dentist tuesday"},
        {"role": "assistant", "content": "Done, 10am."},
        {"role": "user", "content": "when was it again?"},
    ])
    assert prompt == (
        "Conversation so far:\n"
        "User: book dentist tuesday\n"
        "Waku: Done, 10am.\n\n"
        "User: when was it again?"
    )


def test_build_prompt_skips_non_string_content():
    from waku.loop.sdk_agent import build_prompt

    # tool blocks never appear in the replayed window (respond() rebuilds it
    # from stored text turns), but a stray one must not crash the prompt
    prompt = build_prompt([
        {"role": "user", "content": [{"type": "tool_result"}]},
        {"role": "user", "content": "hi"},
    ])
    assert prompt == "hi"


def test_tokens_reads_dict_object_and_none():
    from waku.loop.sdk_agent import _tokens

    assert _tokens({"input_tokens": 7, "output_tokens": 3}) == (7, 3)
    obj = types.SimpleNamespace(input_tokens=11, output_tokens=5)
    assert _tokens(obj) == (11, 5)
    assert _tokens(None) == (0, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'waku.loop.sdk_agent'`

- [ ] **Step 3: Write the implementation**

```python
# waku/loop/sdk_agent.py
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
pip install -e '.[claude-code]' to enable, then `claude login` once.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from waku.loop.agent import LoopResult, Observer
from waku.tools.registry import ToolRegistry


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add waku/loop/sdk_agent.py evals/deterministic/test_sdk_agent.py
git commit -m "feat: sdk_agent helpers — transcript prompt + usage normalization

First slice of the subscription loop (Claude Agent SDK provider): the pure
pieces with offline tests. build_prompt renders Waku's replayed history
window into the one-prompt-per-session shape the SDK needs; _tokens absorbs
the SDK's dict-or-object usage; _run makes asyncio.run safe under the
telegram gateway's running loop."
```

---

### Task 2: `ClaudeAgentClient` — one-shot SDK queries for the memory subsystem

**Files:**
- Modify: `waku/loop/sdk_agent.py` (append)
- Test: `evals/deterministic/test_sdk_agent.py` (append)

**Interfaces:**
- Consumes: `build_prompt`, `_tokens`, `_run` from Task 1.
- Produces: `class ClaudeAgentClient` with `.messages.create(*, model, messages, max_tokens=1024, system=None, tools=None)` returning an Anthropic-shaped object: `.content` = list with one `SimpleNamespace(type="text", text=...)`, `.stop_reason="end_turn"`, `.usage.input_tokens` / `.usage.output_tokens`. This is exactly what `waku/memory/retrieval_gate.py:42` and `waku/memory/consolidation.py:54` consume. Deliberately no `.messages.stream` attribute (so `agent.py:60`'s `hasattr` check stays False if it ever sees this client).

- [ ] **Step 1: Write the failing tests (fake SDK fixture + client tests)**

Append to `evals/deterministic/test_sdk_agent.py`:

```python
@pytest.fixture()
def fake_sdk(monkeypatch):
    """Install a stub claude_agent_sdk. Configure it via mod.script — a list of
    messages query() will yield. ToolUseBlock entries are not yielded; instead
    the stub invokes the matching registered MCP tool handler, like the real
    SDK's inner loop does."""
    mod = types.ModuleType("claude_agent_sdk")

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class ToolUseBlock:
        def __init__(self, name, input):
            self.name, self.input = name, input

    class AssistantMessage:
        def __init__(self, content):
            self.content = content

    class ResultMessage:
        def __init__(self, result, subtype="success", num_turns=1, usage=None):
            self.result, self.subtype, self.num_turns = result, subtype, num_turns
            self.usage = {"input_tokens": 10, "output_tokens": 4} if usage is None else usage
            self.session_id, self.total_cost_usd, self.stop_reason = "s1", 0.0, "end_turn"

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            mod.last_options = self

    def tool(name, description, schema):
        def deco(fn):
            fn._meta = (name, description, schema)
            return fn
        return deco

    def create_sdk_mcp_server(name, version, tools):
        mod.last_server = {"name": name, "version": version, "tools": tools}
        return mod.last_server

    async def query(prompt=None, options=None):
        mod.last_prompt = prompt
        for message in mod.script:
            if isinstance(message, ToolUseBlock):
                for fn in getattr(mod, "last_server", {"tools": []})["tools"]:
                    if fn._meta[0] == message.name:
                        await fn(message.input)
                continue
            yield message

    mod.TextBlock, mod.ToolUseBlock = TextBlock, ToolUseBlock
    mod.AssistantMessage, mod.ResultMessage = AssistantMessage, ResultMessage
    mod.ClaudeAgentOptions = ClaudeAgentOptions
    mod.tool, mod.create_sdk_mcp_server, mod.query = tool, create_sdk_mcp_server, query
    mod.script = []
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    return mod


def test_client_create_returns_anthropic_shape(fake_sdk):
    from waku.loop.sdk_agent import ClaudeAgentClient

    fake_sdk.script = [fake_sdk.ResultMessage('{"retrieve": true}')]
    response = ClaudeAgentClient().messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=600,
        messages=[{"role": "user", "content": "gate this"}],
    )
    # exactly what retrieval_gate.py / consolidation.py read:
    assert "".join(b.text for b in response.content if b.type == "text") == '{"retrieve": true}'
    assert response.usage.input_tokens == 10 and response.usage.output_tokens == 4
    assert response.stop_reason == "end_turn"
    # one-shot: no tools, one turn, prompt passed through
    assert fake_sdk.last_options.tools == []
    assert fake_sdk.last_options.max_turns == 1
    assert fake_sdk.last_prompt == "gate this"


def test_client_has_no_stream_attribute(fake_sdk):
    from waku.loop.sdk_agent import ClaudeAgentClient

    assert not hasattr(ClaudeAgentClient().messages, "stream")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py -v -k client`
Expected: FAIL with `ImportError: cannot import name 'ClaudeAgentClient'`

- [ ] **Step 3: Write the implementation**

Append to `waku/loop/sdk_agent.py`:

```python
class ClaudeAgentClient:
    """messages.create() for the memory subsystem (retrieval gate +
    consolidation): plain one-shot completions over the subscription.
    The loop itself never uses this — run_sdk_loop talks to the SDK directly."""

    def __init__(self) -> None:
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, messages: list[dict], max_tokens: int = 1024,
                system: str | None = None, tools: list | None = None):
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

        kwargs: dict[str, Any] = {"model": model, "tools": [], "max_turns": 1}
        if system:
            kwargs["system_prompt"] = system
        options = ClaudeAgentOptions(**kwargs)

        async def run() -> tuple[str, Any]:
            text, usage = "", None
            async for message in query(prompt=build_prompt(messages), options=options):
                if isinstance(message, ResultMessage):
                    text, usage = message.result or "", message.usage
            return text, usage

        text, usage = _run(run())
        tokens_in, tokens_out = _tokens(usage)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add waku/loop/sdk_agent.py evals/deterministic/test_sdk_agent.py
git commit -m "feat: ClaudeAgentClient — memory's small-model calls over the subscription

The retrieval gate and consolidation summarizer call messages.create with no
tools; this thin client satisfies that exact shape with a one-shot Agent SDK
query (tools=[], max_turns=1), so the whole memory pillar runs keyless.
Tested against a fake claude_agent_sdk injected into sys.modules — the
deterministic suite needs neither the SDK nor a login."
```

---

### Task 3: `run_sdk_loop` — the subscription loop with observer/LoopResult parity

**Files:**
- Modify: `waku/loop/sdk_agent.py` (append)
- Test: `evals/deterministic/test_sdk_agent.py` (append)

**Interfaces:**
- Consumes: `build_prompt`, `_tokens`, `_run`, `LoopResult`, `Observer`, `ToolRegistry` (`.schemas() -> [{"name", "description", "input_schema"}]`, `.execute(name, args) -> str`).
- Produces: `run_sdk_loop(client, model, system, messages, tools, max_iterations=10, max_tokens=2048, observer=None, stream=False) -> LoopResult` — signature-identical to `agent.py:run_loop` (`client` accepted but unused, for call-site parity). Emits `notify("tool", {"tool", "args", "output"})` per tool call, `notify("text", {"delta"})` per assistant text block when `stream=True`, `notify("llm", {"iteration", "stop_reason", "usage": {"in", "out"}})` once at the end.

- [ ] **Step 1: Write the failing tests**

Append to `evals/deterministic/test_sdk_agent.py`:

```python
def _registry():
    from waku.tools.registry import Tool, ToolRegistry

    calls = []
    registry = ToolRegistry()
    registry.register(Tool(
        name="save_note", description="Save a note.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}},
                      "required": ["text"]},
        fn=lambda text: calls.append(text) or f"Saved: {text}",
    ))
    return registry, calls


def test_run_sdk_loop_returns_reply_and_usage_events(fake_sdk):
    from waku.loop.sdk_agent import run_sdk_loop

    registry, _ = _registry()
    events = []
    fake_sdk.script = [
        fake_sdk.AssistantMessage([fake_sdk.TextBlock("On it.")]),
        fake_sdk.ResultMessage("All set.", num_turns=2),
    ]
    result = run_sdk_loop(client=None, model="claude-sonnet-5", system="You are Waku.",
                          messages=[{"role": "user", "content": "hi"}], tools=registry,
                          max_iterations=7, observer=lambda kind, ev: events.append((kind, ev)))
    assert result.reply == "All set."
    assert result.iterations == 2
    kinds = [k for k, _ in events]
    assert "llm" in kinds
    llm = dict(events)["llm"]
    assert llm["usage"] == {"in": 10, "out": 4} and llm["stop_reason"] == "success"


def test_run_sdk_loop_executes_waku_tools_and_notifies(fake_sdk):
    from waku.loop.sdk_agent import run_sdk_loop

    registry, calls = _registry()
    events = []
    fake_sdk.script = [
        fake_sdk.ToolUseBlock("save_note", {"text": "milk"}),
        fake_sdk.ResultMessage("Noted.", num_turns=2),
    ]
    result = run_sdk_loop(client=None, model="claude-sonnet-5", system="s",
                          messages=[{"role": "user", "content": "note milk"}],
                          tools=registry, observer=lambda k, e: events.append((k, e)))
    assert calls == ["milk"]                       # the real Waku tool ran
    assert result.tool_calls == [{"tool": "save_note", "args": {"text": "milk"},
                                  "output": "Saved: milk"}]
    assert ("tool", result.tool_calls[0]) in events


def test_run_sdk_loop_locks_down_builtins_and_allows_only_waku_tools(fake_sdk):
    from waku.loop.sdk_agent import run_sdk_loop

    registry, _ = _registry()
    fake_sdk.script = [fake_sdk.ResultMessage("ok")]
    run_sdk_loop(client=None, model="m", system="s",
                 messages=[{"role": "user", "content": "x"}], tools=registry,
                 max_iterations=5)
    options = fake_sdk.last_options
    assert options.tools == []                     # no Bash/Edit/WebSearch/...
    assert options.allowed_tools == ["mcp__waku__save_note"]
    assert options.permission_mode == "dontAsk"
    assert options.max_turns == 5
    assert options.system_prompt == "s"


def test_run_sdk_loop_streams_text_deltas_only_when_asked(fake_sdk):
    from waku.loop.sdk_agent import run_sdk_loop

    registry, _ = _registry()
    fake_sdk.script = [
        fake_sdk.AssistantMessage([fake_sdk.TextBlock("chunk")]),
        fake_sdk.ResultMessage("chunk"),
    ]
    for stream, expected in ((True, [("text", {"delta": "chunk"})]), (False, [])):
        events = []
        run_sdk_loop(client=None, model="m", system="s",
                     messages=[{"role": "user", "content": "x"}], tools=registry,
                     observer=lambda k, e: events.append((k, e)), stream=stream)
        assert [e for e in events if e[0] == "text"] == expected


def test_run_sdk_loop_empty_reply_says_how_to_fix(fake_sdk):
    from waku.loop.sdk_agent import run_sdk_loop

    registry, _ = _registry()
    fake_sdk.script = [fake_sdk.ResultMessage(None, subtype="error_during_execution")]
    result = run_sdk_loop(client=None, model="m", system="s",
                          messages=[{"role": "user", "content": "x"}], tools=registry)
    assert "claude login" in result.reply


def test_run_sdk_loop_max_turns_message_matches_waku_loop(fake_sdk):
    from waku.loop.sdk_agent import run_sdk_loop

    registry, _ = _registry()
    fake_sdk.script = [fake_sdk.ResultMessage(None, subtype="error_max_turns", num_turns=3)]
    result = run_sdk_loop(client=None, model="m", system="s",
                          messages=[{"role": "user", "content": "x"}], tools=registry,
                          max_iterations=3)
    assert "iteration limit" in result.reply
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py -v -k run_sdk_loop`
Expected: FAIL with `ImportError: cannot import name 'run_sdk_loop'`

- [ ] **Step 3: Write the implementation**

Append to `waku/loop/sdk_agent.py`:

```python
def run_sdk_loop(
    client: Any,                # unused — same signature as agent.run_loop
    model: str,
    system: str,
    messages: list[dict],
    tools: ToolRegistry,
    max_iterations: int = 10,
    max_tokens: int = 2048,     # not enforceable via the SDK — see module docstring
    observer: Observer | None = None,
    stream: bool = False,
) -> LoopResult:
    """One agent turn, run by the Agent SDK instead of agent.py's while-loop.
    Waku's tools are served to it as in-process MCP tools; every execution
    still goes through ToolRegistry.execute, so safety and tracing behave
    exactly like the home loop."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        create_sdk_mcp_server,
        query,
        tool as sdk_tool,
    )

    notify = observer or (lambda kind, ev: None)
    result = LoopResult(reply="")

    def to_sdk_tool(schema: dict):
        name = schema["name"]

        async def handler(args: dict) -> dict:
            output = tools.execute(name, args)
            event = {"tool": name, "args": args, "output": output}
            result.tool_calls.append(event)
            notify("tool", event)
            return {"content": [{"type": "text", "text": output}]}

        return sdk_tool(name, schema["description"], schema["input_schema"])(handler)

    schemas = tools.schemas()
    server = create_sdk_mcp_server(name="waku", version="1.0.0",
                                   tools=[to_sdk_tool(s) for s in schemas])
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system,
        tools=[],                          # no Claude Code built-ins — Waku's tools only
        mcp_servers={"waku": server},
        allowed_tools=[f"mcp__waku__{s['name']}" for s in schemas],
        permission_mode="dontAsk",         # never prompt; deny anything not listed
        max_turns=max_iterations,
    )

    async def run() -> None:
        async for message in query(prompt=build_prompt(messages), options=options):
            if isinstance(message, AssistantMessage):
                if stream:
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text:
                            notify("text", {"delta": block.text})
            elif isinstance(message, ResultMessage):
                result.iterations = message.num_turns or 1
                if message.result:
                    result.reply = message.result
                tokens_in, tokens_out = _tokens(message.usage)
                notify("llm", {"iteration": result.iterations,
                               "stop_reason": message.subtype,
                               "usage": {"in": tokens_in, "out": tokens_out}})
                if message.subtype == "error_max_turns":
                    result.reply = ("(I hit my iteration limit before finishing — "
                                    "try breaking the request into smaller steps.)")

    _run(run())
    if not result.reply:
        result.reply = ("(the subscription loop returned no reply — is Claude Code "
                        "logged in? Run `claude login`.)")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py -v`
Expected: 12 PASS

- [ ] **Step 5: Commit**

```bash
git add waku/loop/sdk_agent.py evals/deterministic/test_sdk_agent.py
git commit -m "feat: run_sdk_loop — the subscription loop, contract-identical to agent.py

The Agent SDK runs the reason-act cycle; Waku's tools ride along as
in-process MCP tools (server 'waku', allow-listed, permission_mode=dontAsk,
ALL built-ins disabled via tools=[]). Observer events and LoopResult match
run_loop exactly, so tracing, dashboard telemetry, and turn metadata are
loop-agnostic. Survives: tool execution + notify parity, builtin lockdown,
stream on/off, empty-reply and max-turns fallbacks — all offline via the
fake SDK."
```

---

### Task 4: Provider registration — `claude-code` in `PROVIDERS` + `get_client` branch

**Files:**
- Modify: `waku/loop/models.py` (PROVIDERS dict ~line 54-98; `get_client` ~line 101; imports header)
- Modify: `waku/ops/dashboard.py` (PRICING dict — find with `grep -n "^PRICING" waku/ops/dashboard.py`)
- Modify: `evals/deterministic/test_providers.py` (fixture + two parametrized tests)
- Test: `evals/deterministic/test_sdk_agent.py` (append)

**Interfaces:**
- Consumes: `ClaudeAgentClient` from Task 2.
- Produces: `PROVIDERS["claude-code"]` = `Provider("sdk", "", None, ...)` — kind `"sdk"`, empty `key_env`. `get_client` returns `ClaudeAgentClient` for kind `"sdk"` (before any key check), raises `SystemExit` mentioning `[claude-code]` and `claude login` when the SDK isn't installed. Task 5 and 6 depend on `PROVIDERS[name].kind == "sdk"` as the branch signal.

- [ ] **Step 1: Write the failing tests**

Append to `evals/deterministic/test_sdk_agent.py`:

```python
def test_claude_code_provider_registered():
    from waku.loop.models import PROVIDERS

    provider = PROVIDERS["claude-code"]
    assert provider.kind == "sdk"
    assert provider.key_env == ""          # subscription login, no key env var
    assert provider.base_url is None
    assert provider.default_pair() == ["claude-opus-4-8", "claude-sonnet-5"]


def test_get_client_builds_sdk_client_without_any_key(fake_sdk, monkeypatch):
    from waku.config import Settings
    from waku.loop.models import get_client
    from waku.loop.sdk_agent import ClaudeAgentClient

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(provider="claude-code", model="", small_model="",
                        api_key="", base_url=None)
    client = get_client(settings)
    assert isinstance(client, ClaudeAgentClient)
    assert settings.model == "claude-sonnet-5"
    assert settings.small_model == "claude-haiku-4-5-20251001"


def test_get_client_without_sdk_says_how_to_install(monkeypatch):
    from waku.config import Settings
    from waku.loop.models import get_client

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)  # import -> ImportError
    settings = Settings(provider="claude-code", model="", small_model="",
                        api_key="", base_url=None)
    with pytest.raises(SystemExit, match=r"claude-code.*claude login"):
        get_client(settings)
```

Update `evals/deterministic/test_providers.py` — three edits so the keyless provider doesn't break the table checks:

```python
# in the fake_keys fixture (line 21-22), guard the empty key_env:
    for provider in PROVIDERS.values():
        if provider.key_env:                        # claude-code has no key env
            monkeypatch.setenv(provider.key_env, "fake-key-for-tests")
```

```python
# at the top of test_get_client_builds_the_right_wire (after line 30):
    if provider.kind == "sdk":
        pytest.skip("subscription provider — covered offline in test_sdk_agent.py")
```

```python
# at the top of test_missing_key_exits_with_the_key_name (after line 41):
    if not PROVIDERS[name].key_env:
        pytest.skip("subscription provider — no key to miss")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py evals/deterministic/test_providers.py -v`
Expected: the three new tests FAIL with `KeyError: 'claude-code'`; existing provider tests still pass.

- [ ] **Step 3: Write the implementation**

In `waku/loop/models.py`, add `import sys` to the imports header if absent. Add to `PROVIDERS` (after the `"anthropic"` entry, keeping the Claude entries together):

```python
    # Not an API at all: the Claude Agent SDK runs the turn under your Claude
    # subscription (claude login) — kind "sdk" routes app.py to run_sdk_loop
    # and get_client to the thin memory client. No key env; keyless on purpose.
    "claude-code": Provider("sdk", "", None,
                            "claude-sonnet-5", "claude-haiku-4-5-20251001",
                            flagship="claude-opus-4-8", fast="claude-sonnet-5"),
```

In `get_client` (waku/loop/models.py:101), insert after the unknown-provider check and BEFORE the api_key lookup:

```python
    if provider.kind == "sdk":
        settings.model = settings.model or provider.model
        settings.small_model = settings.small_model or provider.small_model
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError:
            raise SystemExit(
                "Provider 'claude-code' runs on the Claude Agent SDK: "
                "pip install -e '.[claude-code]', then log in once with "
                "`claude login`. It uses your Claude subscription — no API key."
            )
        if os.getenv("ANTHROPIC_API_KEY"):
            print("note: ANTHROPIC_API_KEY is set, so the Agent SDK will bill that "
                  "key instead of your subscription. Unset it to use the subscription.",
                  file=sys.stderr)
        from waku.loop.sdk_agent import ClaudeAgentClient

        return ClaudeAgentClient()
```

In `waku/ops/dashboard.py`, add to the `PRICING` dict:

```python
    "claude-code": (0.0, 0.0),   # subscription-covered — no per-token bill
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py evals/deterministic/test_providers.py -v`
Expected: ALL PASS (including `test_dashboard_pricing_covers_every_provider[claude-code]` and `test_model_listing_falls_back_without_a_catalog[claude-code]`, which auto-parametrize over the new entry).

- [ ] **Step 5: Commit**

```bash
git add waku/loop/models.py waku/ops/dashboard.py evals/deterministic/test_sdk_agent.py evals/deterministic/test_providers.py
git commit -m "feat: register the claude-code subscription provider

New PROVIDERS entry with kind 'sdk' and an empty key_env: get_client routes
it to ClaudeAgentClient before any key check, with a fixable SystemExit when
the extra isn't installed and a stderr note when ANTHROPIC_API_KEY would
shadow the subscription. Provider-table checks updated for a keyless entry;
PRICING carries it at (0,0) — subscription-covered."
```

---

### Task 5: `app.py` — pick the loop by provider kind

**Files:**
- Modify: `waku/app.py` (imports; `__init__` ~line 33; `respond` line 69)
- Test: `evals/deterministic/test_sdk_agent.py` (append)

**Interfaces:**
- Consumes: `run_sdk_loop` (Task 3), `PROVIDERS[...].kind == "sdk"` (Task 4).
- Produces: `Waku._run_loop` — the chosen loop callable; `respond()` calls it with the exact same keyword arguments it passed `run_loop` before. No other behavior changes.

- [ ] **Step 1: Write the failing test**

Append to `evals/deterministic/test_sdk_agent.py`:

```python
def test_waku_dispatches_to_the_sdk_loop(fake_sdk, tmp_path, monkeypatch):
    from waku.app import Waku
    from waku.config import Settings
    from waku.loop.agent import run_loop
    from waku.loop.sdk_agent import run_sdk_loop

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sdk_settings = Settings(provider="claude-code", model="", small_model="",
                            api_key="", base_url=None, home=tmp_path / "sdk")
    assert Waku(settings=sdk_settings, client=object())._run_loop is run_sdk_loop

    api_settings = Settings(provider="anthropic", model="", small_model="",
                            api_key="", base_url=None, home=tmp_path / "api")
    assert Waku(settings=api_settings, client=object())._run_loop is run_loop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py::test_waku_dispatches_to_the_sdk_loop -v`
Expected: FAIL with `AttributeError: 'Waku' object has no attribute '_run_loop'`

- [ ] **Step 3: Write the implementation**

In `waku/app.py`, change the models import (line 12) to:

```python
from waku.loop.models import PROVIDERS, get_client
```

In `__init__`, after `self.tracer = Tracer(self.settings)` (line 34), add:

```python
        # Two loops, one contract: our own while-loop (agent.py) for API
        # providers, the Agent SDK's managed loop for the subscription
        # provider. Same signature, same LoopResult, same observer events.
        provider = PROVIDERS.get(self.settings.provider)
        if provider is not None and provider.kind == "sdk":
            from waku.loop.sdk_agent import run_sdk_loop

            self._run_loop = run_sdk_loop
        else:
            self._run_loop = run_loop
```

In `respond()` (line 69), change `result = run_loop(` to `result = self._run_loop(` — arguments unchanged.

- [ ] **Step 4: Run the test and the neighbors that exercise Waku wiring**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py evals/deterministic/test_working_memory.py evals/deterministic/test_history_window.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add waku/app.py evals/deterministic/test_sdk_agent.py
git commit -m "feat: app.py picks the loop by provider kind

One branch at assembly time: kind 'sdk' wires respond() to run_sdk_loop,
everything else keeps agent.py's loop. Call site and arguments unchanged —
the LoopResult/observer contract makes the two interchangeable."
```

---

### Task 6: Dashboard — keyless provider in Settings (backend + frontend)

**Files:**
- Modify: `waku/ops/dashboard.py` (`default_pinned_specs` line 1260-1270; `settings_info` line 1318-1368; `apply_settings` line 1388-1390)
- Modify: `waku/ops/static/js/views.js` (line 273-276)
- Test: `evals/deterministic/test_sdk_agent.py` (append)

**Interfaces:**
- Consumes: `PROVIDERS["claude-code"].kind == "sdk"`, empty `key_env`.
- Produces: `settings_info()["providers"]` entries gain `"subscription": bool`; for the sdk provider `key_set` means "SDK installed" (`importlib.util.find_spec("claude_agent_sdk") is not None`). `apply_settings` writable set skips empty key envs. `default_pinned_specs` includes claude-code when the SDK is installed.

- [ ] **Step 1: Write the failing tests**

Append to `evals/deterministic/test_sdk_agent.py`:

```python
def test_settings_info_marks_subscription_provider(fake_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("WAKU_HOME", str(tmp_path))
    from waku.ops.dashboard import settings_info

    rows = {p["name"]: p for p in settings_info()["providers"]}
    assert rows["claude-code"]["subscription"] is True
    assert rows["claude-code"]["key_env"] == ""
    assert rows["claude-code"]["key_set"] is True      # fake SDK is importable
    assert rows["anthropic"]["subscription"] is False


def test_default_pins_include_claude_code_when_sdk_installed(fake_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("WAKU_HOME", str(tmp_path))
    from waku.ops.dashboard import default_pinned_specs

    specs = default_pinned_specs()
    assert "claude-code:claude-opus-4-8" in specs
    assert "claude-code:claude-sonnet-5" in specs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py -v -k "settings_info or default_pins"`
Expected: FAIL (`KeyError: 'subscription'`; missing specs)

- [ ] **Step 3: Write the implementation**

In `waku/ops/dashboard.py`:

`default_pinned_specs` (line 1267-1269) — replace the loop body:

```python
    import importlib.util

    sdk_ready = importlib.util.find_spec("claude_agent_sdk") is not None
    for name, prov in PROVIDERS.items():
        usable = sdk_ready if prov.kind == "sdk" else bool(os.getenv(prov.key_env))
        if usable:
            specs += [f"{name}:{m}" for m in prov.default_pair()]
```

`settings_info` providers list (line 1351-1357) — replace with:

```python
        "providers": [
            {"name": name, "key_env": p.key_env,
             # for the subscription provider, "key_set" means "SDK installed"
             "key_set": (sdk_ready if p.kind == "sdk" else bool(os.getenv(p.key_env))),
             "key_last4": (os.getenv(p.key_env) or "")[-4:] if p.key_env else "",
             "subscription": p.kind == "sdk",
             "default_model": p.model, "default_small_model": p.small_model}
            for name, p in PROVIDERS.items()
        ],
```

and near the top of `settings_info` (after `s = load_settings()`, line 1324):

```python
    import importlib.util

    sdk_ready = importlib.util.find_spec("claude_agent_sdk") is not None
```

`apply_settings` writable set (line 1390) — change the union term to skip empty names:

```python
                | {p.key_env for p in PROVIDERS.values() if p.key_env})
```

In `waku/ops/static/js/views.js` line 273-276, wrap the per-provider field so subscription providers show a status line instead of a password input:

```javascript
      ${st.providers.map(p=>p.subscription
        ?`<label class="fld"><span>${p.name} <span class="meta">Claude subscription — sign in once with "claude login", no key</span>
          ${p.key_set?`<span class="srcpill" style="background:var(--good-soft);color:var(--good)">Agent SDK installed</span>`
                     :`<span class="srcpill">not installed — pip install -e '.[claude-code]'</span>`}</span></label>`
        :`<label class="fld"><span>${p.name} key <span class="meta">(${p.key_env})</span>
        ${p.key_set?`<span class="srcpill" style="background:var(--good-soft);color:var(--good)">set ····${esc(p.key_last4)}</span>`
                   :`<span class="srcpill">not set</span>`}</span>
        <input type="password" data-key="${p.key_env}" placeholder="${p.key_set?"key on file — blank keeps it":"paste key"}"></label>`).join("")}
```

(Keep the non-subscription branch byte-identical to the current lines 273-276 apart from the wrapping ternary — read the file first; the `not set` pill text above must match what is currently on line 275.)

- [ ] **Step 4: Run the dashboard-touching tests**

Run: `python -m pytest evals/deterministic/test_sdk_agent.py evals/deterministic/test_providers.py evals/deterministic/test_pinned_models.py evals/deterministic/test_static_assets.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add waku/ops/dashboard.py waku/ops/static/js/views.js evals/deterministic/test_sdk_agent.py
git commit -m "feat: dashboard understands a keyless subscription provider

settings_info flags claude-code as subscription (key_set = SDK importable),
the API-keys panel shows install/login status instead of a password field,
default pins include it when the SDK is present, and the .env whitelist
skips empty key envs."
```

---

### Task 7: Packaging + docs — extra, .env.example, README

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)
- Modify: `.env.example` (provider header block)
- Modify: `README.md` (provider setup section — find with `grep -n "provider" README.md`)

**Interfaces:**
- Consumes: everything above; this task is docs/packaging only, no code.

- [ ] **Step 1: Add the extra to `pyproject.toml`**

After the `notion` extra:

```toml
# Optional provider: run Waku on your Claude subscription (Max/Pro) via the
# Claude Agent SDK — sign in once with `claude login`, no API key needed.
claude-code = [
    "claude-agent-sdk>=0.1",
]
```

- [ ] **Step 2: Document in `.env.example`**

In the Provider header block, change the provider list line to include `claude-code` and add below the anthropic key line:

```bash
#   anthropic (default) · claude-code (subscription) · openai · gemini · deepseek · minimax · kimi · glm · openrouter
```

```bash
# claude-code → no key. Runs on your Claude subscription via the Agent SDK:
#   pip install -e '.[claude-code]'   then sign in once:   claude login
# (If ANTHROPIC_API_KEY is set it bills that key instead — leave it empty.)
# WAKU_PROVIDER=claude-code
```

- [ ] **Step 3: Add a short README subsection**

Next to the existing provider setup prose, add (neutral framing, no ranking):

```markdown
### Using a Claude subscription instead of an API key

If you have a Claude Max or Pro subscription, Waku can run through the
Claude Agent SDK instead of a metered API key:

    pip install -e '.[claude-code]'
    claude login          # once — signs the Agent SDK into your subscription
    WAKU_PROVIDER=claude-code make run

In this mode the Agent SDK runs the agent loop (see `waku/loop/sdk_agent.py`
— a readable counterpart to `waku/loop/agent.py`); Waku's tools, memory,
tracing, and dashboard all work unchanged. Two honest limitations:
`WAKU_MAX_TOKENS` is not enforced (the SDK has no per-call output cap) and
streaming arrives in per-message chunks rather than per-token.
```

- [ ] **Step 4: Verify docs are consistent**

Run: `grep -n "claude-code" .env.example README.md pyproject.toml`
Expected: all three mention it; `make lint` passes.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example README.md
git commit -m "docs: claude-code extra + subscription setup instructions

The Agent SDK dependency stays out of core behind the [claude-code] extra;
.env.example and README show the two-step setup (install extra, claude
login) and state the mode's real limitations."
```

---

### Task 8: Gate, live smoke test, ship

**Files:** none new — verification and shipping.

- [ ] **Step 1: Full deterministic suite + lint**

Run: `make lint && make eval`
Expected: PASS. Fix any fallout before proceeding.

- [ ] **Step 2: Release gate**

Run: `make gate`
Expected: deterministic gate PASS (judge evals run only if a key is present — fine either way).

- [ ] **Step 3: Live smoke test (only if this machine has Claude Code logged in AND the real SDK installed)**

```bash
pip install -e '.[claude-code]' && env -u ANTHROPIC_API_KEY WAKU_PROVIDER=claude-code \
  python -c "
from waku.app import Waku
from waku.config import load_settings
s = load_settings(); s.provider = 'claude-code'
w = Waku(settings=s)
print(w.respond('Say OK and nothing else.').reply)"
```

Expected: prints a reply containing `OK`. If the machine has no login, skip and note it in the PR body — the offline suite is the merge gate.

- [ ] **Step 4: Push the branch and open a draft PR**

```bash
git push -u origin worktree-agent-sdk-provider
gh pr create --draft --title "feat: claude-code subscription provider via the Agent SDK" \
  --body "Adds a keyless 'claude-code' provider: the Claude Agent SDK runs the turn under a Claude Max/Pro subscription (claude login), with Waku's tools as in-process MCP tools and full observer/LoopResult parity (tracing, dashboard, telemetry unchanged). Memory's gate/summarizer calls ride the same subscription via one-shot SDK queries. SDK dependency is behind the [claude-code] extra; deterministic tests run against a fake SDK, fully offline. Known limitations (documented): WAKU_MAX_TOKENS not enforced in this mode; streaming is per-message."
```

Expected: draft PR URL printed.
