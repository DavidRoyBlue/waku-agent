"""OFFLINE checks for the subscription loop (waku/loop/sdk_agent.py).

The real claude-agent-sdk needs a Claude Code login, so these tests inject a
fake `claude_agent_sdk` module into sys.modules — the deterministic suite
stays keyless and offline, like every other provider test."""

from __future__ import annotations

import importlib.machinery
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


@pytest.fixture()
def fake_sdk(monkeypatch):
    """Install a stub claude_agent_sdk. Configure it via mod.script — a list of
    messages query() will yield. ToolUseBlock entries are not yielded; instead
    the stub invokes the matching registered MCP tool handler, like the real
    SDK's inner loop does."""
    mod = types.ModuleType("claude_agent_sdk")
    # a real ModuleSpec so importlib.util.find_spec() works on the stub —
    # a bare ModuleType has __spec__=None, which find_spec raises ValueError on
    mod.__spec__ = importlib.machinery.ModuleSpec("claude_agent_sdk", None)

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
    # a leftover ANTHROPIC_API_KEY must be blanked for the SDK subprocess —
    # claude-code mode always runs on the subscription, never a stray key
    assert fake_sdk.last_options.env == {"ANTHROPIC_API_KEY": ""}


def test_client_has_no_stream_attribute(fake_sdk):
    from waku.loop.sdk_agent import ClaudeAgentClient

    assert not hasattr(ClaudeAgentClient().messages, "stream")


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
    assert options.env == {"ANTHROPIC_API_KEY": ""}   # subscription, never a stray key


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


def test_autodetects_subscription_when_no_key_and_sdk_installed(fake_sdk, monkeypatch):
    """Installing the extra IS the opt-in: default provider + no key anywhere
    + SDK importable -> claude-code, with a stderr note."""
    from waku.config import Settings
    from waku.loop.models import get_client
    from waku.loop.sdk_agent import ClaudeAgentClient

    monkeypatch.delenv("WAKU_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(provider="anthropic", model="", small_model="",
                        api_key="", base_url=None)
    assert isinstance(get_client(settings), ClaudeAgentClient)
    assert settings.provider == "claude-code"


def test_no_autodetect_when_a_key_or_explicit_provider_exists(fake_sdk, monkeypatch):
    import anthropic
    from waku.config import Settings
    from waku.loop.models import get_client

    # a key present -> the API path is untouched
    monkeypatch.delenv("WAKU_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-tests")
    settings = Settings(provider="anthropic", model="", small_model="",
                        api_key="", base_url=None)
    assert isinstance(get_client(settings), anthropic.Anthropic)
    assert settings.provider == "anthropic"

    # an explicit WAKU_PROVIDER=anthropic without a key -> still exits, no reroute
    monkeypatch.setenv("WAKU_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(provider="anthropic", model="", small_model="",
                        api_key="", base_url=None)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        get_client(settings)


def test_missing_anthropic_key_error_mentions_the_subscription_path(monkeypatch):
    from waku.config import Settings
    from waku.loop.models import get_client

    monkeypatch.setenv("WAKU_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(provider="anthropic", model="", small_model="",
                        api_key="", base_url=None)
    with pytest.raises(SystemExit, match=r"claude-code"):
        get_client(settings)


def test_price_for_subscription_is_zero():
    """price_for must short-circuit BEFORE the per-model rate table — the
    default claude models all exist in MODEL_PRICING, and subscription turns
    must never show dollar spend in the arena or the usage ledger."""
    from waku.ops.dashboard import price_for

    assert price_for("claude-code", "claude-sonnet-5") == (0.0, 0.0)
    assert price_for("claude-code", "claude-opus-4-8") == (0.0, 0.0)


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
