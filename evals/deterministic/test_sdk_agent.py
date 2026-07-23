"""OFFLINE checks for the subscription loop (waku/loop/sdk_agent.py).

The real claude-agent-sdk needs a Claude Code login, so these tests inject a
fake `claude_agent_sdk` module into sys.modules — the deterministic suite
stays keyless and offline, like every other provider test."""

from __future__ import annotations

import types


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
