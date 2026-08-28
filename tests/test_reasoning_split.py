"""Thinking vs answering — and, above all, refusing to guess.

Qwen3's chat template opens `<think>` in the prompt, so a run truncated at
max_tokens emits no marker at all. Recording that as "all answer" credits an
answer that never arrived.
"""
from __future__ import annotations

import pytest

from betterbench.client import stream_chat_sync

MSGS = [{"role": "user", "content": "hello"}]


def _run(base, max_tokens=40):
    r = stream_chat_sync(base, "mock", MSGS, max_tokens=max_tokens, temperature=0.0)
    assert r.ok, r.error
    return r


def test_reasoning_channel_splits_and_times_the_answer(server):
    base = server(ttft_ms=5.0, itl_ms=5.0, tokens=40,
                  reasoning="channel", reasoning_tokens=20)
    r = _run(base)
    assert r.reasoning_source == "channel"
    assert r.reasoning_tokens_est == pytest.approx(20, abs=2)
    assert r.answer_tokens_est == pytest.approx(20, abs=2)
    # TTFA is the wait people actually feel on a thinking model, and it is
    # necessarily later than TTFT — which only sees the first thinking token.
    assert r.ttfa_ms > r.ttft_ms


def test_inline_marker_splits_the_same_way(server):
    base = server(ttft_ms=5.0, itl_ms=5.0, tokens=40,
                  reasoning="inline", reasoning_tokens=20)
    r = _run(base)
    assert r.reasoning_source == "inline_marker"
    assert r.reasoning_tokens_est == pytest.approx(20, abs=3)
    assert r.ttfa_ms > r.ttft_ms


def test_no_thinking_at_all_is_none_and_ttfa_equals_ttft(server):
    base = server(ttft_ms=5.0, itl_ms=5.0, tokens=40)
    r = _run(base)
    assert r.reasoning_source == "none"
    assert r.reasoning_tokens_est == 0
    assert r.ttfa_ms == pytest.approx(r.ttft_ms)


def test_truncated_before_any_marker_is_unknown_never_zero(server):
    """The prompt opened <think>; max_tokens hit before </think>. We cannot
    tell thinking from answering, so we must not claim either."""
    base = server(ttft_ms=5.0, itl_ms=2.0, tokens=30, finish_reason="length")
    r = _run(base, max_tokens=30)
    assert r.reasoning_source == "unknown"
    assert r.reasoning_tokens_est is None
    assert r.answer_tokens_est is None
    assert r.ttfa_ms is None
    # The flattering failure mode, pinned explicitly: it must not be zero.
    assert r.reasoning_tokens_est is not 0        # noqa: F632  (intentional)
    assert r.answer_chars is None


def test_truncated_inside_the_thinking_channel_is_a_real_zero(server):
    """Here we *did* see the reasoning channel, so we know the answer never
    started — that is a measured zero, not an unknown."""
    base = server(ttft_ms=5.0, itl_ms=2.0, tokens=30, reasoning="channel",
                  reasoning_tokens=30, finish_reason="length")
    r = _run(base, max_tokens=30)
    assert r.reasoning_source == "channel"
    assert r.truncated_in_reasoning is True
    assert r.answer_tokens_est == 0
    assert r.ttfa_ms is None


def test_marker_split_across_two_updates_is_still_found(server):
    """`</think>` can straddle an SSE boundary; the carry buffer catches it."""
    base = server(ttft_ms=5.0, itl_ms=2.0, tokens=40, tokens_per_chunk=1,
                  reasoning="inline", reasoning_tokens=20)
    r = _run(base)
    assert r.reasoning_source == "inline_marker"
    assert r.answer_chars > 0
