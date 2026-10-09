"""Two invocations of the prefill sweep must not send byte-identical prompts.

A tiered KV cache that persists to RAM or SSD would serve a replayed prompt
back from a previous run, and the sweep would time cache lookups while
reporting prompt processing. The nonce RNG is therefore seeded from entropy.
"""
from __future__ import annotations

import asyncio

from betterbench import runner
from betterbench.client import RunResult
from betterbench.config import Config

CFG = dict(prefill_depths=[200], prefill_runs=2, prefill_warmup=0)


def _prompts(monkeypatch):
    sent: list[list[dict]] = []

    async def capture(endpoint, model, msgs, **kw):
        sent.append(msgs)
        return RunResult(ok=True, category="prefill", prompt_id="pp",
                         ttft_ms=1.0, prompt_tokens=10, completion_tokens=1)

    monkeypatch.setattr(runner, "stream_chat", capture)
    asyncio.run(runner.prefill_sweep("http://x/v1", "m", Config(**CFG),
                                     log=lambda *a: None))
    assert len(sent) == 2
    return sent


def test_two_invocations_share_no_prompt(monkeypatch):
    first, second = _prompts(monkeypatch), _prompts(monkeypatch)
    assert not any(a == b for a in first for b in second)


def test_a_run_is_still_unique_within_itself(monkeypatch):
    a, b = _prompts(monkeypatch)
    assert a != b
