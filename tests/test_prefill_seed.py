"""Two invocations of the prefill sweep must not send byte-identical prompts.

The depth sweep's nonce RNG is seeded so a single run is reproducible. That is
right for one run and wrong across two: a lane that keeps its prefix cache warm
serves the second invocation entirely from cache, and the sweep times cache
lookups while reporting prompt processing. (Measured: an 8k-token depth read
153,900 t/s — a 100 % cache hit rate.)

`BETTERBENCH_PREFILL_SEED` is the escape hatch: set it per invocation and the prompts
change. Unset, behaviour — and comparability with older results — is unchanged.
"""
from __future__ import annotations

import asyncio

from betterbench import runner
from betterbench.client import RunResult
from betterbench.config import Config

CFG = dict(prefill_depths=[200], prefill_runs=2, prefill_warmup=0)


def _prompts(monkeypatch, seed_env):
    """Run the sweep against a stub transport; return the messages it sent."""
    sent: list[list[dict]] = []

    async def capture(endpoint, model, msgs, **kw):
        sent.append(msgs)
        return RunResult(ok=True, category="prefill", prompt_id="pp",
                         ttft_ms=1.0, prompt_tokens=10, completion_tokens=1)

    monkeypatch.setattr(runner, "stream_chat", capture)
    if seed_env is None:
        monkeypatch.delenv("BETTERBENCH_PREFILL_SEED", raising=False)
    else:
        monkeypatch.setenv("BETTERBENCH_PREFILL_SEED", str(seed_env))
    asyncio.run(runner.prefill_sweep("http://x/v1", "m", Config(**CFG),
                                     log=lambda *a: None))
    assert len(sent) == 2, "the sweep should have issued both passes"
    return sent


def test_unset_seed_keeps_a_run_reproducible(monkeypatch):
    """Historical behaviour, and why the default cannot simply move."""
    assert _prompts(monkeypatch, None) == _prompts(monkeypatch, None)


def test_the_first_pass_is_the_pathology(monkeypatch):
    """Characterisation. Within a run 0.6.0 reshuffles every pass, but the FIRST
    prompt of two invocations is byte-identical under the default seed — so a
    lane that kept its cache warm between runs answers that pass from cache.
    This is what the variable exists to break."""
    assert _prompts(monkeypatch, None)[0] == _prompts(monkeypatch, None)[0]


def test_a_fresh_seed_changes_what_the_lane_is_asked(monkeypatch):
    first = _prompts(monkeypatch, 1)
    second = _prompts(monkeypatch, 2)
    assert first[0] != second[0]
