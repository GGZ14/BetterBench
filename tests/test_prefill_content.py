"""The prefill filler must be unique all the way through, not just at the front.

A nonce prefix defeats a prefix cache, whose block hashes chain from the start
of the prompt. It does nothing about a cache that keys on block *content* — a
per-layer LRU over KV blocks — which the old repeated-paragraph body hit on
almost every block, within one pass and across every pass of the sweep.

These tests work in words rather than tokens: a KV block is ~16 tokens, which
is ~11 words, so a window of 12 words is a conservative stand-in for a block
and needs no tokenizer to check.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from betterbench.config import Config
from betterbench.prefill import _PARA, _filler, make_prefill_messages

WINDOW = 12


def _windows(text: str, n: int = WINDOW) -> list[tuple[str, ...]]:
    w = text.split()
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


# --------------------------------------------------------------------------- #
# What regressed: the old body, kept reachable as the deliberate warm-cache path
# --------------------------------------------------------------------------- #
def test_the_repeated_body_is_the_pathology_this_fixes():
    """Characterisation. `nonce=None` still asks for it — deliberately, for
    measuring a warm cache — so this documents what that choice costs."""
    body = _filler(40_000, None)
    win = _windows(body)
    assert len(set(win)) < len(win) / 50        # >98% of windows are duplicates
    assert _filler(40_000, None) == body        # and identical on the next pass


# --------------------------------------------------------------------------- #
# What it does now
# --------------------------------------------------------------------------- #
def test_no_window_repeats_within_one_pass():
    win = _windows(_filler(40_000, "pass-1"))
    assert len(set(win)) == len(win)


def test_two_passes_share_no_window():
    a, b = _windows(_filler(40_000, "pass-1")), _windows(_filler(40_000, "pass-2"))
    assert set(a).isdisjoint(b)


def test_the_body_is_a_reordering_of_the_same_words():
    """Not new text: the same word multiset, so the characters-per-token ratio
    — and therefore the depth a target maps to — does not move. The last word
    may be clipped by the exact-length slice, as it always has been."""
    words = _filler(40_000, "seed").split()
    assert set(words[:-1]) <= set(_PARA.split())


def test_the_filler_is_exactly_as_long_as_it_used_to_be():
    """Character length is the lever the depth target pulls. If a re-ordering
    changed it, every recorded depth would shift with it."""
    for n in (64, 1_000, 40_000):
        assert len(_filler(n, "seed")) == len(_filler(n, None)) == n


def test_the_same_seed_rebuilds_the_same_body():
    assert _filler(10_000, "abc") == _filler(10_000, "abc")
    assert _filler(10_000, "abc") != _filler(10_000, "abd")


# --------------------------------------------------------------------------- #
# The whole prompt
# --------------------------------------------------------------------------- #
def test_the_prompt_still_carries_the_nonce_tag_for_prefix_caches():
    assert make_prefill_messages(2_000, "beef")[0]["content"].startswith("[bb:beef] ")


def test_two_prompts_at_the_same_depth_differ_beyond_the_tag():
    a = make_prefill_messages(8_000, "n1")[0]["content"]
    b = make_prefill_messages(8_000, "n2")[0]["content"]
    assert a != b
    assert a[len("[bb:n1] "):] != b[len("[bb:n2] "):]     # not just the tag


def test_no_nonce_means_a_byte_identical_prompt():
    a = make_prefill_messages(8_000, None)[0]["content"]
    assert a == make_prefill_messages(8_000, None)[0]["content"]
    assert not a.startswith("[bb:")


def test_a_tiny_depth_still_builds(server):
    assert make_prefill_messages(1, "n")[0]["content"]


# --------------------------------------------------------------------------- #
# Wiring: the sweep honours unique_nonce
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("unique,distinct_bodies", [(True, True), (False, False)])
def test_the_sweep_reseeds_every_pass_unless_told_not_to(server, monkeypatch,
                                                         unique, distinct_bodies):
    """unique_nonce=False is the deliberate warm-cache measurement; the default
    must send a body no block of which the server has seen before."""
    import betterbench.runner as runner

    sent = []
    real = runner.make_prefill_messages

    def record(depth, nonce):
        msgs = real(depth, nonce)
        sent.append(msgs[0]["content"])
        return msgs

    monkeypatch.setattr(runner, "make_prefill_messages", record)
    cfg = Config(prefill_depths=[300], prefill_runs=3, prefill_warmup=0,
                 unique_nonce=unique)
    asyncio.run(runner.prefill_sweep(server(), "mock", cfg, log=lambda *a: None))

    assert len(sent) == 3
    assert (len(set(sent)) == len(sent)) is distinct_bodies
