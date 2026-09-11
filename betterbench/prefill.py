"""Synthetic long-input prompts for the prompt-processing (prefill) sweep.

We target a token depth by building filler text at ~4 chars/token; the *actual*
depth is read back from the server's `usage.prompt_tokens`, so the reported
throughput is always tied to the real token count, not our estimate.

The filler has to be unique, and not only at the front. A nonce prefix defeats
a *prefix* cache, whose block hashes are chained from the start of the prompt,
and that was all this module used to do: the body after the nonce was one
paragraph repeated to length. That body is pathologically repetitive — at a
16-token block size an 80k-character prompt is 918 blocks with **25 distinct
values among them**. Any cache that keys on block content rather than on a
chained prefix — a per-layer LRU over KV blocks, say — therefore hits on
essentially every block, both within a single pass and across every pass of the
sweep, and the sweep measures cache lookups instead of prompt processing.

So the body is now a fresh random ordering of the same words on every pass.
Shuffling rather than inventing text is deliberate: the word multiset, and with
it the characters-per-token ratio, is exactly the one this module has always
produced, so depths do not move — a re-ordering is invisible to the tokenizer's
statistics but total to a block hash. Measured on Qwen3's tokenizer, the same
80k characters go from 25 distinct blocks to 917 of 917, and two passes share
none.

Prefill cost is a function of token count alone, so scrambled word order costs
the server nothing it would not have spent on prose.

`nonce=None` asks for the old byte-identical prompt — a cacheable prompt on
purpose, for measuring a warm cache deliberately (`unique_nonce: false`).
"""
from __future__ import annotations

import random

_PARA = (
    "In distributed systems the tension between consistency, availability, and "
    "partition tolerance shapes almost every design decision. A service that "
    "prioritizes strong consistency may reject writes during a network split, "
    "while an available-first design accepts them and reconciles later. Caches, "
    "replication logs, quorums, and vector clocks are the everyday tools used to "
    "navigate these trade-offs, and the right choice depends on the workload, the "
    "cost of a stale read, and how users perceive latency. "
)

# Whitespace-split, so punctuation stays attached to the word it belongs to and
# the shuffled stream keeps the source's exact word *and* punctuation frequencies.
_WORDS = _PARA.split()
_MEAN_WORD_CHARS = sum(len(w) for w in _WORDS) / len(_WORDS) + 1   # + the space


def _filler(approx_chars: int, seed: str | None) -> str:
    """`approx_chars` of filler: a fresh word order per seed, or the repeated
    paragraph when `seed` is None."""
    if seed is None:
        return (_PARA * (approx_chars // len(_PARA) + 1))[:approx_chars]
    rng = random.Random(seed)
    # Drawn in one C-level call rather than a Python loop: a 250k-token depth is
    # over a megabyte of filler, rebuilt for every pass of the sweep.
    k = int(approx_chars / _MEAN_WORD_CHARS) + 16
    text = " ".join(rng.choices(_WORDS, k=k))
    while len(text) < approx_chars:            # short draw: top up
        text += " " + " ".join(rng.choices(_WORDS, k=256))
    return text[:approx_chars]


def make_prefill_messages(target_tokens: int, nonce: str | None) -> list[dict]:
    approx_chars = max(64, int(target_tokens * 4))
    body = _filler(approx_chars, nonce)
    tag = f"[bb:{nonce}] " if nonce else ""
    content = (f"{tag}Read the following context, then reply with the single "
               f"word: ack.\n\n{body}\n\nReply with one word: ack.")
    return [{"role": "user", "content": content}]
