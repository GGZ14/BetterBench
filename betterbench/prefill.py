"""Synthetic long-input prompts for the prompt-processing (prefill) sweep.

We target a token depth by building filler text at ~4 chars/token; the *actual*
depth is read back from the server's `usage.prompt_tokens`, so the reported
throughput is always tied to the real token count, not our estimate. A unique
nonce prefix defeats the prefix cache so prefill timing is honest.
"""
from __future__ import annotations

_PARA = (
    "In distributed systems the tension between consistency, availability, and "
    "partition tolerance shapes almost every design decision. A service that "
    "prioritizes strong consistency may reject writes during a network split, "
    "while an available-first design accepts them and reconciles later. Caches, "
    "replication logs, quorums, and vector clocks are the everyday tools used to "
    "navigate these trade-offs, and the right choice depends on the workload, the "
    "cost of a stale read, and how users perceive latency. "
)


def make_prefill_messages(target_tokens: int, nonce: str) -> list[dict]:
    approx_chars = max(64, int(target_tokens * 4))
    reps = approx_chars // len(_PARA) + 1
    body = (_PARA * reps)[:approx_chars]
    content = (f"[bb:{nonce}] Read the following context, then reply with the single "
               f"word: ack.\n\n{body}\n\nReply with one word: ack.")
    return [{"role": "user", "content": content}]
